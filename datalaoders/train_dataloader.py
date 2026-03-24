# datalaoders/train_dataloader.py
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow as pa
import pyarrow.parquet as pq


def _rank_world(num_gpus_fallback: int = 1):
    """
    Works for:
      - Lightning ddp spawn/subprocess (LOCAL_RANK/RANK/WORLD_SIZE set)
      - Slurm (SLURM_LOCALID/SLURM_PROCID/SLURM_NTASKS set)
    """
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", str(local_rank))))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", str(num_gpus_fallback))))
    return rank, world, local_rank


def _read_parquet_row_slice(parquet_path: str, columns, row_start: int, row_end: int) -> pa.Table:
    """
    Read ONLY rows [row_start, row_end) from a parquet file by reading overlapping row-groups.
    """
    pf = pq.ParquetFile(parquet_path, memory_map=True)
    total = pf.metadata.num_rows
    row_start = max(0, min(row_start, total))
    row_end = max(0, min(row_end, total))
    if row_end <= row_start:
        return pa.table({c: pa.array([]) for c in columns})

    tables = []
    base = 0
    for rg in range(pf.num_row_groups):
        rg_n = pf.metadata.row_group(rg).num_rows
        rg_start = base
        rg_end = base + rg_n

        if rg_end <= row_start:
            base = rg_end
            continue
        if rg_start >= row_end:
            break

        t = pf.read_row_group(rg, columns=columns, use_threads=True)

        s = max(0, row_start - rg_start)
        e = min(rg_n, row_end - rg_start)
        t = t.slice(s, e - s)
        if t.num_rows > 0:
            tables.append(t)

        base = rg_end

    if not tables:
        return pa.table({c: pa.array([]) for c in columns})
    return pa.concat_tables(tables, promote=True) if len(tables) > 1 else tables[0]


class PHMDataset(Dataset):
    """
    Parquet schema:
      - samples : FixedSizeList[float32] length = L*C  (flattened from (L,C))
      - labels  : FixedSizeList[float32] length = H*C  (flattened from (H,C))
      - orig_row: int64 (optional, recommended)

    Output:
      X: (N, C, L)
      Y: (N, C, H)
      orig_row: (N,)
    """
    def __init__(self, args, data_type: str):
        super().__init__()

        # ---- resolve file path ----
        if args.data_percentage == "100" or data_type != "train":
            data_path = os.path.join(args.data_path, args.data_id, f"{data_type}.parquet")
        elif data_type == "train" and "shot" in args.data_percentage:
            data_path = os.path.join(args.data_path, args.data_id, f"{data_type}.parquet")
        else:
            data_path = os.path.join(args.data_path, args.data_id, f"{data_type}_{args.data_percentage}p.parquet")

        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)

        C = int(getattr(args, "num_channels", 0))
        H = int(getattr(args, "horizon", 1))
        if C <= 0:
            raise ValueError("Set --num_channels (e.g., 11).")
        if H <= 0:
            raise ValueError("Set --horizon (e.g., 50).")

        shard_by_rank = bool(getattr(args, "shard_by_rank", True))
        rank, world, _ = _rank_world(num_gpus_fallback=int(getattr(args, "num_gpus", 1)))

        # ---- find total rows cheaply ----
        pf = pq.ParquetFile(data_path, memory_map=True)
        N_total = pf.metadata.num_rows

        # ---- choose per-rank slice (equal lengths across ranks) ----
        if shard_by_rank and world > 1:
            n_per = N_total // world
            start = rank * n_per
            end = start + n_per
        else:
            start, end = 0, N_total

        cols = ["samples"]
        if "labels" in pf.schema_arrow.names:
            cols.append("labels")
        if "orig_row" in pf.schema_arrow.names:
            cols.append("orig_row")

        tbl = _read_parquet_row_slice(data_path, cols, start, end)
        if tbl.num_rows == 0:
            raise RuntimeError(f"[{data_type}] got 0 rows after slicing: {data_path} start={start} end={end}")

        # ---- samples ----
        samples = tbl["samples"].combine_chunks()  # safe: only shard portion
        list_size = samples.type.list_size  # = L*C
        if list_size % C != 0:
            raise ValueError(f"[{data_type}] samples list_size={list_size} not divisible by C={C}")
        L = list_size // C

        x_vals = samples.values.to_numpy(zero_copy_only=False)
        x_vals = x_vals.astype(np.float32, copy=False)
        x_np = x_vals.reshape(tbl.num_rows, L, C).transpose(0, 2, 1)  # (N,C,L)
        self.x_data = torch.from_numpy(x_np)

        # ---- labels ----
        self.y_data = None
        if "labels" in tbl.column_names:
            labels = tbl["labels"].combine_chunks()
            y_list_size = labels.type.list_size
            expected = H * C
            if y_list_size != expected:
                raise ValueError(f"[{data_type}] labels list_size={y_list_size} expected={expected} (H={H},C={C})")

            y_vals = labels.values.to_numpy(zero_copy_only=False)
            y_vals = y_vals.astype(np.float32, copy=False)
            y_np = y_vals.reshape(tbl.num_rows, H, C).transpose(0, 2, 1)  # (N, C, H)
            self.y_data = torch.from_numpy(y_np)

        # ---- orig_row ----
        self.orig_row = None
        if "orig_row" in tbl.column_names:
            self.orig_row = tbl["orig_row"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        else:
            # fallback: recover absolute row index in the original split file
            # NOTE: this is NOT the original "one_step_ws1024.parquet" row unless you wrote orig_row earlier.
            self.orig_row = np.arange(start, start + tbl.num_rows, dtype=np.int64)

        self.len = self.x_data.shape[0]

        if rank == 0:
            print(f"[INFO] {data_type}: total_rows={N_total} shard_by_rank={shard_by_rank} world={world}")
        print(f"[{data_type}] rank={rank}/{world} rows={self.len} x={tuple(self.x_data.shape)} "
              f"y={None if self.y_data is None else tuple(self.y_data.shape)} file={data_path}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.y_data is None:
            return self.x_data[idx]
        return self.x_data[idx], self.y_data[idx], torch.tensor(self.orig_row[idx], dtype=torch.long)


def _make_loader(ds, batch_size, shuffle, drop_last, num_workers, pin_memory):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=(int(num_workers) > 0),
    )


def get_train_val_loaders(args):
    train_ds = PHMDataset(args, "train")
    val_ds = PHMDataset(args, "val")
    return (
        _make_loader(train_ds, args.batch_size, shuffle=True,  drop_last=True,
                     num_workers=getattr(args, "num_workers", 0), pin_memory=getattr(args, "pin_memory", True)),
        _make_loader(val_ds,   args.batch_size, shuffle=False, drop_last=False,
                     num_workers=getattr(args, "num_workers", 0), pin_memory=getattr(args, "pin_memory", True)),
    )


def get_test_loader(args):
    test_ds = PHMDataset(args, "test")
    return _make_loader(test_ds, args.batch_size, shuffle=False, drop_last=False,
                        num_workers=getattr(args, "num_workers", 0), pin_memory=getattr(args, "pin_memory", True))


# Backward-compatible signature
def get_datasets(args):
    train_loader, val_loader = get_train_val_loaders(args)
    test_loader = get_test_loader(args)
    return train_loader, val_loader, test_loader