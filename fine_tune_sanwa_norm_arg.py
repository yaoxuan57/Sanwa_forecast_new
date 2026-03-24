# # fine_tune_sanwa_norm_arg.py
# import os
# import argparse
# import datetime
# import math
# import copy
# import numpy as np

# import torch
# import torch.nn.functional as F
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar

# from datalaoders.train_dataloader import get_train_val_loaders, get_test_loader
# from model.model import Transformer_bkbone
# from utils import save_copy_of_files, str2bool


# # -------------------------
# # helpers / configs
# # -------------------------
# def apply_model_config(args):
#     cfg = {
#         "tiny":  {"embed_dim": 128, "heads": 4,  "depth": 4},
#         "small": {"embed_dim": 256, "heads": 8,  "depth": 8},
#         "base":  {"embed_dim": 512, "heads": 12, "depth": 16},
#     }[args.model_type]
#     for k, v in cfg.items():
#         setattr(args, k, v)


# def construct_experiment_dir(args):
#     ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     run = "FT" if args.load_from_pretrained else "Supervised"
#     run += f"_{args.model_type}"
#     run += f"_{args.data_id}_from{args.pretraining_epoch_id}_{args.model_id}"
#     run += f"_bs{args.batch_size}_lr{args.lr}_seed{args.random_seed}"
#     run += f"_xnorm{args.x_norm}_ynorm{args.y_norm}_H{args.horizon}_{ts}"
#     return run


# # -------------------------
# # Normalizers
# # -------------------------
# class XNormalizer:
#     def __init__(self, mode="none", eps=1e-6):
#         self.mode = mode
#         self.eps = eps
#         self.stats = {}

#     def fit(self, x_train: torch.Tensor):
#         x = x_train.float()
#         if self.mode == "none":
#             self.stats = {}
#             return self
#         if self.mode in ("minmax01", "minmax11"):
#             x_min = x.amin(dim=(0, 2), keepdim=True)
#             x_max = x.amax(dim=(0, 2), keepdim=True)
#             scale = torch.clamp(x_max - x_min, min=self.eps)
#             self.stats = {"x_min": x_min.cpu(), "x_scale": scale.cpu()}
#             return self
#         if self.mode == "zscore":
#             mean = x.mean(dim=(0, 2), keepdim=True)
#             std = x.std(dim=(0, 2), keepdim=True, unbiased=False)
#             std = torch.clamp(std, min=self.eps)
#             self.stats = {"x_mean": mean.cpu(), "x_std": std.cpu()}
#             return self
#         raise ValueError(self.mode)

#     def apply(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.float()
#         if self.mode == "none":
#             return x
#         if self.mode == "minmax01":
#             return (x - self.stats["x_min"].to(x.device)) / self.stats["x_scale"].to(x.device)
#         if self.mode == "minmax11":
#             return 2.0 * ((x - self.stats["x_min"].to(x.device)) / self.stats["x_scale"].to(x.device)) - 1.0
#         if self.mode == "zscore":
#             return (x - self.stats["x_mean"].to(x.device)) / self.stats["x_std"].to(x.device)
#         raise ValueError(self.mode)


# class YNormalizer:
#     """
#     y: (N,C,H) or (N,C)
#     stats computed per-channel across N*H
#     """
#     def __init__(self, mode="none", eps=1e-8):
#         self.mode = mode
#         self.eps = eps
#         self.stats = {}

#     def fit(self, y_train: torch.Tensor):
#         y = y_train.detach().cpu().float().numpy()
#         if y.ndim == 2:
#             y2 = y
#         elif y.ndim == 3:
#             y2 = y.transpose(0, 2, 1).reshape(-1, y.shape[1])  # (N*H, C)
#         else:
#             raise ValueError(y.shape)

#         if self.mode == "none":
#             self.stats = {}
#             return self

#         if self.mode in ("minmax01", "minmax11"):
#             y_min = y2.min(axis=0)
#             y_max = y2.max(axis=0)
#             y_scale = (y_max - y_min) + self.eps
#             self.stats = {
#                 "y_min": torch.tensor(y_min, dtype=torch.float32),
#                 "y_scale": torch.tensor(y_scale, dtype=torch.float32),
#             }
#             return self

#         if self.mode == "zscore":
#             mean = y2.mean(axis=0)
#             std = y2.std(axis=0) + self.eps
#             self.stats = {
#                 "y_mean": torch.tensor(mean, dtype=torch.float32),
#                 "y_std": torch.tensor(std, dtype=torch.float32),
#             }
#             return self

#         raise ValueError(self.mode)

#     def apply(self, y: torch.Tensor) -> torch.Tensor:
#         y = y.float()
#         if self.mode == "none":
#             return y

#         def bcast(v):
#             return v[None, :, None] if y.ndim == 3 else v[None, :]

#         if self.mode == "minmax01":
#             y_min = self.stats["y_min"].to(y.device)
#             y_scale = self.stats["y_scale"].to(y.device)
#             return (y - bcast(y_min)) / bcast(y_scale)

#         if self.mode == "minmax11":
#             y_min = self.stats["y_min"].to(y.device)
#             y_scale = self.stats["y_scale"].to(y.device)
#             y01 = (y - bcast(y_min)) / bcast(y_scale)
#             return 2.0 * y01 - 1.0

#         if self.mode == "zscore":
#             mean = self.stats["y_mean"].to(y.device)
#             std = self.stats["y_std"].to(y.device)
#             return (y - bcast(mean)) / bcast(std)

#         raise ValueError(self.mode)

#     def invert(self, y: torch.Tensor) -> torch.Tensor:
#         y = y.float()
#         if self.mode == "none":
#             return y

#         def bcast(v):
#             return v[None, :, None] if y.ndim == 3 else v[None, :]

#         if self.mode == "minmax01":
#             y_min = self.stats["y_min"].to(y.device)
#             y_scale = self.stats["y_scale"].to(y.device)
#             return y * bcast(y_scale) + bcast(y_min)

#         if self.mode == "minmax11":
#             y_min = self.stats["y_min"].to(y.device)
#             y_scale = self.stats["y_scale"].to(y.device)
#             y01 = (y + 1.0) / 2.0
#             return y01 * bcast(y_scale) + bcast(y_min)

#         if self.mode == "zscore":
#             mean = self.stats["y_mean"].to(y.device)
#             std = self.stats["y_std"].to(y.device)
#             return y * bcast(std) + bcast(mean)

#         raise ValueError(self.mode)


# # -------------------------
# # Lightning module
# # -------------------------
# class Model(pl.LightningModule):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.net = Transformer_bkbone(args)
#         self.loss_beta = float(getattr(args, "huber_beta", 1.0))

#         self.total_steps = args.num_epochs * max(1, getattr(args, "tl_length", 1))
#         self.num_warmup_steps = int(0.1 * self.total_steps)

#         # buffers used ONLY in single-GPU test pass
#         self._test_pred = []
#         self._test_true = []
#         self._test_orig = []

#     def forward(self, x):
#         return self.net(x)

#     def configure_optimizers(self):
#         opt = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wt_decay)
#         sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: self._lr_lambda(step))
#         return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

#     def _lr_lambda(self, current_step):
#         if current_step < self.num_warmup_steps:
#             return float(current_step) / float(max(1, self.num_warmup_steps))
#         progress = float(current_step - self.num_warmup_steps) / float(max(1, self.total_steps - self.num_warmup_steps))
#         return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

#     def _rmse(self, pred, true):
#         return torch.sqrt(F.mse_loss(pred, true))

#     def _shared_step(self, batch, stage: str):
#         x, y, orig = batch  # our dataset always returns 3

#         feats = self(x)
#         pred = self.net.predict(feats).float()  # (B,C,H)
#         y = y.float()

#         loss = F.smooth_l1_loss(pred, y, beta=self.loss_beta)

#         # RMSE in original units
#         if getattr(self.args, "y_norm", "none") != "none" and hasattr(self.args, "y_norm_stats"):
#             yn: YNormalizer = self.args.y_norm_stats
#             pred_o = yn.invert(pred)
#             y_o = yn.invert(y)
#         else:
#             pred_o, y_o = pred, y

#         rmse = self._rmse(pred_o, y_o)

#         if stage == "train":
#             self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
#             self.log("train_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
#         elif stage == "val":
#             self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
#             self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
#         else:
#             self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=False)
#             self.log("test_rmse", rmse, on_step=False, on_epoch=True, sync_dist=False)

#             # ✅ ONLY save arrays when running single-GPU test pass
#             if self.trainer.world_size == 1:
#                 self._test_pred.append(pred_o.detach().cpu().numpy().astype(np.float32))
#                 self._test_true.append(y_o.detach().cpu().numpy().astype(np.float32))
#                 self._test_orig.append(orig.detach().cpu().numpy().astype(np.int64))

#         return loss

#     def training_step(self, batch, batch_idx):
#         return self._shared_step(batch, "train")

#     def validation_step(self, batch, batch_idx):
#         return self._shared_step(batch, "val")

#     def test_step(self, batch, batch_idx):
#         return self._shared_step(batch, "test")

#     def on_test_epoch_end(self):
#         # ✅ only meaningful for single-GPU test pass
#         if self.trainer.world_size != 1:
#             return
#         if len(self._test_pred) == 0:
#             print("[WARN] No test batches collected, nothing saved.")
#             return

#         pred = np.concatenate(self._test_pred, axis=0)  # (N,C,H)
#         true = np.concatenate(self._test_true, axis=0)
#         orig = np.concatenate(self._test_orig, axis=0)  # (N,)

#         order = np.argsort(orig)
#         pred, true, orig = pred[order], true[order], orig[order]

#         ckpt_dir = os.path.abspath(self.args.ckpt_dir)
#         os.makedirs(ckpt_dir, exist_ok=True)

#         np.save(os.path.join(ckpt_dir, "test_preds.npy"), pred)
#         np.save(os.path.join(ckpt_dir, "test_targets.npy"), true)
#         np.save(os.path.join(ckpt_dir, "test_orig_row.npy"), orig)

#         rmse_all = float(np.sqrt(np.mean((pred - true) ** 2)))
#         print(f"[OK] Saved FULL test outputs to {ckpt_dir}")
#         print(f"     pred={pred.shape} true={true.shape} orig={orig.shape} rmse={rmse_all:.6f}")

#         self._test_pred, self._test_true, self._test_orig = [], [], []


# def main(args):
#     pl.seed_everything(args.random_seed, workers=True)

#     # ---- DDP TRAINING LOADERS (sharded) ----
#     train_loader, val_loader = get_train_val_loaders(args)

#     # dataset-derived shapes
#     args.seq_len = train_loader.dataset.x_data.shape[-1]
#     args.num_channels = train_loader.dataset.x_data.shape[1]
#     args.tl_length = len(train_loader)

#     # ---- fit normalizers on TRAIN shard (fine for training stability) ----
#     x_normer = XNormalizer(mode=args.x_norm).fit(train_loader.dataset.x_data)
#     if args.x_norm != "none":
#         train_loader.dataset.x_data = x_normer.apply(train_loader.dataset.x_data)
#         val_loader.dataset.x_data = x_normer.apply(val_loader.dataset.x_data)

#     y_normer = YNormalizer(mode=args.y_norm).fit(train_loader.dataset.y_data)
#     if args.y_norm != "none":
#         train_loader.dataset.y_data = y_normer.apply(train_loader.dataset.y_data)
#         val_loader.dataset.y_data = y_normer.apply(val_loader.dataset.y_data)
#     args.y_norm_stats = y_normer

#     # ---- run dir ----
#     run = construct_experiment_dir(args)
#     ckpt_dir = os.path.abspath(os.path.join("checkpoints", run))
#     args.ckpt_dir = ckpt_dir
#     os.makedirs(ckpt_dir, exist_ok=True)
#     print(f"========== {run} ==========")

#     # save norm stats (rank0 only)
#     if int(os.environ.get("RANK", "0")) == 0:
#         stats = {"x_norm": args.x_norm, "y_norm": args.y_norm}
#         stats.update({k: v for k, v in x_normer.stats.items()})
#         stats.update({k: v for k, v in y_normer.stats.items()})
#         torch.save(stats, os.path.join(ckpt_dir, "norm_stats.pt"))
#         print("Saved norm stats:", os.path.join(ckpt_dir, "norm_stats.pt"))

#     # callbacks
#     checkpoint = ModelCheckpoint(monitor="val_rmse", mode="min", save_top_k=1, dirpath=ckpt_dir, filename="best")
#     early_stop = EarlyStopping(monitor="val_rmse", patience=args.patience, mode="min")
#     save_copy_of_files(checkpoint)

#     model = Model(args)

#     # pretrained load (map_location CPU to avoid cuda availability edge-cases)
#     if args.load_from_pretrained:
#         path = os.path.join(args.pretrained_model_dir, f"pretrain-epoch={args.pretraining_epoch_id}.ckpt")
#         ckpt = torch.load(path, map_location="cpu", weights_only=False)
#         ckpt_state = ckpt["state_dict"]
#         model_state = model.state_dict()
#         matched = {k: v for k, v in ckpt_state.items() if k in model_state and model_state[k].shape == v.shape}
#         model.load_state_dict(matched, strict=False)
#         print(f"Loaded pretrained: {path}  matched={len(matched)}/{len(model_state)}")

#     # ---- TRAIN with DDP (Lightning spawns processes) ----
#     trainer = pl.Trainer(
#         default_root_dir=ckpt_dir,
#         max_epochs=args.num_epochs,
#         callbacks=[checkpoint, early_stop, TQDMProgressBar(refresh_rate=200)],
#         accelerator="gpu",
#         devices=args.num_gpus if args.num_gpus > 1 else 1,
#         strategy="ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto",
#         precision="bf16-mixed",
#         num_sanity_val_steps=0,
#         use_distributed_sampler=False,  # IMPORTANT: dataset already sharded by rank
#     )

#     trainer.fit(model, train_loader, val_loader)

#     # ---- SINGLE-GPU TEST PASS (rank0 only) to save ALL preds/targets ----
#     if trainer.is_global_zero:
#         print("[INFO] Running single-GPU test pass to save FULL test_preds.npy ...")

#         test_args = copy.deepcopy(args)
#         test_args.num_gpus = 1
#         test_args.shard_by_rank = False          # IMPORTANT: load full test set
#         test_args.num_workers = 0                # keep memory stable
#         test_args.pin_memory = False

#         test_loader = get_test_loader(test_args)

#         if test_args.x_norm != "none":
#             test_loader.dataset.x_data = x_normer.apply(test_loader.dataset.x_data)

#         if test_args.y_norm != "none":
#             test_loader.dataset.y_data = y_normer.apply(test_loader.dataset.y_data)

#         # derive seq_len/num_channels for test model init
#         test_args.seq_len = test_loader.dataset.x_data.shape[-1]
#         test_args.num_channels = test_loader.dataset.x_data.shape[1]
#         test_args.tl_length = 1
#         test_args.y_norm_stats = args.y_norm_stats  # reuse fitted stats
#         test_args.ckpt_dir = ckpt_dir

#         best_path = checkpoint.best_model_path
#         test_model = Model.load_from_checkpoint(best_path, args=test_args, strict=False)

#         test_trainer = pl.Trainer(
#             accelerator="gpu",
#             devices=1,
#             precision="bf16-mixed",
#             logger=False,
#             enable_checkpointing=False,
#         )
#         test_trainer.test(test_model, test_loader)

#     print("[DONE] outputs at:", ckpt_dir)


# if __name__ == "__main__":
#     p = argparse.ArgumentParser()

#     p.add_argument("--data_path", type=str, required=True)
#     p.add_argument("--data_id", type=str, required=True)
#     p.add_argument("--data_percentage", type=str, default="1")
#     p.add_argument("--model_id", type=str, default="Sw_fc")

#     p.add_argument("--model_type", choices=["tiny", "small", "base"], default="tiny")
#     p.add_argument("--patch_size", type=int, default=64)
#     p.add_argument("--dropout", type=float, default=0.1)

#     p.add_argument("--num_epochs", type=int, default=200)
#     p.add_argument("--patience", type=int, default=50)
#     p.add_argument("--batch_size", type=int, default=8)
#     p.add_argument("--lr", type=float, default=1e-4)
#     p.add_argument("--wt_decay", type=float, default=1e-4)
#     p.add_argument("--random_seed", type=int, default=42)

#     p.add_argument("--task_type", type=str, default="RUL", choices=["FD", "RUL"])
#     p.add_argument("--num_classes", type=int, default=1)

#     p.add_argument("--num_channels", type=int, default=11)
#     p.add_argument("--horizon", type=int, default=50)
#     p.add_argument("--huber_beta", type=float, default=1.0)

#     p.add_argument("--x_norm", choices=["none", "minmax01", "minmax11", "zscore"], default="zscore")
#     p.add_argument("--y_norm", choices=["none", "minmax01", "minmax11", "zscore"], default="zscore")

#     p.add_argument("--num_gpus", type=int, default=1)
#     p.add_argument("--num_workers", type=int, default=0)
#     p.add_argument("--pin_memory", type=str2bool, default=False)

#     p.add_argument("--shard_by_rank", type=str2bool, default=True)

#     p.add_argument("--load_from_pretrained", type=str2bool, default=True)
#     p.add_argument("--pretrained_model_dir", type=str, required=True)
#     p.add_argument("--pretraining_epoch_id", type=int, default=1)
#     p.add_argument("--pretrained_model_type", choices=["normal", "mae"], default="normal")

#     args = p.parse_args()
#     apply_model_config(args)
#     main(args)

#!/usr/bin/env python3
# fine_tune_sanwa_norm_arg.py

# fine_tune_sanwa_norm_arg.py
import os
import time
import copy
import math
import argparse
import datetime
import numpy as np

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar

import pyarrow.parquet as pq

from datalaoders.train_dataloader import get_train_val_loaders, get_test_loader
from model.model import Transformer_bkbone
from utils import save_copy_of_files, str2bool


# -------------------------
# model config
# -------------------------
def apply_model_config(args):
    cfg = {
        "tiny":  {"embed_dim": 128, "heads": 4,  "depth": 4},
        "small": {"embed_dim": 256, "heads": 8,  "depth": 8},
        "base":  {"embed_dim": 512, "heads": 12, "depth": 16},
    }[args.model_type]
    for k, v in cfg.items():
        setattr(args, k, v)


def _rank_world():
    # Lightning DDP subprocess sets these
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local_rank


def _stable_run_tag(args):
    # MUST be identical across ranks
    if getattr(args, "run_tag", ""):
        return str(args.run_tag)
    if "SLURM_JOB_ID" in os.environ:
        return str(os.environ["SLURM_JOB_ID"])
    # fallback: fixed string (avoid per-rank time!)
    return "local"


def construct_run_name(args):
    tag = _stable_run_tag(args)
    run = "FT" if args.load_from_pretrained else "Supervised"
    run += f"_{args.model_type}"
    run += f"_{args.data_id}_{args.model_id}"
    run += f"_bs{args.batch_size}_lr{args.lr}_seed{args.random_seed}"
    run += f"_xnorm{args.x_norm}_ynorm{args.y_norm}_H{args.horizon}"
    run += f"_tag{tag}"
    return run


def resolve_parquet_path(args, split: str) -> str:
    """
    Matches your dataset naming:
      train: train_1p.parquet (when data_percentage != 100)
      val  : val.parquet
      test : test.parquet
    """
    base = os.path.join(args.data_path, args.data_id)
    if split == "train":
        if args.data_percentage == "100" or "shot" in str(args.data_percentage):
            return os.path.join(base, "train.parquet")
        return os.path.join(base, f"train_{args.data_percentage}p.parquet")
    return os.path.join(base, f"{split}.parquet")


def wait_for_file(path: str, timeout_s: int = 3600):
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(1.0)


# -------------------------
# GLOBAL stats from parquet (rank0 only)
# -------------------------
@torch.no_grad()
def compute_global_stats_from_train_parquet(train_parquet: str, C: int, H: int, x_mode: str, y_mode: str, eps=1e-6):
    """
    Streaming computation over TRAIN parquet only.
    Parquet columns:
      - samples: FixedSizeList[float] length = L*C   (flattened from (L,C))
      - labels : FixedSizeList[float] length = H*C   (flattened from (H,C))
    """
    pf = pq.ParquetFile(train_parquet, memory_map=True)
    cols = ["samples"]
    if "labels" in pf.schema_arrow.names:
        cols.append("labels")

    # X accumulators
    if x_mode == "zscore":
        sum_x = np.zeros((C,), dtype=np.float64)
        sumsq_x = np.zeros((C,), dtype=np.float64)
        cnt_x = 0
    elif x_mode in ("minmax01", "minmax11"):
        min_x = np.full((C,), np.inf, dtype=np.float64)
        max_x = np.full((C,), -np.inf, dtype=np.float64)
    else:
        sum_x = sumsq_x = cnt_x = None

    # Y accumulators
    if y_mode == "zscore":
        sum_y = np.zeros((C,), dtype=np.float64)
        sumsq_y = np.zeros((C,), dtype=np.float64)
        cnt_y = 0
    elif y_mode in ("minmax01", "minmax11"):
        min_y = np.full((C,), np.inf, dtype=np.float64)
        max_y = np.full((C,), -np.inf, dtype=np.float64)
    else:
        sum_y = sumsq_y = cnt_y = None

    for rb in pf.iter_batches(batch_size=8192, columns=cols):
        n = rb.num_rows

        # -------- X --------
        samples = rb.column(rb.schema.get_field_index("samples"))  # FixedSizeListArray
        list_size = samples.type.list_size  # L*C
        if list_size % C != 0:
            raise ValueError(f"samples list_size={list_size} not divisible by C={C}")
        L = list_size // C

        x_vals = samples.values.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        x = x_vals.reshape(n, L, C)  # (n,L,C)

        if x_mode == "zscore":
            sum_x += x.sum(axis=(0, 1))
            sumsq_x += (x * x).sum(axis=(0, 1))
            cnt_x += n * L
        elif x_mode in ("minmax01", "minmax11"):
            min_x = np.minimum(min_x, x.min(axis=(0, 1)))
            max_x = np.maximum(max_x, x.max(axis=(0, 1)))

        # -------- Y --------
        if "labels" in cols:
            labels = rb.column(rb.schema.get_field_index("labels"))
            y_list_size = labels.type.list_size  # H*C
            if y_list_size != H * C:
                raise ValueError(f"labels list_size={y_list_size} expected={H*C} (H={H},C={C})")

            y_vals = labels.values.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            y = y_vals.reshape(n, H, C)  # (n,H,C)

            if y_mode == "zscore":
                sum_y += y.sum(axis=(0, 1))
                sumsq_y += (y * y).sum(axis=(0, 1))
                cnt_y += n * H
            elif y_mode in ("minmax01", "minmax11"):
                min_y = np.minimum(min_y, y.min(axis=(0, 1)))
                max_y = np.maximum(max_y, y.max(axis=(0, 1)))

    stats = {"x_mode": x_mode, "y_mode": y_mode}

    # pack X
    if x_mode == "zscore":
        mean_x = sum_x / max(1, cnt_x)
        var_x = sumsq_x / max(1, cnt_x) - mean_x**2
        var_x = np.maximum(var_x, eps**2)
        std_x = np.sqrt(var_x)
        stats["x_mean"] = torch.tensor(mean_x, dtype=torch.float32).view(1, C, 1)
        stats["x_std"] = torch.tensor(std_x, dtype=torch.float32).view(1, C, 1)
    elif x_mode in ("minmax01", "minmax11"):
        scale = np.maximum(max_x - min_x, eps)
        stats["x_min"] = torch.tensor(min_x, dtype=torch.float32).view(1, C, 1)
        stats["x_scale"] = torch.tensor(scale, dtype=torch.float32).view(1, C, 1)

    # pack Y
    if y_mode == "zscore":
        mean_y = sum_y / max(1, cnt_y)
        var_y = sumsq_y / max(1, cnt_y) - mean_y**2
        var_y = np.maximum(var_y, eps**2)
        std_y = np.sqrt(var_y)
        stats["y_mean"] = torch.tensor(mean_y, dtype=torch.float32).view(1, C, 1)
        stats["y_std"] = torch.tensor(std_y, dtype=torch.float32).view(1, C, 1)
    elif y_mode in ("minmax01", "minmax11"):
        scale = np.maximum(max_y - min_y, eps)
        stats["y_min"] = torch.tensor(min_y, dtype=torch.float32).view(1, C, 1)
        stats["y_scale"] = torch.tensor(scale, dtype=torch.float32).view(1, C, 1)

    return stats


def apply_x_norm_(x: torch.Tensor, stats: dict):
    mode = stats["x_mode"]
    if mode == "none":
        return x
    if mode == "zscore":
        return x.sub_(stats["x_mean"]).div_(stats["x_std"])
    if mode == "minmax01":
        return x.sub_(stats["x_min"]).div_(stats["x_scale"])
    if mode == "minmax11":
        x.sub_(stats["x_min"]).div_(stats["x_scale"])
        return x.mul_(2.0).sub_(1.0)
    raise ValueError(mode)


def apply_y_norm_(y: torch.Tensor, stats: dict):
    mode = stats["y_mode"]
    if mode == "none":
        return y
    if mode == "zscore":
        return y.sub_(stats["y_mean"]).div_(stats["y_std"])
    if mode == "minmax01":
        return y.sub_(stats["y_min"]).div_(stats["y_scale"])
    if mode == "minmax11":
        y.sub_(stats["y_min"]).div_(stats["y_scale"])
        return y.mul_(2.0).sub_(1.0)
    raise ValueError(mode)


def invert_y(y_norm: torch.Tensor, stats: dict):
    mode = stats["y_mode"]
    if mode == "none":
        return y_norm
    if mode == "zscore":
        return y_norm * stats["y_std"].to(y_norm.device) + stats["y_mean"].to(y_norm.device)
    if mode == "minmax01":
        return y_norm * stats["y_scale"].to(y_norm.device) + stats["y_min"].to(y_norm.device)
    if mode == "minmax11":
        y01 = (y_norm + 1.0) / 2.0
        return y01 * stats["y_scale"].to(y_norm.device) + stats["y_min"].to(y_norm.device)
    raise ValueError(mode)


# -------------------------
# Lightning module
# -------------------------
class LitForecast(pl.LightningModule):
    def __init__(self, args, norm_stats: dict):
        super().__init__()
        self.args = args
        self.model = Transformer_bkbone(args)
        self.loss_beta = float(getattr(args, "huber_beta", 1.0))

        # keep norm stats (buffers so they move to GPU)
        self.norm = norm_stats
        # register only what exists
        for k, v in norm_stats.items():
            if torch.is_tensor(v):
                self.register_buffer(k, v)

        # buffers for single-GPU test saving
        self._test_pred = []
        self._test_true = []
        self._test_orig = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wt_decay)

        total_steps = int(self.args.num_epochs * max(1, getattr(self.args, "tl_length", 1)))
        warmup = int(0.1 * total_steps)

        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

    def _rmse(self, pred, true):
        return torch.sqrt(F.mse_loss(pred, true))

    def _shared_step(self, batch, stage: str):
        x, y, orig = batch
        feats = self(x)
        pred = self.model.predict(feats).float()  # (B,C,H)
        y = y.float()

        loss = F.smooth_l1_loss(pred, y, beta=self.loss_beta)

        # RMSE in original units (invert y_norm)
        pred_o = invert_y(pred, self._buffered_norm())
        y_o = invert_y(y, self._buffered_norm())
        rmse = self._rmse(pred_o, y_o)

        if stage == "train":
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        elif stage == "val":
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            # during DDP test we do NOT save arrays; we only save in single-GPU test pass
            self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=False)
            self.log("test_rmse", rmse, on_step=False, on_epoch=True, sync_dist=False)

            if self.trainer.world_size == 1:
                self._test_pred.append(pred_o.detach().cpu().numpy().astype(np.float32))
                self._test_true.append(y_o.detach().cpu().numpy().astype(np.float32))
                self._test_orig.append(orig.detach().cpu().numpy().astype(np.int64))

        return loss

    def _buffered_norm(self):
        # rebuild dict using buffers so device placement is correct
        out = {"x_mode": self.norm["x_mode"], "y_mode": self.norm["y_mode"]}
        for k in ("y_mean", "y_std", "y_min", "y_scale"):
            if hasattr(self, k):
                out[k] = getattr(self, k)
        return out

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_test_epoch_end(self):
        # only save on single-GPU test pass
        if self.trainer.world_size != 1:
            return
        if len(self._test_pred) == 0:
            print("[WARN] No test batches collected, nothing saved.")
            return

        pred = np.concatenate(self._test_pred, axis=0)  # (N,C,H)
        true = np.concatenate(self._test_true, axis=0)
        orig = np.concatenate(self._test_orig, axis=0)  # (N,)

        order = np.argsort(orig)
        pred, true, orig = pred[order], true[order], orig[order]

        ckpt_dir = os.path.abspath(self.args.ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

        np.save(os.path.join(ckpt_dir, "test_preds.npy"), pred)
        np.save(os.path.join(ckpt_dir, "test_targets.npy"), true)
        np.save(os.path.join(ckpt_dir, "test_orig_row.npy"), orig)

        rmse_all = float(np.sqrt(np.mean((pred - true) ** 2)))
        print(f"[OK] Saved FULL test outputs to {ckpt_dir}")
        print(f"     pred={pred.shape} true={true.shape} orig={orig.shape} rmse={rmse_all:.6f}")

        self._test_pred, self._test_true, self._test_orig = [], [], []


# -------------------------
# main
# -------------------------
def main(args):
    pl.seed_everything(args.random_seed, workers=True)

    rank, world, local_rank = _rank_world()

    run_name = construct_run_name(args)
    ckpt_dir = os.path.abspath(os.path.join("checkpoints", run_name))
    os.makedirs(ckpt_dir, exist_ok=True)
    args.ckpt_dir = ckpt_dir

    # -------- compute + share global norm stats --------
    norm_path = os.path.join(ckpt_dir, "norm_stats.pt")

    if rank == 0:
        train_parquet = resolve_parquet_path(args, "train")
        print(f"[rank0] computing GLOBAL norm stats from: {train_parquet}")
        stats = compute_global_stats_from_train_parquet(
            train_parquet=train_parquet,
            C=args.num_channels,
            H=args.horizon,
            x_mode=args.x_norm,
            y_mode=args.y_norm,
            eps=1e-6,
        )
        torch.save(stats, norm_path)
        print(f"[rank0] saved norm stats to: {norm_path}")

    # other ranks wait
    wait_for_file(norm_path)
    norm_stats = torch.load(norm_path, map_location="cpu")
    # make sure tensors are torch tensors
    for k, v in list(norm_stats.items()):
        if isinstance(v, np.ndarray):
            norm_stats[k] = torch.from_numpy(v)

    # -------- loaders (sharded by rank via your dataset) --------
    train_loader, val_loader = get_train_val_loaders(args)

    # dataset-derived shapes
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]
    args.tl_length = len(train_loader)

    # -------- apply norms in-place to shard tensors --------
    if args.x_norm != "none":
        apply_x_norm_(train_loader.dataset.x_data, norm_stats)
        apply_x_norm_(val_loader.dataset.x_data, norm_stats)

    if args.y_norm != "none":
        apply_y_norm_(train_loader.dataset.y_data, norm_stats)
        apply_y_norm_(val_loader.dataset.y_data, norm_stats)

    # -------- callbacks --------
    checkpoint = ModelCheckpoint(
        monitor="val_rmse",
        mode="min",
        save_top_k=1,
        dirpath=ckpt_dir,
        filename="best",
    )
    early_stop = EarlyStopping(monitor="val_rmse", patience=args.patience, mode="min")
    save_copy_of_files(checkpoint)

    # -------- model --------
    lit = LitForecast(args, norm_stats)

    # -------- pretrained partial load (CPU safe) --------
    if args.load_from_pretrained:
        path = os.path.join(args.pretrained_model_dir, f"pretrain-epoch={args.pretraining_epoch_id}.ckpt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ck_state = ck["state_dict"]
        model_state = lit.state_dict()
        matched = {k: v for k, v in ck_state.items() if k in model_state and model_state[k].shape == v.shape}
        lit.load_state_dict(matched, strict=False)
        if rank == 0:
            print(f"[rank0] Loaded pretrained: {path}  matched={len(matched)}/{len(model_state)}")

    # -------- TRAIN: single SLURM task, Lightning spawns DDP processes --------
    trainer = pl.Trainer(
        default_root_dir=ckpt_dir,
        max_epochs=args.num_epochs,
        callbacks=[checkpoint, early_stop, TQDMProgressBar(refresh_rate=200)],
        accelerator="gpu",
        devices=args.num_gpus if args.num_gpus > 1 else 1,
        strategy="ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto",
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        use_distributed_sampler=False,  # IMPORTANT: your dataset already shards by rank
    )

    trainer.fit(lit, train_loader, val_loader)

    # -------- SINGLE-GPU TEST PASS on rank0 to save FULL preds/targets --------
    if trainer.is_global_zero:
        print("[rank0] Running SINGLE-GPU test pass to save FULL test_preds.npy ...")

        test_args = copy.deepcopy(args)
        test_args.num_gpus = 1
        test_args.shard_by_rank = False
        test_args.num_workers = 0
        test_args.pin_memory = False

        test_loader = get_test_loader(test_args)

        # apply same norms
        if test_args.x_norm != "none":
            apply_x_norm_(test_loader.dataset.x_data, norm_stats)
        if test_args.y_norm != "none":
            apply_y_norm_(test_loader.dataset.y_data, norm_stats)

        # build model + load best
        best_path = checkpoint.best_model_path
        ck = torch.load(best_path, map_location="cpu")
        test_model = LitForecast(test_args, norm_stats)
        test_model.load_state_dict(ck["state_dict"], strict=False)
        test_model.args.ckpt_dir = ckpt_dir

        test_trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            logger=False,
            enable_checkpointing=False,
        )
        test_trainer.test(test_model, test_loader)

    if rank == 0:
        print("[DONE] outputs at:", ckpt_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--data_id", type=str, required=True)
    p.add_argument("--data_percentage", type=str, default="1")
    p.add_argument("--model_id", type=str, default="Sw_fc")
    p.add_argument("--run_tag", type=str, default="")  # optional; default uses SLURM_JOB_ID

    p.add_argument("--model_type", choices=["tiny", "small", "base"], default="tiny")
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wt_decay", type=float, default=1e-4)
    p.add_argument("--random_seed", type=int, default=42)

    p.add_argument("--task_type", type=str, default="RUL", choices=["FD", "RUL"])
    p.add_argument("--num_classes", type=int, default=1)

    p.add_argument("--num_channels", type=int, default=11)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--huber_beta", type=float, default=1.0)

    p.add_argument("--x_norm", choices=["none", "minmax01", "minmax11", "zscore"], default="zscore")
    p.add_argument("--y_norm", choices=["none", "minmax01", "minmax11", "zscore"], default="zscore")

    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", type=str2bool, default=False)

    p.add_argument("--shard_by_rank", type=str2bool, default=True)

    p.add_argument("--load_from_pretrained", type=str2bool, default=True)
    p.add_argument("--pretrained_model_dir", type=str, required=True)
    p.add_argument("--pretraining_epoch_id", type=int, default=1)

    args = p.parse_args()
    apply_model_config(args)
    main(args)