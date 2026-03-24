# test_ckpt.py
# - Loads ckpt (auto task_type/num_classes from ckpt head)
# - Applies saved normalization stats (norm_stats.pt next to ckpt, or via --norm_stats_path)
# - Runs ONE pass over concatenated train+val+test (chronological)
# - Plots Pred vs Actual as a single time-series with dotted vertical lines marking train|val and val|test

import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt

from datalaoders.train_dataloader import get_datasets
from utils import str2bool
from fine_tune_sanwa import Model, apply_model_config


# -------------------------
# Helpers
# -------------------------
def ckpt_output_dim(ckpt_path: str) -> int:
    """Return output dim of last classifier/reg head from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", {})
    key = "model.cls_head.4.weight"
    if key in sd and isinstance(sd[key], torch.Tensor) and sd[key].ndim == 2:
        return int(sd[key].shape[0])

    # fallback: find ANY 2D weight that looks like final head [out_dim, hidden]
    candidates = [(k, v) for k, v in sd.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
    if not candidates:
        raise RuntimeError("No 2D weights found in checkpoint; cannot infer output dim.")
    candidates.sort(key=lambda kv: kv[1].shape[0])
    k, v = candidates[0]
    print(f"[WARN] Could not find {key}. Using fallback key={k} shape={tuple(v.shape)}")
    return int(v.shape[0])


def apply_x_minmax11_to_splits(train_loader, val_loader, test_loader, x_min, x_scale):
    """Apply per-channel min-max normalization to [-1,1] using TRAIN stats."""
    for ds in [train_loader.dataset, val_loader.dataset, test_loader.dataset]:
        x = ds.x_data.float()
        ds.x_data = 2.0 * ((x - x_min) / x_scale) - 1.0


def apply_y_minmax11_vec_to_splits(train_loader, val_loader, test_loader, y_min, y_scale):
    """Apply per-channel min-max normalization to [-1,1] for RUL labels."""
    def _to_np(a):
        return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)

    def _pm1(y_np):
        y_np = np.asarray(y_np, dtype=np.float32)  # [N,C]
        y01 = (y_np - y_min) / y_scale
        return (2.0 * y01 - 1.0).astype(np.float32)

    for ds in [train_loader.dataset, val_loader.dataset, test_loader.dataset]:
        y = _to_np(ds.y_data)
        y_pm1 = _pm1(y)
        ds.y_data = torch.tensor(y_pm1, dtype=torch.float32)


def try_load_norm_stats(args):
    """
    Try to load norm_stats.pt saved during training.
    Priority:
      1) --norm_stats_path if provided
      2) ckpt_dir/norm_stats.pt (same folder as ckpt)
    """
    if getattr(args, "norm_stats_path", None):
        p = args.norm_stats_path
        if os.path.isfile(p):
            return torch.load(p, map_location="cpu")
        else:
            print(f"[WARN] norm_stats_path provided but not found: {p}")

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
    p = os.path.join(ckpt_dir, "norm_stats.pt")
    if os.path.isfile(p):
        return torch.load(p, map_location="cpu")

    print("[WARN] norm_stats.pt not found (next to ckpt or via --norm_stats_path). "
          "Inference will run, but normalization may NOT match training.")
    return None


def plot_pred_vs_true_timeseries(args, pred, true):
    """
    Plot one long time series with train|val|test separators.
    pred/true are [N,C] (RUL) or [N,] (FD class ids - not recommended for this plot).
    """
    pred = np.asarray(pred)
    true = np.asarray(true)

    # If [N,] -> make [N,1] for unified plotting
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if true.ndim == 1:
        true = true.reshape(-1, 1)

    N, C = true.shape
    t = np.arange(N)

    b1 = args.n_train
    b2 = args.n_train + args.n_val

    names = getattr(args, "y_channel_names", None)
    if names is None or len(names) < C:
        names = [f"ch{i}" for i in range(C)]

    K = min(getattr(args, "plot_channels", 4), C)

    for ch in range(K):
        plt.figure(figsize=(14, 5))
        plt.plot(t, true[:, ch], label="Actual")
        plt.plot(t, pred[:, ch], label="Pred")

        # dotted separators
        plt.axvline(b1 - 0.5, linestyle=":", linewidth=2)
        plt.axvline(b2 - 0.5, linestyle=":", linewidth=2)

        # region labels
        ymax = float(max(true[:, ch].max(), pred[:, ch].max()))
        ymin = float(min(true[:, ch].min(), pred[:, ch].min()))
        ytxt = ymin + 0.95 * (ymax - ymin + 1e-9)
        plt.text(b1 * 0.5, ytxt, "train", ha="center", va="top")
        plt.text(b1 + (args.n_val * 0.5), ytxt, "val", ha="center", va="top")
        plt.text(b2 + (args.n_test * 0.5), ytxt, "test", ha="center", va="top")

        plt.title(f"Pred vs Actual (chronological): {names[ch]}")
        plt.xlabel("Time (index)")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        safe = names[ch].replace(" ", "_").replace("/", "_")
        out = os.path.join(args.ckpt_dir, f"ts_pred_vs_true_{ch:02d}_{safe}.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()

    print("[INFO] Saved time-series plots to:", args.ckpt_dir)


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # data/model config
    p.add_argument('--data_path', type=str, default=r'./dataset/')
    p.add_argument('--data_id', type=str, default=r'M01')
    p.add_argument('--data_percentage', type=str, default="1")
    p.add_argument('--model_id', type=str, default="CNC_FT")

    p.add_argument('--model_type', type=str, choices=['tiny', 'small', 'base'], default='tiny')
    p.add_argument('--patch_size', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--use_moe', type=str2bool, default=False)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--gpu_id', type=int, default=0)

    # output
    p.add_argument('--out_dir', type=str, required=True)

    # checkpoint to load
    p.add_argument('--ckpt_path', type=str, required=True)

    # optional norm stats (if not in same folder as ckpt)
    p.add_argument('--norm_stats_path', type=str, default=None)

    # plotting
    p.add_argument('--plot_channels', type=int, default=4,
                   help="How many channels to plot (RUL). Default 4.")

    # seed
    p.add_argument('--random_seed', type=int, default=42)

    # exist because Model() expects them (even for testing)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--wt_decay', type=float, default=1e-4)
    p.add_argument('--num_epochs', type=int, default=1)
    p.add_argument('--patience', type=int, default=1)

    # allow override, but default "auto"
    p.add_argument('--task_type', type=str, default='auto', choices=['auto', 'FD', 'RUL'])

    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    apply_model_config(args)
    pl.seed_everything(args.random_seed)

    os.makedirs(args.out_dir, exist_ok=True)
    args.ckpt_dir = args.out_dir
    print(f"[INFO] Inference outputs will be saved to: {args.ckpt_dir}")

    # -----------------------
    # Decide task_type/num_classes from checkpoint
    # -----------------------
    out_dim = ckpt_output_dim(args.ckpt_path)
    if args.task_type == "auto":
        args.task_type = "RUL" if out_dim == 1 else "FD"

    if args.task_type == "FD":
        args.num_classes = out_dim
        args.class_names = [str(i) for i in range(args.num_classes)]
        if args.num_classes < 2:
            print("[WARN] FD chosen but checkpoint out_dim < 2. "
                  "CrossEntropy FD usually expects num_classes>=2. "
                  "You probably want task_type=RUL for this checkpoint.")
    else:
        args.num_classes = 1

    # optional names for nicer plots (edit if you want)
    args.y_channel_names = [
        "max_injection_pressure",
        "plastification_time",
        "end_of_packing_stroke",
        "switchover_pressure",
    ]

    print("[INFO] CKPT head out_dim =", out_dim)
    print("[INFO] Using task_type   =", args.task_type)
    print("[INFO] num_classes       =", args.num_classes)

    # -----------------------
    # Load datasets
    # -----------------------
    train_loader, val_loader, test_loader = get_datasets(args)

    # boundaries for plotting
    args.n_train = len(train_loader.dataset)
    args.n_val   = len(val_loader.dataset)
    args.n_test  = len(test_loader.dataset)

    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]
    args.tl_length = len(train_loader)

    # -----------------------
    # Load + apply normalization stats (recommended)
    # -----------------------
    stats = try_load_norm_stats(args)
    if stats is not None:
        if stats.get("x_min", None) is not None and stats.get("x_scale", None) is not None:
            x_min = stats["x_min"].float()      # [1,C,1]
            x_scale = stats["x_scale"].float()  # [1,C,1]
            apply_x_minmax11_to_splits(train_loader, val_loader, test_loader, x_min, x_scale)
            print("[INFO] Applied X min-max [-1,1] using saved TRAIN stats.")
        else:
            print("[WARN] norm_stats.pt missing x_min/x_scale.")

        # For RUL, your training normalized y to [-1,1], so do the same here
        if args.task_type == "RUL":
            if stats.get("y_min", None) is not None and stats.get("y_scale", None) is not None:
                y_min = np.asarray(stats["y_min"], dtype=np.float32)      # [C]
                y_scale = np.asarray(stats["y_scale"], dtype=np.float32)  # [C]

                # attach for your LightningModule inverse transform reporting
                args.y_min = torch.tensor(y_min, dtype=torch.float32)
                args.y_scale = torch.tensor(y_scale, dtype=torch.float32)

                apply_y_minmax11_vec_to_splits(train_loader, val_loader, test_loader, y_min, y_scale)
                print("[INFO] Applied Y min-max [-1,1] using saved TRAIN stats (RUL).")
            else:
                print("[WARN] norm_stats.pt missing y_min/y_scale (RUL). "
                      "Loss/metrics may be inconsistent with training.")

    # -----------------------
    # Combine train+val+test for one pass (chronological)
    # -----------------------
    all_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset, test_loader.dataset])
    all_loader = DataLoader(
        all_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    print(f"[INFO] Combined dataset size: {len(all_dataset)} "
          f"(train={args.n_train}, val={args.n_val}, test={args.n_test})")

    # -----------------------
    # Load model from checkpoint
    # -----------------------
    model = Model.load_from_checkpoint(args.ckpt_path, args=args, strict=True)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=[args.gpu_id] if torch.cuda.is_available() else "auto",
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=False,
        enable_checkpointing=False,
    )

    # Run inference
    trainer.test(model=model, dataloaders=all_loader)

    # -----------------------
    # Plot Pred vs Actual (RUL)
    # -----------------------
    pred_path = os.path.join(args.ckpt_dir, "test_preds.npy")
    true_path = os.path.join(args.ckpt_dir, "test_targets.npy")

    if os.path.isfile(pred_path) and os.path.isfile(true_path):
        pred = np.load(pred_path)  # expected [N,C] for RUL
        true = np.load(true_path)
        plot_pred_vs_true_timeseries(args, pred, true)
    else:
        print("[WARN] Could not find test_preds.npy / test_targets.npy in out_dir.")
        print("       If task_type=FD, your current code doesn't save continuous preds anyway.")
        print("       If task_type=RUL, check that on_test_epoch_end ran and wrote the files.")


if __name__ == "__main__":
    main()
