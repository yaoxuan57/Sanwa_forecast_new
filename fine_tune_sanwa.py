# X normalization = per-channel min-max to [-1,1] using TRAIN stats.
# Y per-channel Min-Max scaling of y to [-1, 1], fitted on TRAIN and applied to TRAIN/VAL/TEST.
import os
import argparse
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Qt5Agg')
from sklearn.metrics import classification_report

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torchmetrics.classification import Accuracy, MulticlassF1Score, MulticlassConfusionMatrix
from torchmetrics.regression import MeanSquaredError
from datalaoders.train_dataloader import get_datasets
from model.model import Transformer_bkbone
from utils import save_copy_of_files, str2bool, get_rul_report, scoring_function_v2


def minmax_fit_vec(y_np, eps=1e-8):
    y = np.asarray(y_np, dtype=np.float32)
    y_min = y.min(axis=0)
    y_max = y.max(axis=0)
    scale = (y_max - y_min) + eps
    return y_min, y_max, scale


def minmax_apply_pm1_vec(y_np, y_min, scale):
    y = np.asarray(y_np, dtype=np.float32)
    y01 = (y - y_min) / scale
    return (2.0 * y01 - 1.0).astype(np.float32)


def fit_channel_zscore_torch(x: torch.Tensor, eps: float = 1e-6):
    x = x.float()
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True, unbiased=False)
    std = torch.clamp(std, min=eps)
    return mean, std


def fit_channel_minmax_torch(x: torch.Tensor, eps: float = 1e-6):
    x = x.float()
    x_min = x.amin(dim=(0, 2), keepdim=True)
    x_max = x.amax(dim=(0, 2), keepdim=True)
    scale = torch.clamp(x_max - x_min, min=eps)
    return x_min, scale


def apply_channel_minmax11_torch(x: torch.Tensor, x_min: torch.Tensor, scale: torch.Tensor):
    return 2.0 * ((x.float() - x_min) / scale) - 1.0

# def minmax_fit_vec(y_np, eps=1e-8):
#     y = np.asarray(y_np, dtype=np.float32)          # [N, C]
#     y_min = y.min(axis=0)                           # [C]
#     y_max = y.max(axis=0)                           # [C]
#     scale = (y_max - y_min) + eps                   # [C]
#     return y_min, y_max, scale

# def minmax_apply_pm1_vec(y_np, y_min, scale):
#     y = np.asarray(y_np, dtype=np.float32)          # [N, C]
#     y01 = (y - y_min) / scale
#     return (2.0 * y01 - 1.0).astype(np.float32)

# def minmax_apply_pm1(y_np, y_min, scale):
#     # map to [-1, 1]
#     y = np.asarray(y_np, dtype=np.float32).reshape(-1)
#     y01 = (y - y_min) / scale
#     return (2.0 * y01 - 1.0).astype(np.float32)

# def minmax_invert_pm1(y_pm1_torch, y_min, scale):
#     # inverse from [-1,1] back to original
#     y01 = (y_pm1_torch + 1.0) / 2.0
#     return y01 * scale + y_min

# def fit_channel_minmax_torch(x: torch.Tensor, eps: float = 1e-6):
#     # x: [N, C, L]
#     x = x.float()
#     x_min = x.amin(dim=(0,2), keepdim=True)   # [1,C,1]
#     x_max = x.amax(dim=(0,2), keepdim=True)
#     scale = torch.clamp(x_max - x_min, min=eps)
#     return x_min, scale

# def apply_channel_minmax01_torch(x: torch.Tensor, x_min: torch.Tensor, scale: torch.Tensor):
#     # -> [0, 1]
#     return (x.float() - x_min) / scale

# def apply_channel_minmax11_torch(x: torch.Tensor, x_min: torch.Tensor, scale: torch.Tensor):
#     # -> [-1, 1]
#     return 2.0 * ((x.float() - x_min) / scale) - 1.0

# def minmax_fit(y_np, eps=1e-8):
#     y = np.asarray(y_np, dtype=np.float32).reshape(-1)
#     y_min = float(np.min(y))
#     y_max = float(np.max(y))
#     scale = float((y_max - y_min) + eps)
#     return y_min, y_max, scale

# def minmax_apply(y_np, y_min, scale):
#     y = np.asarray(y_np, dtype=np.float32).reshape(-1)
#     return ((y - y_min) / scale).astype(np.float32)

# def minmax_invert(y_norm_torch, y_min, scale):
#     # y_norm_torch: torch tensor
#     return y_norm_torch * scale + y_min

# def zscore_fit(y_np, eps=1e-8):
#     y_np = np.asarray(y_np, dtype=np.float32).reshape(-1)
#     mean = float(y_np.mean())
#     std = float(y_np.std() + eps)
#     return mean, std

# def zscore_apply(y_np, mean, std):
#     y_np = np.asarray(y_np, dtype=np.float32).reshape(-1)
#     return ((y_np - mean) / std).astype(np.float32)

# def fit_channel_zscore_torch(x: torch.Tensor, eps: float = 1e-6):
#     """
#     x: [N, C, L]
#     returns mean/std with shape [1, C, 1] (broadcastable)
#     """
#     x = x.float()
#     mean = x.mean(dim=(0, 2), keepdim=True)                 # [1, C, 1]
#     std  = x.std(dim=(0, 2), keepdim=True, unbiased=False)  # [1, C, 1]
#     std = torch.clamp(std, min=eps)
#     return mean, std

# def apply_channel_zscore_torch(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
#     return (x.float() - mean) / std

# ==================== Model Wrapper ====================
class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Transformer_bkbone(args)
        if args.task_type == 'FD':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif args.task_type == 'RUL':
            self.loss_fn = torch.nn.MSELoss()
        if args.task_type == 'FD':
            self.train_metrics = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=args.num_classes),
                "f1": MulticlassF1Score(num_classes=args.num_classes, average="macro")
            })
            self.val_metrics = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=args.num_classes),
                "f1": MulticlassF1Score(num_classes=args.num_classes, average="macro")
            })
            self.test_f1 = MulticlassF1Score(num_classes=args.num_classes, average="macro")
            self.confusion_matrix = MulticlassConfusionMatrix(num_classes=args.num_classes)
        elif args.task_type == 'RUL':
            self.train_metrics = MetricCollection({
                "rmse": MeanSquaredError(squared=False)
            })
            self.val_metrics = MetricCollection({
                "rmse": MeanSquaredError(squared=False)
            })
            # original-unit RMSE (for reporting only)
            self.train_rmse_orig = MeanSquaredError(squared=False)
            self.val_rmse_orig   = MeanSquaredError(squared=False)

            self.test_rmse = MeanSquaredError(squared=False)

        print("=== DEBUG: head/reg parameters ===")
        for n, p in self.named_parameters():
            if ("head" in n) or ("reg" in n) or ("predict" in n):
                print("[DEBUG param]", n,
                      "requires_grad=", p.requires_grad,
                      "shape=", tuple(p.shape))
        print("==================================")


        self.total_steps = args.num_epochs * args.tl_length
        self.num_warmup_steps = int(0.1 * self.total_steps)  # 2048

        self.test_preds = []
        self.test_targets = []

    def _rul_composite_loss(self, preds, y):
        # Point term: robust value matching with Huber/SmoothL1.
        point = F.smooth_l1_loss(preds, y, beta=self.args.huber_beta, reduction="mean")

        # First difference term: align local trend/direction.
        if preds.size(-1) >= 2:
            d_pred = preds[..., 1:] - preds[..., :-1]
            d_true = y[..., 1:] - y[..., :-1]
            diff = torch.mean((d_pred - d_true) ** 2)
        else:
            diff = preds.new_tensor(0.0)

        # Second difference term: align turning/curvature behavior.
        if preds.size(-1) >= 3:
            dd_pred = preds[..., 2:] - 2.0 * preds[..., 1:-1] + preds[..., :-2]
            dd_true = y[..., 2:] - 2.0 * y[..., 1:-1] + y[..., :-2]
            curvature = torch.mean((dd_pred - dd_true) ** 2)
        else:
            curvature = preds.new_tensor(0.0)

        # Variance term: discourage overly flat predictions.
        pred_std = torch.std(preds, dim=-1, unbiased=False)
        true_std = torch.std(y, dim=-1, unbiased=False)
        variance = torch.mean((pred_std - true_std) ** 2)

        total = (
            self.args.point_loss_weight * point
            + self.args.diff_loss_weight * diff
            + self.args.curvature_loss_weight * curvature
            + self.args.variance_loss_weight * variance
        )
        return total, point, diff, curvature, variance

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wt_decay)

        scheduler = {
            'scheduler': self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup_steps,
                                                              num_training_steps=self.total_steps),
            'name': 'learning_rate', 'interval': 'step', 'frequency': 1,
        }
        return [optimizer], [scheduler]


    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _shared_step(self, batch, stage):
        x, y = batch

        if self.args.task_type == "FD":
            # ---- target: force (B,) long ----
            if y.ndim > 1:
                # one-hot (B,K) OR (B,1)
                if y.size(-1) == 1:
                    y = y.view(-1)
                else:
                    y = torch.argmax(y, dim=1)
            y = y.long()

            # ---- forward gives features/tokens ----
            feats = self(x)

            # ---- predict() must output class logits (B,num_classes) ----
            class_logits = self.model.predict(feats)

            # fix the B=1 squeeze case: (num_classes,) -> (1,num_classes)
            if class_logits.ndim == 1:
                class_logits = class_logits.unsqueeze(0)

            # if predict returns (B,1,num_classes) etc, flatten to (B,num_classes)
            if class_logits.ndim > 2:
                class_logits = class_logits.view(class_logits.size(0), -1)

            loss = self.loss_fn(class_logits, y)

            # for metrics use class indices (B,)
            preds = torch.argmax(class_logits, dim=1)
        #
        elif self.args.task_type == "RUL":
            feats = self(x)
            preds = self.model.predict(feats).float()   # [B, C]
            y = y.float()                               # [B, C]

            # Composite loss in normalized space.
            loss, point_loss, diff_loss, curvature_loss, variance_loss = self._rul_composite_loss(preds, y)

            # inverse-transform for metrics/reporting (original units)
            y_min  = getattr(self.args, "y_min", 0.0)
            scale  = getattr(self.args, "y_scale", 1.0)

            # ensure tensors on correct device and broadcastable: [C]
            if not torch.is_tensor(y_min):
                y_min = torch.tensor(y_min, dtype=torch.float32, device=preds.device)
            else:
                y_min = y_min.to(preds.device).float()

            if not torch.is_tensor(scale):
                scale = torch.tensor(scale, dtype=torch.float32, device=preds.device)
            else:
                scale = scale.to(preds.device).float()

            # broadcast to [B, C]
            preds_orig = (preds + 1.0) / 2.0 * scale + y_min
            y_orig     = (y     + 1.0) / 2.0 * scale + y_min

            # optional debug once
            if stage == "train" and self.current_epoch == 0 and self.global_step == 0:
                print("DEBUG preds/y shapes:", preds.shape, y.shape)  # expect [B,14]

            if stage == "train" and self.current_epoch == 0 and self.global_step == 0:
                print("[DEBUG x norm] mean/std:", x.float().mean().item(), x.float().std().item())
                print("[DEBUG] y batch mean/std (should be ~0/~1 if normalized):",
                    y.detach().mean().item(), y.detach().std().item())

            if stage == "train" and self.current_epoch == 0 and self.global_step < 3:
                print("[DEBUG] preds mean/std:", preds.detach().float().mean().item(),
                    preds.detach().float().std().item())

        # ---- metrics/logging ----
        if stage == "train":
            self.train_metrics.update(preds, y)
            self.log_dict({f"train_{k}": m for k, m in self.train_metrics.items()},
                        on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

            if self.args.task_type == "RUL":
                self.log("train_point_loss", point_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("train_diff_loss", diff_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("train_curvature_loss", curvature_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("train_variance_loss", variance_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.train_rmse_orig.update(preds_orig, y_orig)
                self.log("train_rmse_orig", self.train_rmse_orig, on_step=False, on_epoch=True, prog_bar=False)


        elif stage == "val":
            self.val_metrics.update(preds, y)
            self.log_dict({f"val_{k}": m for k, m in self.val_metrics.items()},
                        on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

            if self.args.task_type == "RUL":
                self.log("val_point_loss", point_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("val_diff_loss", diff_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("val_curvature_loss", curvature_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("val_variance_loss", variance_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.val_rmse_orig.update(preds_orig, y_orig)
                self.log("val_rmse_orig", self.val_rmse_orig, on_step=False, on_epoch=True, prog_bar=False)

        elif stage == "test":
            if self.args.task_type == "FD":
                self.test_f1.update(preds, y)
                self.confusion_matrix.update(preds, y)
                self.test_preds.extend(preds.cpu().numpy())
                self.test_targets.extend(y.cpu().numpy())

                acc = Accuracy(task="multiclass", num_classes=self.args.num_classes).to(preds.device)(preds, y)
                self.log("test_accuracy", acc, on_step=False, on_epoch=True)
            else:
                self.test_rmse.update(preds_orig, y_orig)

                preds_np = preds_orig.detach().to(torch.float32).cpu().numpy()  # [B, C]
                y_np     = y_orig.detach().to(torch.float32).cpu().numpy()      # [B, C]

                self.test_preds.append(preds_np)
                self.test_targets.append(y_np)

                self.log("test_composite_loss", loss, on_step=False, on_epoch=True)
                self.log("test_point_loss", point_loss, on_step=False, on_epoch=True)
                self.log("test_diff_loss", diff_loss, on_step=False, on_epoch=True)
                self.log("test_curvature_loss", curvature_loss, on_step=False, on_epoch=True)
                self.log("test_variance_loss", variance_loss, on_step=False, on_epoch=True)

            self.log("test_loss", loss, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.train_metrics.reset()
        if self.args.task_type == "RUL":
            self.train_rmse_orig.reset()

    def on_validation_epoch_end(self):
        self.val_metrics.reset()
        if self.args.task_type == "RUL":
            self.val_rmse_orig.reset()

    def on_test_epoch_end(self):
        if self.args.task_type == 'FD':
            f1_score = self.test_f1.compute()
            self.log("test_f1", f1_score)
            self.test_f1.reset()

            fig, ax = self.confusion_matrix.plot()
            fig.tight_layout()
            fig.savefig(f"{self.args.ckpt_dir}/confusion_matrix.png", bbox_inches="tight")
            print("Test Confusion Matrix saved.")
            self.confusion_matrix.reset()

            labels = list(range(self.args.num_classes))  # [0,1,2,3]
            print("unique y_true:", np.unique(self.test_targets, return_counts=True))
            print("unique y_pred:", np.unique(self.test_preds, return_counts=True))
            print("args.num_classes:", self.args.num_classes)
            print("class_names:", self.args.class_names)

            report = classification_report(
                self.test_targets,
                self.test_preds,
                labels=labels,
                target_names=self.args.class_names,
                digits=4,
                zero_division=0
            )
            print("=== Classification Report ===")
            print(report)

            with open(f"{self.args.ckpt_dir}/classification_report.txt", "w") as f:
                f.write(report)
        #
        elif self.args.task_type == 'RUL':
            rmse = self.test_rmse.compute()
            self.log("test_rmse", rmse)
            self.test_rmse.reset()

            # concat all batches -> [N, C]
            pred = np.concatenate(self.test_preds, axis=0).astype(np.float32)
            true = np.concatenate(self.test_targets, axis=0).astype(np.float32)

            # per-channel RMSE and mean RMSE
            rmse_per_ch = np.sqrt(np.mean((pred - true) ** 2, axis=0))  # [C]
            rmse_mean = float(rmse_per_ch.mean())

            print(f"[TEST] RMSE(mean over channels)={rmse_mean:.6f} | first 5 ch RMSE={rmse_per_ch[:5]}")

            # save arrays
            np.save(f"{self.args.ckpt_dir}/test_preds.npy", pred)
            np.save(f"{self.args.ckpt_dir}/test_targets.npy", true)

            # optional: log mean rmse explicitly
            self.log("test_rmse_mean", rmse_mean)

            # ---- write report (multi-channel) ----
            report = "=== Forecasting Report (multi-channel) ===\n"
            report += f"RMSE(mean over channels): {rmse_mean:.6f}\n"
            report += "RMSE per channel:\n"
            report += np.array2string(rmse_per_ch, precision=4, separator=", ")
            report += "\n\nFirst 3 samples (true -> pred) for first 5 channels:\n"

            for i in range(min(3, true.shape[0])):
                report += f"sample {i}: true={true[i, :5]} pred={pred[i, :5]}\n"

            print(report)
            with open(f"{self.args.ckpt_dir}/rul_report.txt", "w") as f:
                f.write(report)

            # ---- optional plot: scatter for ONE channel only ----
            # ---- scatter plots for the 4 named channels ----
            names = getattr(self.args, "y_channel_names", None)
            if names is None:
                names = [f"ch{i}" for i in range(true.shape[1])]

            C = true.shape[1]
            K = min(4, C, len(names))

            for ch in range(K):
                ch_name = names[ch]

                plt.figure(figsize=(8, 6))
                plt.scatter(true[:, ch], pred[:, ch], alpha=0.5)

                mn = float(min(true[:, ch].min(), pred[:, ch].min()))
                mx = float(max(true[:, ch].max(), pred[:, ch].max()))
                plt.plot([mn, mx], [mn, mx], 'r--')

                plt.xlabel(f"True ({ch_name})")
                plt.ylabel(f"Pred ({ch_name})")
                plt.title(f"Forecast scatter: {ch_name} | RMSE={rmse_per_ch[ch]:.4f}")

                plt.tight_layout()
                safe_name = ch_name.replace(" ", "_").replace("/", "_")
                plt.savefig(f"{self.args.ckpt_dir}/forecast_scatter_{ch:02d}_{safe_name}.png", bbox_inches="tight")
                plt.close()



        self.test_preds = []
        self.test_targets = []


# ==================== Callbacks ====================
def construct_experiment_dir(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_description = "FT" if args.load_from_pretrained else "Supervised"
    run_description += f"_{args.model_type}"
    run_description += f"_{args.data_id}_from{args.pretraining_epoch_id}_{args.model_id}"
    run_description += f"_bs{args.batch_size}_lr{args.lr}_seed{args.random_seed}_{timestamp}"
    return run_description


def plot_metrics(metrics, ckpt_dir, task_type):
    plt.figure()
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(f"{ckpt_dir}/loss.png", bbox_inches="tight")

    plt.figure()
    if task_type == 'FD':
        plt.plot(metrics["train_acc"], label="Train Acc")
        plt.plot(metrics["val_acc"], label="Val Acc")
        plt.legend()
        plt.title("Accuracy Curve")
    elif task_type == 'RUL':
        plt.plot(metrics["train_rmse"], label="Train RMSE")
        plt.plot(metrics["val_rmse"], label="Val RMSE")
        plt.legend()
        plt.title("RMSE Curve")
    plt.tight_layout()
    plt.savefig(f"{ckpt_dir}/performance_metric.png", bbox_inches="tight")


class MetricTrackerCallback(pl.Callback):
    def __init__(self, task_type):
        super().__init__()
        self.task_type = task_type
        self.losses = {"train_loss": [], "val_loss": []}
        if task_type == 'FD':
            self.accuracies = {"train_acc": [], "val_acc": []}
        elif task_type == 'RUL':
            self.rmses = {"train_rmse": [], "val_rmse": []}

    def _get(self, trainer, key):
        # Try both the plain key and Lightning's *_epoch variant
        for k in (key, f"{key}_epoch"):
            if k in trainer.callback_metrics:
                v = trainer.callback_metrics[k]
                return v.item() if hasattr(v, "item") else float(v)

        # If missing, print available keys once to help debugging
        print(f"[MetricTrackerCallback] Missing key: {key}")
        print("[MetricTrackerCallback] Available keys:", list(trainer.callback_metrics.keys()))
        return None  # don't crash

    def on_validation_epoch_end(self, trainer, pl_module):
        v = self._get(trainer, "val_loss")
        if v is not None:
            self.losses["val_loss"].append(v)

        if self.task_type == 'FD':
            v = self._get(trainer, "val_acc")
            if v is not None:
                self.accuracies["val_acc"].append(v)
        elif self.task_type == 'RUL':
            v = self._get(trainer, "val_rmse")
            if v is not None:
                self.rmses["val_rmse"].append(v)

    def on_train_epoch_end(self, trainer, pl_module):
        v = self._get(trainer, "train_loss")
        if v is not None:
            self.losses["train_loss"].append(v)

        if self.task_type == 'FD':
            v = self._get(trainer, "train_acc")
            if v is not None:
                self.accuracies["train_acc"].append(v)
        elif self.task_type == 'RUL':
            v = self._get(trainer, "train_rmse")
            if v is not None:
                self.rmses["train_rmse"].append(v)


# ==================== Main ====================
def main(args):
    pl.seed_everything(args.random_seed)
    train_loader, val_loader, test_loader = get_datasets(args)

    # ================= X Channel Z-score (fit on TRAIN, apply to VAL/TEST) =================
    x_train = train_loader.dataset.x_data  # torch.Tensor [N, C, L]
    x_mean, x_std = fit_channel_zscore_torch(x_train, eps=1e-6)
    x_min, x_scale = fit_channel_minmax_torch(x_train, eps=1e-6)

    args.x_min = x_min.detach().cpu()
    args.x_scale = x_scale.detach().cpu()

    # store if you want to debug later
    args.x_mean = x_mean.detach().cpu()
    args.x_std  = x_std.detach().cpu()

    # apply to all splits using TRAIN stats
    # for ds in [train_loader.dataset, val_loader.dataset, test_loader.dataset]:
    #     ds.x_data = apply_channel_zscore_torch(ds.x_data, x_mean, x_std)

    # print("[X ZScore] per-channel mean/std (first 5 channels):")
    # print(" mean:", x_mean.view(-1)[:5].detach().cpu().numpy())
    # print(" std :", x_std.view(-1)[:5].detach().cpu().numpy())

    for ds in [train_loader.dataset, val_loader.dataset, test_loader.dataset]:
        ds.x_data = apply_channel_minmax11_torch(ds.x_data, x_min, x_scale)

    print("[X MinMax] first 5 channels:")
    print(" min  :", x_min.view(-1)[:5].cpu().numpy())
    print(" scale:", x_scale.view(-1)[:5].cpu().numpy())

    # ================= Label Z-score (RUL only) =================
    if args.task_type == "RUL":
        # fit on TRAIN only
        y_train = train_loader.dataset.y_data              # should be [N, C]
        if torch.is_tensor(y_train):
            y_train_np = y_train.detach().cpu().numpy()
        else:
            y_train_np = np.asarray(y_train)

        args.y_min, args.y_max, args.y_scale = minmax_fit_vec(y_train_np)

        def _to_np(a):
            return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
        
        train_loader.dataset.y_data = minmax_apply_pm1_vec(_to_np(train_loader.dataset.y_data), args.y_min, args.y_scale)
        val_loader.dataset.y_data   = minmax_apply_pm1_vec(_to_np(val_loader.dataset.y_data),   args.y_min, args.y_scale)
        test_loader.dataset.y_data  = minmax_apply_pm1_vec(_to_np(test_loader.dataset.y_data),  args.y_min, args.y_scale)
        
        train_loader.dataset.y_data = torch.tensor(train_loader.dataset.y_data, dtype=torch.float32)
        val_loader.dataset.y_data   = torch.tensor(val_loader.dataset.y_data, dtype=torch.float32)
        test_loader.dataset.y_data  = torch.tensor(test_loader.dataset.y_data, dtype=torch.float32)

        args.y_min   = torch.tensor(args.y_min, dtype=torch.float32)
        args.y_scale = torch.tensor(args.y_scale, dtype=torch.float32)
        print("[Label MinMax per-channel] y_min first5:", args.y_min[:5].numpy(), "y_scale first5:", args.y_scale[:5].numpy())
        ytr = np.asarray(train_loader.dataset.y_data)  # [N,C] normalized
        yte = np.asarray(test_loader.dataset.y_data)   # [N,C] normalized
        yhat = np.broadcast_to(ytr.mean(axis=0, keepdims=True), yte.shape)
        rmse_baseline = np.sqrt(np.mean((yhat - yte)**2))
        print("[Baseline per-channel mean predictor] RMSE:", rmse_baseline)

    # args extracted from the running dataset
    if args.task_type == 'FD':
        args.num_classes = len(np.unique(train_loader.dataset.y_data))
        args.class_names = [str(i) for i in range(args.num_classes)]
    else:
        args.num_classes = 1
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]
    # --- channel names for plotting (edit to your real names if you have them) ---
    args.y_channel_names = [
        "max_injection_pressure",
        "plastification_time",
        "end_of_packing_stroke",
        "switchover_pressure",
    ]

    args.tl_length = len(train_loader)

    # Callbacks
    run_description = construct_experiment_dir(args)
    print(f"========== {run_description} ===========")
    ckpt_dir = f"checkpoints/{run_description}"

    args.ckpt_dir = ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(
        {
            "x_min": args.x_min,
            "x_scale": args.x_scale,
            # optional: also save label scaling for RUL
            "y_min": getattr(args, "y_min", None),
            "y_scale": getattr(args, "y_scale", None),
        },
        os.path.join(args.ckpt_dir, "norm_stats.pt")
    )
    print("Saved norm stats to:", os.path.join(args.ckpt_dir, "norm_stats.pt"))


    # Set monitoring metric based on task type
    if args.task_type == 'FD':
        checkpoint = ModelCheckpoint(monitor="train_f1_epoch", mode="max", save_top_k=1, dirpath=ckpt_dir,
                                     filename="best")
        early_stop = EarlyStopping(monitor="train_f1_epoch", patience=args.patience, mode="max")
    elif args.task_type == 'RUL':
        # For RUL, use validation composite loss for model selection / early stopping.
        checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath=ckpt_dir, filename="best")
        early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")

    tracker = MetricTrackerCallback(args.task_type)

    save_copy_of_files(checkpoint)

    model = Model(args)

    # Optional load pretrained weights
    if args.load_from_pretrained and args.pretrained_model_type != 'mae':
        path = os.path.join(args.pretrained_model_dir, f"pretrain-epoch={args.pretraining_epoch_id}.ckpt")
        checkpoint_data = torch.load(path, map_location='cuda', weights_only=False)


        # Filter and count matching keys with the same shape
        matched_weights = {
            k: v for k, v in checkpoint_data['state_dict'].items()
            if k in model.state_dict() and model.state_dict()[k].size() == v.size()
        }

        total_pretrained = len(checkpoint_data['state_dict'])
        model.load_state_dict(matched_weights, strict=False)

        print(f"Loaded pretrained weights from {path}")
        print(f"Matched weights: {len(matched_weights)}/{len(model.state_dict())} model parameters matched "
              f"(from {total_pretrained} pretrained parameters)")
        print("")

    elif args.load_from_pretrained:  #
        path = os.path.join(args.pretrained_model_dir, f"pretrain-epoch={args.pretraining_epoch_id}.ckpt")
        checkpoint_data = torch.load(path, map_location='cuda', weights_only=False)
        checkpoint_state = checkpoint_data['state_dict']
        model_state = model.state_dict()

        remapped_weights = {}
        for ckpt_key, ckpt_value in checkpoint_state.items():
            # Fix the redundant nesting: "model.encoder.encoder." → "model.encoder."
            if ckpt_key.startswith("model.encoder.encoder."):
                new_key = "model.encoder." + ckpt_key[len("model.encoder.encoder."):]
            else:
                new_key = ckpt_key

            # Match if key exists and shape is the same
            if new_key in model_state and model_state[new_key].shape == ckpt_value.shape:
                remapped_weights[new_key] = ckpt_value

        model.load_state_dict(remapped_weights, strict=False)

        print(f"Loaded pretrained weights from {path}")
        print(f"Matched weights: {len(remapped_weights)}/{len(model_state)} model parameters matched "
              f"(from {len(checkpoint_state)} pretrained parameters)")

    trainer = pl.Trainer(
        default_root_dir=ckpt_dir,
        max_epochs=args.num_epochs,
        callbacks=[checkpoint, early_stop, tracker, TQDMProgressBar(refresh_rate=500)],
        accelerator="auto",
        precision='bf16-mixed',
        devices=[args.gpu_id],
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    if args.task_type == 'FD':
        plot_metrics(
            {"train_loss": tracker.losses["train_loss"], "val_loss": tracker.losses["val_loss"],
             "train_acc": tracker.accuracies["train_acc"], "val_acc": tracker.accuracies["val_acc"]},
            args.ckpt_dir,
            args.task_type
        )
    elif args.task_type == 'RUL':
        plot_metrics(
            {"train_loss": tracker.losses["train_loss"], "val_loss": tracker.losses["val_loss"],
             "train_rmse": tracker.rmses["train_rmse"], "val_rmse": tracker.rmses["val_rmse"]},
            args.ckpt_dir,
            args.task_type
        )

def apply_model_config(args):
    config_map = {
        'tiny':  {'embed_dim': 128, 'heads': 4,  'depth': 4},
        'small': {'embed_dim': 256, 'heads': 8,  'depth': 8},
        'base':  {'embed_dim': 512, 'heads': 12, 'depth': 16},
    }
    config = config_map[args.model_type]
    for k, v in config.items():
        setattr(args, k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=r'./dataset/')
    parser.add_argument('--data_id', type=str, default=r'M01', help= 'choose [M01, M02,M03] for FD task and [FEMTO] for RUL task')
    parser.add_argument('--data_percentage', type=str, default="1")
    parser.add_argument('--model_id', type=str, default="CNC_FT", help= 'CNC_FT or FEMTO_FT')

    parser.add_argument('--model_type', type=str, choices=['tiny', 'small', 'base'], default='tiny')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_moe', type=str2bool, default=False, help='[use MoE or default]')

    parser.add_argument('--load_from_pretrained', type=str2bool, default=True)
    parser.add_argument('--pretrained_model_dir', type=str, default="pretrained_models/Tiny")
    parser.add_argument('--pretraining_epoch_id', type=int, default=1)
    parser.add_argument('--pretrained_model_type', type=str, default='normal', help='model can be [normal, mae]')

    parser.add_argument('--num_epochs', type=int, default=600)
    parser.add_argument('--patience', type=int, default=50, help="For early stopping")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4) #1e-3
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--task_type',type=str,default='FD',choices=['FD', 'RUL'])

    # RUL composite loss weights
    parser.add_argument('--huber_beta', type=float, default=1.0)
    parser.add_argument('--point_loss_weight', type=float, default=1.0)
    parser.add_argument('--diff_loss_weight', type=float, default=0.5)
    parser.add_argument('--curvature_loss_weight', type=float, default=0.25)
    parser.add_argument('--variance_loss_weight', type=float, default=0.1)

    args = parser.parse_args()
    apply_model_config(args)
    main(args)

    

