# model.py
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .Transformer_utils import FullAttention, AttentionLayer, Encoder, EncoderLayer


# ------------ AUGMENTATIONS for Contrastive learning -------------------
def time_shift(x, shift_ratio=0.2):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    signal_length = x.shape[2]
    shift = int(signal_length * shift_ratio)
    shifted_sample = torch.cat((x[:, :, signal_length - shift:], x[:, :, :signal_length - shift]), dim=2)
    return shifted_sample


def scaling_with_jitter(x, sigma=0.05):
    factor = torch.normal(mean=1.0, std=sigma, size=(x.shape[0], x.shape[2]), device=x.device)
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append((xi * factor[:, :]).unsqueeze(1))
    return torch.cat(ai, dim=1)


class PatchEmbed(nn.Module):
    """
    Input: x (B, C, L)
    Output: (B*C, P, D)
    """
    def __init__(self, seq_len, patch_size=16, stride=16, embed_dim=768):
        super().__init__()
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches

        self.kernel = patch_size
        self.stride = stride
        self.input_layer = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.kernel, step=self.stride)  # (B,C,P,patch)
        x = rearrange(x, 'b c p k -> (b c) p k')
        x = self.input_layer(x)  # (B*C,P,D)
        return x


class Transformer_bkbone(L.LightningModule):
    """
    - forward() returns token features (B, C*P, D)
    - predict() returns:
        FD : (B, num_classes)
        RUL: (B, C, H)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len,
            patch_size=args.patch_size,
            stride=args.patch_size,
            embed_dim=args.embed_dim
        )
        P = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, P, args.embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=args.dropout, output_attention=False),
                        args.embed_dim,
                        args.heads
                    ),
                    args.embed_dim,
                    4 * args.embed_dim,
                    dropout=args.dropout,
                    activation="gelu"
                )
                for _ in range(args.depth)
            ],
            norm_layer=nn.LayerNorm(args.embed_dim),
        )

        # Pretraining-only layers kept for checkpoint compatibility
        self.input_layer = nn.Linear(args.patch_size, args.embed_dim)
        self.pretrain_head = nn.Linear(args.embed_dim, args.patch_size)

        # Freeze pretrain-only layers during supervised/forecast training (avoids DDP unused-params issues)
        if getattr(args, "freeze_pretrain_layers", True) and args.task_type in ("FD", "RUL"):
            for p in self.input_layer.parameters():
                p.requires_grad = False
            for p in self.pretrain_head.parameters():
                p.requires_grad = False

        # Heads
        # NOTE: we use mean+max pooling => per-channel feature dim = 2*embed_dim
        per_ch_dim = 2 * args.embed_dim

        # FD head mixes channels
        self.cls_in_dim = args.num_channels * per_ch_dim
        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.cls_in_dim),
            nn.Linear(self.cls_in_dim, 256),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, args.num_classes),
        )

        # RUL/forecast head outputs H values per channel
        self.reg_head = nn.Sequential(
            nn.LayerNorm(per_ch_dim),
            nn.Linear(per_ch_dim, 256),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, args.horizon),
        )

    def forward(self, x):
        # x -> (B*C, P, D)
        x_patch = self.patch_embed(x)
        x_patch = x_patch + self.pos_embed
        x_patch = self.pos_drop(x_patch)

        feats, _ = self.encoder(x_patch)  # (B*C, P, D)

        # reshape to (B, C*P, D)
        feats = torch.reshape(feats, (-1, self.args.num_channels * feats.shape[-2], feats.shape[-1]))
        return feats

    def predict(self, features):
        """
        features: (B, C*P, D)
        returns:
          FD : (B, num_classes)
          RUL: (B, C, H)
        """
        B, T, D = features.shape
        C = self.args.num_channels
        P = T // C

        f = features.view(B, C, P, D)          # (B,C,P,D)
        f_mean = f.mean(dim=2)                 # (B,C,D)
        f_max = f.max(dim=2).values            # (B,C,D)
        f = torch.cat([f_mean, f_max], dim=-1) # (B,C,2D)

        if self.args.task_type == "FD":
            z = f.reshape(B, C * f.shape[-1])  # (B, C*2D)
            return self.cls_head(z)
        else:
            return self.reg_head(f)            # (B,C,H)

    # Optional pretrain funcs kept
    def cl_pretrain(self, x):
        x_aug1 = time_shift(x)
        x_aug2 = scaling_with_jitter(x, sigma=0.2)

        features1 = self.forward(x_aug1)
        features2 = self.forward(x_aug2)

        features1 = self.predict(features1)
        features2 = self.predict(features2)

        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        return features1, features2