"""C3: Cross-attention model -- target ticker attends to correlated tickers.

Instead of concatenating all features, this model:
1. Embeds target, correlates, and volume features separately
2. Applies self-attention over time on target
3. Cross-attends: target queries, correlates as keys/values
4. Gated fusion of all streams -> prediction
"""
import math

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .base import PredictionModel
from .gan import pinball_loss_torch


class GatedResidualNetwork(nn.Module):
    """GRN: LayerNorm(x + GLU(W1 * ELU(W2 * x)))"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model * 2)  # for GLU
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.nn.functional.elu(self.fc1(x))
        h = self.drop(h)
        gate_input = self.fc2(h)
        h = torch.nn.functional.glu(gate_input, dim=-1)
        return self.norm(x + h)


class CrossAttentionNet(nn.Module):
    """Network with separate streams for target/correlates + cross-attention."""

    def __init__(self, n_target, n_corr, n_other, d_model=64, nhead=4,
                 num_layers=2, dropout=0.1, num_quantiles=3):
        super().__init__()
        self.d_model = d_model

        self.target_proj = nn.Linear(n_target, d_model)
        self.corr_proj = nn.Linear(max(n_corr, 1), d_model)
        self.other_proj = nn.Linear(max(n_other, 1), d_model) if n_other > 0 else None

        # Positional encoding
        pe = torch.zeros(512, d_model)
        pos = torch.arange(0, 512, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

        # Self-attention on target stream
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.self_attn = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Cross-attention: Q=target, K/V=correlates
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # Fusion GRN
        fusion_in = d_model * (3 if n_other > 0 else 2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.grn = GatedResidualNetwork(d_model, dropout)

        # Output heads
        self.head_reg = nn.Linear(d_model, 1)
        self.head_cls = nn.Linear(d_model, 1)
        self.head_q = nn.Linear(d_model, num_quantiles)

    def forward(self, x_target, x_corr, x_other=None):
        B, T, _ = x_target.shape
        pe = self.pe[:, :T, :]

        tgt = self.target_proj(x_target) + pe
        corr = self.corr_proj(x_corr) + pe

        tgt = self.self_attn(tgt)

        cross_out, _ = self.cross_attn(tgt, corr, corr)
        tgt = self.cross_norm(tgt + cross_out)

        h_last_tgt = tgt[:, -1, :]
        h_last_corr = corr[:, -1, :]

        if x_other is not None and self.other_proj is not None:
            other = self.other_proj(x_other)
            h_last_other = other[:, -1, :]
            h_cat = torch.cat([h_last_tgt, h_last_corr, h_last_other], dim=-1)
        else:
            h_cat = torch.cat([h_last_tgt, h_last_corr], dim=-1)

        h = self.fusion(h_cat)
        h = self.grn(h)

        y_reg = self.head_reg(h).squeeze(-1)
        y_cls = self.head_cls(h).squeeze(-1)
        y_q = self.head_q(h)
        return y_reg, y_cls, y_q


class CrossAttentionModel(PredictionModel):
    """C3: Cross-attention between target ticker and correlates."""

    def __init__(self, input_size, seq_len, cfg, col_split=None):
        """
        Args:
            col_split: dict with keys "target_idx", "corr_idx", "other_idx"
                       containing lists of column indices for each stream.
                       If None, assumes first column is target, rest are correlates.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantiles = cfg.quantiles
        self.l1_weight = cfg.l1_weight
        self.cls_weight = cfg.cls_weight
        self.q_weight = cfg.q_weight
        self.grad_clip = cfg.grad_clip

        if col_split is None:
            self.target_idx = [0]
            self.corr_idx = list(range(1, input_size))
            self.other_idx = []
        else:
            self.target_idx = col_split["target_idx"]
            self.corr_idx = col_split["corr_idx"]
            self.other_idx = col_split.get("other_idx", [])

        n_target = len(self.target_idx)
        n_corr = len(self.corr_idx)
        n_other = len(self.other_idx)

        self.net = CrossAttentionNet(
            n_target, n_corr, n_other,
            d_model=cfg.d_model, nhead=cfg.nhead,
            num_layers=cfg.num_layers_tst, dropout=cfg.dropout_tst,
            num_quantiles=len(cfg.quantiles),
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=cfg.lr_g,
            betas=(0.9, 0.99), weight_decay=1e-4,
        )

        self.reg_loss = nn.HuberLoss(delta=1.0)
        self.bce = nn.BCEWithLogitsLoss()

        _is_cuda = self.device == "cuda"
        self._amp_enabled = _is_cuda
        self.scaler = torch.amp.GradScaler(self.device, enabled=self._amp_enabled)

        self.scheduler = None
        if cfg.use_lr_scheduler and cfg.n_epochs > 1:
            eta_min = cfg.lr_g * cfg.lr_scheduler_min_factor
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=cfg.n_epochs, eta_min=eta_min,
            )

    def _split_input(self, xb):
        x_target = xb[:, :, self.target_idx]
        x_corr = xb[:, :, self.corr_idx]
        x_other = xb[:, :, self.other_idx] if self.other_idx else None
        return x_target, x_corr, x_other

    def train_epoch(self, train_loader):
        self.net.train()
        losses = []
        autocast = torch.amp.autocast
        for xb, yb in train_loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            x_tgt, x_corr, x_other = self._split_input(xb)

            with autocast(self.device, enabled=self._amp_enabled):
                y_reg, y_cls, y_q = self.net(x_tgt, x_corr, x_other)
                loss = self.l1_weight * self.reg_loss(y_reg, yb)
                if self.cls_weight > 0:
                    y_dir = (yb > 0).float()
                    loss = loss + self.cls_weight * self.bce(y_cls, y_dir)
                if self.q_weight > 0:
                    loss = loss + self.q_weight * pinball_loss_torch(
                        yb, y_q, self.quantiles,
                    )

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses.append(loss.item())

        avg = sum(losses) / len(losses)
        return {"loss": avg, "g_loss": avg, "d_loss": 0.0}

    def predict(self, xb):
        self.net.eval()
        with torch.no_grad():
            xb = xb.to(self.device)
            x_tgt, x_corr, x_other = self._split_input(xb)
            y_reg, y_cls, y_q = self.net(x_tgt, x_corr, x_other)
        return y_reg.cpu(), y_cls.cpu(), y_q.cpu()

    def step_schedulers(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def get_state_dict(self):
        return {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, state):
        self.net.load_state_dict(state)
