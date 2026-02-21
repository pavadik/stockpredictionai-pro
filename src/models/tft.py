"""C4: Simplified Temporal Fusion Transformer.

Key components:
- Variable Selection Network (VSN): learns which features matter
- Gated Residual Network (GRN): building block
- LSTM encoder (not Transformer -- proven better on 17K bars)
- Interpretable Multi-Head Attention over LSTM output
- Single-step prediction with regression + quantile heads
"""
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .base import PredictionModel
from .classifier import FocalLoss
from .gan import pinball_loss_torch


class GatedResidualNetwork(nn.Module):
    """GRN(x, c=None) = LayerNorm(x + GLU(W1 * ELU(W2 * x + W3 * c)))

    When context c is provided, it conditions the gating.
    """

    def __init__(self, d_input, d_hidden, d_output=None, d_context=0, dropout=0.1):
        super().__init__()
        d_output = d_output or d_input
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.fc_context = nn.Linear(d_context, d_hidden, bias=False) if d_context > 0 else None
        self.fc2 = nn.Linear(d_hidden, d_output * 2)  # for GLU
        self.norm = nn.LayerNorm(d_output)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(d_input, d_output) if d_input != d_output else nn.Identity()

    def forward(self, x, context=None):
        skip = self.skip(x)
        h = self.fc1(x)
        if self.fc_context is not None and context is not None:
            h = h + self.fc_context(context)
        h = torch.nn.functional.elu(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = torch.nn.functional.glu(h, dim=-1)
        return self.norm(skip + h)


class VariableSelectionNetwork(nn.Module):
    """VSN: per-variable GRN + softmax attention -> weighted combination.

    Input: [B, T, F] where F = n_vars
    Output: [B, T, d_model] weighted representation + var_weights [F]
    """

    def __init__(self, n_vars, d_model, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model

        # Each variable gets its own embedding
        self.var_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_vars)
        ])

        # GRN to compute variable weights
        self.weight_grn = GatedResidualNetwork(
            d_input=n_vars * d_model, d_hidden=d_model, d_output=n_vars,
            dropout=dropout,
        )

        # Per-variable GRN
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model, dropout=dropout)
            for _ in range(n_vars)
        ])

    def forward(self, x):
        # x: [B, T, F]
        B, T, F = x.shape

        # Embed each variable separately
        var_embeds = []
        for i in range(self.n_vars):
            emb = self.var_embeddings[i](x[:, :, i:i+1])  # [B, T, d_model]
            emb = self.var_grns[i](emb.reshape(B * T, -1)).reshape(B, T, -1)
            var_embeds.append(emb)

        # Stack: [B, T, n_vars, d_model]
        stacked = torch.stack(var_embeds, dim=2)

        # Flatten for weight computation: [B*T, n_vars * d_model]
        flat = stacked.reshape(B * T, -1)
        weights = self.weight_grn(flat)  # [B*T, n_vars]
        weights = torch.softmax(weights, dim=-1)  # [B*T, n_vars]
        weights_3d = weights.reshape(B, T, self.n_vars, 1)

        # Weighted sum: [B, T, d_model]
        output = (stacked * weights_3d).sum(dim=2)

        # Average weights for interpretability
        avg_weights = weights.reshape(B, T, self.n_vars).mean(dim=(0, 1))
        return output, avg_weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention that outputs interpretable attention weights."""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_k)  # single value head for interpretability
        self.out_proj = nn.Linear(self.d_k, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, T, _ = q.shape
        Q = self.W_q(q).reshape(B, T, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(k).reshape(B, T, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(v).unsqueeze(1).expand(-1, self.nhead, -1, -1)

        scale = self.d_k ** 0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)

        # Average across heads for interpretability
        attn_avg = attn.mean(dim=1)  # [B, T, T]
        out = torch.matmul(attn_avg, V[:, 0, :, :])  # [B, T, d_k]
        return self.out_proj(out), attn_avg


class SimplifiedTFTNet(nn.Module):
    """Simplified TFT network."""

    def __init__(self, input_size, d_model=32, nhead=2, num_lstm_layers=1,
                 dropout=0.1, num_quantiles=3, n_horizons=1, n_classes=2):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_classes = n_classes
        self.vsn = VariableSelectionNetwork(input_size, d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=num_lstm_layers,
                            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0.0)
        self.post_lstm_grn = GatedResidualNetwork(d_model, d_model, dropout=dropout)
        self.attention = InterpretableMultiHeadAttention(d_model, nhead, dropout)
        self.post_attn_grn = GatedResidualNetwork(d_model, d_model, dropout=dropout)
        self.post_attn_norm = nn.LayerNorm(d_model)
        self.output_grn = GatedResidualNetwork(d_model, d_model, dropout=dropout)
        self.head_reg = nn.Linear(d_model, n_horizons)
        cls_out = n_classes if n_classes >= 3 else 1
        self.head_cls = nn.Linear(d_model, cls_out)
        self.head_q = nn.Linear(d_model, n_horizons * num_quantiles)
        self.num_quantiles = num_quantiles

    def forward(self, x, return_weights=False):
        # Variable selection
        vsn_out, var_weights = self.vsn(x)  # [B, T, d_model]

        # Temporal processing
        lstm_out, _ = self.lstm(vsn_out)  # [B, T, d_model]
        lstm_out = self.post_lstm_grn(
            lstm_out.reshape(-1, lstm_out.size(-1))
        ).reshape(lstm_out.shape)

        # Interpretable attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.post_attn_grn(
            attn_out.reshape(-1, attn_out.size(-1))
        ).reshape(attn_out.shape)
        h_seq = self.post_attn_norm(lstm_out + attn_out)

        # Last time step
        h = h_seq[:, -1, :]
        h = self.output_grn(h)

        y_reg = self.head_reg(h)  # [B, n_horizons]
        if self.n_horizons == 1:
            y_reg = y_reg.squeeze(-1)  # [B] for backward compat
        y_cls_raw = self.head_cls(h)  # [B, n_classes] or [B, 1]
        y_cls = y_cls_raw if self.n_classes >= 3 else y_cls_raw.squeeze(-1)
        y_q_raw = self.head_q(h)  # [B, n_horizons * num_quantiles]
        if self.n_horizons == 1:
            y_q = y_q_raw  # [B, Q]
        else:
            y_q = y_q_raw.reshape(-1, self.n_horizons, self.num_quantiles)  # [B, H, Q]

        if return_weights:
            return y_reg, y_cls, y_q, var_weights, attn_weights
        return y_reg, y_cls, y_q


class SimplifiedTFT(PredictionModel):
    """C4: Simplified Temporal Fusion Transformer."""

    def __init__(self, input_size, seq_len, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantiles = cfg.quantiles
        self.l1_weight = cfg.l1_weight
        self.cls_weight = cfg.cls_weight
        self.q_weight = cfg.q_weight
        self.grad_clip = cfg.grad_clip
        self.input_size = input_size
        self.n_horizons = len(getattr(cfg, 'forecast_horizons', (1,)))
        self.n_classes = getattr(cfg, 'n_classes', 2)
        self.threshold = getattr(cfg, 'cls_threshold', 0.0)

        self.net = SimplifiedTFTNet(
            input_size, d_model=cfg.d_model, nhead=cfg.nhead,
            num_lstm_layers=cfg.num_layers, dropout=cfg.dropout_tst,
            num_quantiles=len(cfg.quantiles),
            n_horizons=self.n_horizons,
            n_classes=self.n_classes,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=cfg.lr_g,
            betas=(0.9, 0.99), weight_decay=1e-4,
        )

        self.reg_loss = nn.HuberLoss(delta=1.0)
        if self.n_classes >= 3:
            loss_fn = getattr(cfg, 'loss_fn', 'huber')
            if loss_fn == "focal":
                self.cls_criterion = FocalLoss(gamma=2.0)
            else:
                self.cls_criterion = nn.CrossEntropyLoss()
        else:
            self.cls_criterion = nn.BCEWithLogitsLoss()

        _is_cuda = self.device == "cuda"
        self._amp_enabled = _is_cuda
        self.scaler = torch.amp.GradScaler(self.device, enabled=self._amp_enabled)

        self.scheduler = None
        if cfg.use_lr_scheduler and cfg.n_epochs > 1:
            eta_min = cfg.lr_g * cfg.lr_scheduler_min_factor
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=cfg.n_epochs, eta_min=eta_min,
            )

        self._last_var_weights = None

    def _to_labels(self, yb):
        """Convert continuous deltas to class labels (same logic as DirectionClassifier)."""
        if self.n_classes == 2:
            return (yb > self.threshold).long()
        labels = torch.ones_like(yb, dtype=torch.long)  # 1 = flat
        labels[yb > self.threshold] = 2   # up
        labels[yb < -self.threshold] = 0  # down
        return labels

    def train_epoch(self, train_loader):
        self.net.train()
        losses = []
        autocast = torch.amp.autocast
        is_multi = self.n_horizons > 1
        use_multiclass = self.n_classes >= 3
        for xb, yb in train_loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            with autocast(self.device, enabled=self._amp_enabled):
                y_reg, y_cls, y_q = self.net(xb)

                loss = torch.tensor(0.0, device=self.device)

                if self.l1_weight > 0:
                    loss = loss + self.l1_weight * self.reg_loss(y_reg, yb)

                if self.cls_weight > 0:
                    yb_cls = yb[:, 0] if is_multi else yb
                    if use_multiclass:
                        labels = self._to_labels(yb_cls)
                        loss = loss + self.cls_weight * self.cls_criterion(y_cls, labels)
                    else:
                        y_dir = (yb_cls > 0).float()
                        loss = loss + self.cls_weight * self.cls_criterion(y_cls, y_dir)

                if self.q_weight > 0 and not is_multi:
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
            y_reg, y_cls, y_q, var_w, _ = self.net(xb, return_weights=True)
            self._last_var_weights = var_w.cpu().numpy()

            if self.n_classes >= 3:
                probs = torch.softmax(y_cls, dim=-1)
                y_reg_out = probs[:, 2] - probs[:, 0]
                y_cls_out = y_cls[:, -1]
                n_q = len(self.quantiles)
                y_q_out = y_reg_out.unsqueeze(1).expand(-1, n_q)
                return y_reg_out.cpu(), y_cls_out.cpu(), y_q_out.cpu()

        return y_reg.cpu(), y_cls.cpu(), y_q.cpu()

    def get_var_weights(self):
        return self._last_var_weights

    def step_schedulers(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def get_state_dict(self):
        return {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, state):
        self.net.load_state_dict(state)
