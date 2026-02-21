"""C2: Direction classifier -- predict up/down (or up/flat/down) instead of regression.

Optimizes directly for direction accuracy via CrossEntropy or FocalLoss.
"""
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from .base import PredictionModel


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (Lin et al. 2017)."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class LSTMClassifier(nn.Module):
    """LSTM backbone with classification head. No noise injection."""

    def __init__(self, input_size, hidden_size=64, num_layers=1,
                 n_classes=2, dropout=0.1):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])
        return self.head(h)


class DirectionClassifier(PredictionModel):
    """Classification model: predict direction (up/down or up/flat/down)."""

    def __init__(self, input_size, seq_len, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_classes = cfg.n_classes
        self.threshold = cfg.cls_threshold
        self.grad_clip = cfg.grad_clip

        self.net = LSTMClassifier(
            input_size, hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers, n_classes=self.n_classes,
            dropout=cfg.dropout_lstm,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=cfg.lr_g,
            betas=(0.9, 0.99), weight_decay=1e-4,
        )

        if cfg.loss_fn == "focal":
            self.criterion = FocalLoss(gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss()

        _is_cuda = self.device == "cuda"
        self._amp_enabled = _is_cuda
        self.scaler = torch.amp.GradScaler(self.device, enabled=self._amp_enabled)

        self.scheduler = None
        if cfg.use_lr_scheduler and cfg.n_epochs > 1:
            eta_min = cfg.lr_g * cfg.lr_scheduler_min_factor
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=cfg.n_epochs, eta_min=eta_min,
            )

    def _to_labels(self, yb):
        """Convert continuous deltas to class labels."""
        if self.n_classes == 2:
            return (yb > self.threshold).long()
        else:
            labels = torch.ones_like(yb, dtype=torch.long)  # 1 = flat
            labels[yb > self.threshold] = 2   # up
            labels[yb < -self.threshold] = 0  # down
            return labels

    def train_epoch(self, train_loader):
        self.net.train()
        losses = []
        autocast = torch.amp.autocast
        for xb, yb in train_loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            labels = self._to_labels(yb)

            with autocast(self.device, enabled=self._amp_enabled):
                logits = self.net(xb)
                loss = self.criterion(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        return {"loss": avg_loss, "g_loss": avg_loss, "d_loss": 0.0}

    def predict(self, xb):
        self.net.eval()
        with torch.no_grad():
            xb = xb.to(self.device)
            logits = self.net(xb)
            probs = torch.softmax(logits, dim=-1)

            if self.n_classes == 2:
                y_reg = probs[:, 1] - probs[:, 0]
            else:
                y_reg = probs[:, 2] - probs[:, 0]

            y_cls = logits[:, -1] if logits.shape[1] > 1 else logits[:, 0]
            n_q = 3
            y_q = y_reg.unsqueeze(1).expand(-1, n_q)

        return y_reg.cpu(), y_cls.cpu(), y_q.cpu()

    def step_schedulers(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def get_state_dict(self):
        return {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, state):
        self.net.load_state_dict(state)
