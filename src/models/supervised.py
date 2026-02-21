"""C1: Supervised model -- direct regression without GAN wrapper.

Reuses existing LSTMGenerator/TransformerGenerator but trains with
Huber/MSE + optional classification + quantile losses.
No discriminator, no adversarial training.
"""
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .base import PredictionModel
from .generator import LSTMGenerator, TransformerGenerator
from .gan import pinball_loss_torch


class SupervisedModel(PredictionModel):
    """Supervised wrapper around existing generator architectures."""

    def __init__(self, input_size, seq_len, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantiles = cfg.quantiles
        self.l1_weight = cfg.l1_weight
        self.cls_weight = cfg.cls_weight
        self.q_weight = cfg.q_weight
        self.grad_clip = cfg.grad_clip
        self.loss_fn_name = cfg.loss_fn

        if cfg.generator == "tst":
            self.net = TransformerGenerator(
                input_size, d_model=cfg.d_model, nhead=cfg.nhead,
                num_layers=cfg.num_layers_tst, dropout=cfg.dropout_tst,
                num_quantiles=len(cfg.quantiles),
            ).to(self.device)
        else:
            self.net = LSTMGenerator(
                input_size, cfg.hidden_size, cfg.num_layers,
                num_quantiles=len(cfg.quantiles),
                dropout=cfg.dropout_lstm,
            ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=cfg.lr_g,
            betas=(0.9, 0.99), weight_decay=1e-4,
        )

        if cfg.loss_fn == "huber":
            self.reg_loss = nn.HuberLoss(delta=1.0)
        else:
            self.reg_loss = nn.MSELoss()
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

    def train_epoch(self, train_loader):
        self.net.train()
        losses = []
        autocast = torch.amp.autocast
        for xb, yb in train_loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            with autocast(self.device, enabled=self._amp_enabled):
                y_reg, y_cls, y_q = self.net(xb)
                loss_reg = self.reg_loss(y_reg, yb)
                loss = self.l1_weight * loss_reg

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

        avg_loss = sum(losses) / len(losses)
        return {"loss": avg_loss, "g_loss": avg_loss, "d_loss": 0.0}

    def predict(self, xb):
        self.net.eval()
        with torch.no_grad():
            xb = xb.to(self.device)
            y_reg, y_cls, y_q = self.net(xb)
        return y_reg.cpu(), y_cls.cpu(), y_q.cpu()

    def step_schedulers(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def get_state_dict(self):
        return {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, state):
        self.net.load_state_dict(state)
