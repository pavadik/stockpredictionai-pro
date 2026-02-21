import torch
from torch import nn, autograd
from torch.optim.lr_scheduler import CosineAnnealingLR
from .generator import LSTMGenerator, TransformerGenerator
from .discriminator import CNNDiscriminator


def gradient_penalty(discriminator, real_x, real_y, fake_y, device):
    """WGAN-GP gradient penalty on interpolated y (conditional discriminator).

    Interpolation is between real_y and fake_y (the discriminator's conditional input).
    Gradients are computed w.r.t. the interpolated y.
    """
    alpha = torch.rand(real_y.size(0), device=device)
    interp_y = (alpha * real_y + (1 - alpha) * fake_y).requires_grad_(True)
    d_interpolates = discriminator(real_x, interp_y)
    ones = torch.ones_like(d_interpolates, device=device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interp_y,
        grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gp = ((gradients.abs() - 1) ** 2).mean()
    return gp


def pinball_loss_torch(y_true, y_quantiles, quantiles):
    """Differentiable pinball (quantile) loss for training.

    Args:
        y_true: [B] true values
        y_quantiles: [B, Q] predicted quantile values
        quantiles: tuple of Q quantile levels, e.g. (0.1, 0.5, 0.9)
    """
    total = torch.tensor(0.0, device=y_true.device)
    for i, q in enumerate(quantiles):
        err = y_true - y_quantiles[:, i]
        total = total + torch.mean(torch.max(q * err, (q - 1) * err))
    return total / len(quantiles)


class WGAN_GP:
    def __init__(self, input_size, seq_len, hidden_size=64, num_layers=1,
                 lr_g=1e-3, lr_d=1e-4, critic_steps=5, device=None,
                 generator_type="lstm", d_model=64, nhead=4,
                 num_layers_tst=2, dropout_tst=0.1, dropout_lstm=0.1,
                 num_quantiles=3,
                 quantiles=(0.1, 0.5, 0.9),
                 adv_weight=1.0, l1_weight=0.4, cls_weight=0.2, q_weight=0.3,
                 use_lr_scheduler=True, lr_scheduler_min_factor=0.1, n_epochs=20,
                 grad_clip=1.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantiles = quantiles
        self.adv_weight = adv_weight
        self.l1_weight = l1_weight
        self.cls_weight = cls_weight
        self.q_weight = q_weight
        self.grad_clip = grad_clip

        # Build generator
        if generator_type == "tst":
            self.G = TransformerGenerator(
                input_size, d_model=d_model, nhead=nhead,
                num_layers=num_layers_tst, dropout=dropout_tst,
                num_quantiles=num_quantiles,
            ).to(self.device)
        else:
            self.G = LSTMGenerator(
                input_size, hidden_size, num_layers,
                num_quantiles=num_quantiles,
                dropout=dropout_lstm,
            ).to(self.device)

        self.D = CNNDiscriminator(input_size, seq_len).to(self.device)

        # torch.compile with inductor backend requires Triton (Linux only as of 2025).
        # On Windows or when Triton is missing, skip compilation silently.
        _is_cuda = (self.device.startswith("cuda")
                    if isinstance(self.device, str)
                    else (self.device.type == "cuda"))
        self._compiled = False
        if _is_cuda:
            try:
                import triton  # noqa: F401
                self.G = torch.compile(self.G)
                self.D = torch.compile(self.D)
                self._compiled = True
            except (ImportError, Exception):
                pass

        self.opt_g = torch.optim.AdamW(self.G.parameters(), lr=lr_g, betas=(0.9, 0.99), weight_decay=1e-4)
        self.opt_d = torch.optim.AdamW(self.D.parameters(), lr=lr_d, betas=(0.9, 0.99), weight_decay=1e-4)
        self.critic_steps = critic_steps
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()

        self._amp_enabled = _is_cuda
        self.scaler = torch.amp.GradScaler(self.device, enabled=self._amp_enabled)

        # Learning rate schedulers (CosineAnnealing: decays to eta_min then restarts)
        self.sched_g = None
        self.sched_d = None
        if use_lr_scheduler and n_epochs > 1:
            eta_min_g = lr_g * lr_scheduler_min_factor
            eta_min_d = lr_d * lr_scheduler_min_factor
            self.sched_g = CosineAnnealingLR(self.opt_g, T_max=n_epochs, eta_min=eta_min_g)
            self.sched_d = CosineAnnealingLR(self.opt_d, T_max=n_epochs, eta_min=eta_min_d)

    def step_schedulers(self):
        """Advance LR schedulers by one epoch. Safe to call even if schedulers are disabled."""
        if self.sched_g is not None:
            self.sched_g.step()
        if self.sched_d is not None:
            self.sched_d.step()

    def train_epoch(self, train_loader, lambda_gp=10.0):
        self.G.train()
        self.D.train()
        g_losses, d_losses = [], []
        autocast = torch.amp.autocast

        for xb, yb in train_loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            # --- Train D (critic) ---
            for _ in range(self.critic_steps):
                with autocast(self.device, enabled=self._amp_enabled):
                    y_fake_reg, _, _ = self.G(xb)
                    d_real = self.D(xb, yb)
                    d_fake = self.D(xb, y_fake_reg.detach())
                    gp = gradient_penalty(self.D, xb, yb, y_fake_reg.detach(), self.device)
                    d_loss = -(d_real.mean() - d_fake.mean()) + lambda_gp * gp
                self.opt_d.zero_grad(set_to_none=True)
                self.scaler.scale(d_loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.opt_d)
                    nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip)
                self.scaler.step(self.opt_d)
                self.scaler.update()
            d_losses.append(d_loss.item())

            # --- Train G ---
            with autocast(self.device, enabled=self._amp_enabled):
                y_gen_reg, y_gen_logit, y_gen_q = self.G(xb)
                d_fake = self.D(xb, y_gen_reg)
                g_adv = -d_fake.mean()
                g_l1 = self.l1(y_gen_reg, yb)
                y_dir = (yb > 0).float()
                g_cls = self.bce(y_gen_logit, y_dir)
                g_q = pinball_loss_torch(yb, y_gen_q, self.quantiles)
                g_loss = (self.adv_weight * g_adv
                          + self.l1_weight * g_l1
                          + self.cls_weight * g_cls
                          + self.q_weight * g_q)
            self.opt_g.zero_grad(set_to_none=True)
            self.scaler.scale(g_loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.opt_g)
                nn.utils.clip_grad_norm_(self.G.parameters(), self.grad_clip)
            self.scaler.step(self.opt_g)
            self.scaler.update()
            g_losses.append(g_loss.item())

        return float(sum(g_losses) / len(g_losses)), float(sum(d_losses) / len(d_losses))

    def predict(self, xb):
        self.G.eval()
        with torch.no_grad():
            xb = xb.to(self.device)
            y_reg, y_logit, y_q = self.G(xb)
        return y_reg.cpu(), y_logit.cpu(), y_q.cpu()
