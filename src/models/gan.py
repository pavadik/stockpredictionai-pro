import torch
from torch import nn, autograd
from .generator import LSTMGenerator
from .discriminator import CNNDiscriminator

def gradient_penalty(discriminator, real_x, real_y, device):
    # GP по входу x (условный дискриминатор получает (x, y))
    alpha = torch.rand(real_x.size(0), 1, 1, device=device)
    interpolates = alpha * real_x + (1 - alpha) * real_x  # x ~ real (нет генерации x)
    interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates, real_y)
    ones = torch.ones_like(d_interpolates, device=device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

class WGAN_GP:
    def __init__(self, input_size, seq_len, hidden_size=64, num_layers=1, lr_g=1e-3, lr_d=1e-4, critic_steps=5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.G = LSTMGenerator(input_size, hidden_size, num_layers).to(self.device)
        self.D = CNNDiscriminator(input_size, seq_len).to(self.device)
        # torch.compile (ускорение в PyTorch 2.x)
        try:
            self.G = torch.compile(self.G)
            self.D = torch.compile(self.D)
        except Exception:
            pass
        # Оптимизаторы AdamW
        self.opt_g = torch.optim.AdamW(self.G.parameters(), lr=lr_g, betas=(0.9, 0.99), weight_decay=1e-4)
        self.opt_d = torch.optim.AdamW(self.D.parameters(), lr=lr_d, betas=(0.9, 0.99), weight_decay=1e-4)
        self.critic_steps = critic_steps
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.startswith("cuda"))

    def train_epoch(self, train_loader, lambda_gp=10.0):
        self.G.train(); self.D.train()
        g_losses, d_losses = [], []
        autocast = torch.cuda.amp.autocast
        for xb, yb in train_loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            # --- Train D (critic) ---
            for _ in range(self.critic_steps):
                with autocast(enabled=self.scaler.is_enabled()):
                    y_fake_reg, _ = self.G(xb)
                    d_real = self.D(xb, yb)
                    d_fake = self.D(xb, y_fake_reg.detach())
                    gp = gradient_penalty(self.D, xb, yb, self.device)
                    d_loss = -(d_real.mean() - d_fake.mean()) + lambda_gp * gp
                self.opt_d.zero_grad(set_to_none=True)
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.opt_d)
                self.scaler.update()
            d_losses.append(d_loss.item())

            # --- Train G ---
            with autocast(enabled=self.scaler.is_enabled()):
                y_gen_reg, y_gen_logit = self.G(xb)
                d_fake = self.D(xb, y_gen_reg)
                g_adv = -d_fake.mean()
                g_l1 = self.l1(y_gen_reg, yb)
                y_dir = (yb > 0).float()
                g_cls = self.bce(y_gen_logit, y_dir)
                g_loss = g_adv + 0.4 * g_l1 + 0.2 * g_cls
            self.opt_g.zero_grad(set_to_none=True)
            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.opt_g)
            self.scaler.update()
            g_losses.append(g_loss.item())

        return float(sum(g_losses)/len(g_losses)), float(sum(d_losses)/len(d_losses))

    def predict(self, xb):
        self.G.eval()
        with torch.no_grad():
            xb = xb.to(self.device)
            y_reg, y_logit = self.G(xb)
        return y_reg.cpu(), y_logit.cpu()
