import torch
from torch import nn


class CNNDiscriminator(nn.Module):
    """1-D CNN discriminator for WGAN-GP.

    Architecture: 3 Conv1D layers (32 -> 64 -> 128) with BatchNorm and LeakyReLU,
    followed by two FC layers (220 -> 220 -> 1).
    Input: x [B, T, F] concatenated with y [B, T, 1] along feature dim -> [B, T, F+1].
    Conv1D expects [B, C, T], so we transpose the time/feature axes.
    """

    def __init__(self, input_size: int, seq_len: int):
        super().__init__()
        in_channels = input_size + 1  # +1 for the conditional y feature

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * seq_len, 220),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(220, 220),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(220, 1),  # WGAN score (no sigmoid)
        )

    def forward(self, x, y):
        # x: [B, T, F], y: [B] -- scalar prediction / target
        B, T, F = x.shape
        y_feat = y.view(B, 1, 1).expand(B, T, 1)  # [B, T, 1]
        xy = torch.cat([x, y_feat], dim=-1)         # [B, T, F+1]
        z = xy.permute(0, 2, 1)                     # [B, F+1, T] for Conv1d
        z = self.conv(z)                             # [B, 128, T]
        score = self.fc(z).squeeze(-1)
        return score
