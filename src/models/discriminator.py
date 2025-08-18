import torch
from torch import nn

class CNNDiscriminator(nn.Module):
    def __init__(self, input_size: int, seq_len: int):
        super().__init__()
        # свернём по времени: вход [B, T, F] -> [B, 1, T, F]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 1), padding=(1,0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1,0)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * seq_len * input_size, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)  # WGAN score (без сигмоиды)
        )
    def forward(self, x, y):
        # x: [B, T, F], y: [B] (добавим как доп. канал длиной 1 на последнем шаге)
        B, T, F = x.shape
        y_feat = y.view(B, 1, 1).repeat(1, T, 1)  # [B,T,1]
        xy = torch.cat([x, y_feat], dim=-1)[:, :, :F]  # простая стыковка
        z = xy.unsqueeze(1)  # [B,1,T,F]
        z = self.conv(z)
        score = self.fc(z).squeeze(-1)
        return score
