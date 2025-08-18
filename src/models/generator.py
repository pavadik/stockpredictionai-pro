import torch
from torch import nn

class LSTMGenerator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size + 1, hidden_size, num_layers=num_layers, batch_first=True)
        self.out_reg = nn.Linear(hidden_size, 1)
        self.out_cls = nn.Linear(hidden_size, 1)

    def forward(self, x):
        noise = torch.randn(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)
        x_in = torch.cat([x, noise], dim=-1)
        out, _ = self.lstm(x_in)
        h = out[:, -1, :]
        return self.out_reg(h).squeeze(-1), self.out_cls(h).squeeze(-1)
