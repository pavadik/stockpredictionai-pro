import math
import torch
from torch import nn


class LSTMGenerator(nn.Module):
    """LSTM-based generator with regression, classification, and quantile heads."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1,
                 num_quantiles: int = 3, dropout: float = 0.1):
        super().__init__()
        # nn.LSTM dropout only applies when num_layers > 1; we add explicit Dropout too
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size + 1, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.drop = nn.Dropout(dropout)
        self.out_reg = nn.Linear(hidden_size, 1)
        self.out_cls = nn.Linear(hidden_size, 1)
        self.out_quantiles = nn.Linear(hidden_size, num_quantiles)

    def forward(self, x):
        noise = torch.randn(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)
        x_in = torch.cat([x, noise], dim=-1)
        out, _ = self.lstm(x_in)
        h = self.drop(out[:, -1, :])
        return (
            self.out_reg(h).squeeze(-1),
            self.out_cls(h).squeeze(-1),
            self.out_quantiles(h),
        )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerGenerator(nn.Module):
    """Transformer (TST) generator with regression, classification, and quantile heads."""

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, num_quantiles: int = 3):
        super().__init__()
        # Project input (+1 for noise) to d_model
        self.input_proj = nn.Linear(input_size + 1, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_reg = nn.Linear(d_model, 1)
        self.out_cls = nn.Linear(d_model, 1)
        self.out_quantiles = nn.Linear(d_model, num_quantiles)

    def forward(self, x):
        noise = torch.randn(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)
        x_in = torch.cat([x, noise], dim=-1)       # [B, T, F+1]
        x_in = self.input_proj(x_in)               # [B, T, d_model]
        x_in = self.pos_enc(x_in)
        h_seq = self.transformer_encoder(x_in)      # [B, T, d_model]
        h = h_seq[:, -1, :]                         # last time step
        return (
            self.out_reg(h).squeeze(-1),
            self.out_cls(h).squeeze(-1),
            self.out_quantiles(h),
        )
