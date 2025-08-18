import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Tuple

class GELU(nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)

class StackedAutoencoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, bottleneck: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            GELU(),
            nn.Linear(hidden, bottleneck),
            GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            GELU(),
            nn.Linear(hidden, in_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

def fit_autoencoder(df: pd.DataFrame, hidden=64, bottleneck=32, epochs=10, batch_size=128, lr=1e-3, device=None) -> Tuple[pd.DataFrame, StackedAutoencoder]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(df.values, dtype=torch.float32)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = StackedAutoencoder(df.shape[1], hidden, bottleneck).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
    # extract features
    model.eval()
    with torch.no_grad():
        z = []
        for (xb,) in DataLoader(ds, batch_size=batch_size):
            _, zi = model(xb.to(device))
            z.append(zi.cpu().numpy())
    Z = np.vstack(z)
    cols = [f"ae_{i:02d}" for i in range(Z.shape[1])]
    return pd.DataFrame(Z, index=df.index, columns=cols), model


def transform_autoencoder(df: pd.DataFrame, model: StackedAutoencoder, batch_size=128, device=None) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(df.values, dtype=torch.float32)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    Zs = []
    with torch.no_grad():
        for (xb,) in dl:
            _, zi = model(xb.to(device))
            Zs.append(zi.cpu().numpy())
    import numpy as np
    Z = np.vstack(Zs)
    cols = [f"ae_{i:02d}" for i in range(Z.shape[1])]
    return pd.DataFrame(Z, index=df.index, columns=cols)
