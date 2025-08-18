import numpy as np
import pandas as pd

def fourier_approx(series: pd.Series, k: int = 10) -> pd.Series:
    x = series.values.astype(float)
    n = len(x)
    x -= x.mean()
    freqs = np.fft.fftfreq(n)
    fft = np.fft.fft(x)
    idx = np.argsort(np.abs(fft))[-k:]
    keep = np.zeros_like(fft, dtype=complex)
    keep[idx] = fft[idx]
    recon = np.fft.ifft(keep).real + series.mean()
    return pd.Series(recon, index=series.index, name=f"{series.name}_fft{k}")
