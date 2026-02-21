import numpy as np
import pandas as pd
from typing import Sequence, Tuple


def fourier_approx(series: pd.Series, k: int = 10) -> pd.Series:
    """Reconstruct *series* using only the top-*k* Fourier components.

    WARNING: uses FFT on the entire series (look-ahead bias if applied before
    train/test split). Prefer ``fourier_approx_safe`` for production pipelines.
    """
    x = series.values.astype(float)
    mean = x.mean()
    x = x - mean
    fft = np.fft.fft(x)
    idx = np.argsort(np.abs(fft))[-k:]
    keep = np.zeros_like(fft, dtype=complex)
    keep[idx] = fft[idx]
    recon = np.fft.ifft(keep).real + mean
    return pd.Series(recon, index=series.index, name=f"{series.name}_fft{k}")


# ---------------------------------------------------------------------------
# Leakage-safe Fourier features: fit on train, extrapolate on test
# ---------------------------------------------------------------------------

def _fit_fourier_components(train_values: np.ndarray, k: int):
    """Extract top-k Fourier components (amplitudes + phases) from *train_values*.

    Returns (mean, freqs_idx, amplitudes, phases) needed for reconstruction.
    """
    mean = float(train_values.mean())
    x = train_values - mean
    n = len(x)
    fft = np.fft.fft(x)
    magnitudes = np.abs(fft)
    top_idx = np.argsort(magnitudes)[-k:]
    return mean, n, top_idx, fft[top_idx]


def _reconstruct_from_components(mean, n_train, top_idx, fft_components, n_total):
    """Reconstruct (and extrapolate) a signal of length *n_total* from stored
    Fourier components that were fitted on *n_train* points.

    For indices beyond *n_train* the sinusoidal components are naturally
    extrapolated using their frequencies, so no future data is used.
    """
    t = np.arange(n_total)
    result = np.full(n_total, mean)
    for freq_idx, coeff in zip(top_idx, fft_components):
        amplitude = np.abs(coeff) / n_train
        phase = np.angle(coeff)
        freq = 2 * np.pi * freq_idx / n_train
        result = result + amplitude * np.cos(freq * t + phase)
    return result


def fit_fourier(series: pd.Series, k: int = 10) -> dict:
    """Fit top-k Fourier components on *series* (train only).

    Returns a state dict that can be passed to ``transform_fourier``.
    """
    vals = series.values.astype(float)
    mean, n, top_idx, fft_comps = _fit_fourier_components(vals, k)
    return {
        "mean": mean, "n_train": n, "top_idx": top_idx,
        "fft_comps": fft_comps, "k": k, "name": series.name,
    }


def transform_fourier(index: pd.Index, state: dict) -> pd.Series:
    """Reconstruct / extrapolate Fourier approximation for arbitrary *index* length.

    Uses only information captured during ``fit_fourier`` on training data.
    """
    n_total = len(index)
    recon = _reconstruct_from_components(
        state["mean"], state["n_train"],
        state["top_idx"], state["fft_comps"], n_total,
    )
    k = state["k"]
    name = state["name"]
    return pd.Series(recon, index=index, name=f"{name}_fft{k}")


def fit_fourier_multi(series: pd.Series,
                      components: Sequence[int] = (3, 6, 9)) -> list:
    """Fit multiple Fourier approximations. Returns list of state dicts."""
    return [fit_fourier(series, k) for k in components]


def transform_fourier_multi(index: pd.Index,
                            states: list) -> pd.DataFrame:
    """Transform / extrapolate multiple Fourier approximations.

    *states* is the list returned by ``fit_fourier_multi``.
    """
    cols = {}
    for st in states:
        s = transform_fourier(index, st)
        cols[s.name] = s.values
    return pd.DataFrame(cols, index=index)


# ---------------------------------------------------------------------------
# Legacy convenience wrapper (uses full series -- kept for quick experiments)
# ---------------------------------------------------------------------------

def fourier_multi(series: pd.Series, components: Sequence[int] = (3, 6, 9)) -> pd.DataFrame:
    """Return a DataFrame with one Fourier approximation column per *k* in *components*.

    WARNING: this applies FFT on the entire series (look-ahead bias).
    For train/test safe version use ``fit_fourier_multi`` + ``transform_fourier_multi``.
    """
    cols = {}
    for k in components:
        cols[f"{series.name}_fft{k}"] = fourier_approx(series, k).values
    return pd.DataFrame(cols, index=series.index)
