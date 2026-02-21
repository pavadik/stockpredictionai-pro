from dataclasses import dataclass, field
import re
from typing import Optional

# MOEX session: ~530 min (~8.8h), ~390 active trading minutes (~6.5h)
_SESSION_MINUTES = 390


def parse_timeframe(tf: str) -> dict:
    """Parse a timeframe string into its components.

    Supported formats:
        M1, M3, M5, M7, M14, M30  -- minute bars
        H1, H4                     -- hour bars
        D1                         -- daily bars
        tick                       -- raw tick data (special)

    Returns:
        {"unit": "M"|"H"|"D"|"tick", "n": int, "minutes": int, "resample_rule": str}
    """
    tf = tf.strip().upper()
    if tf == "TICK":
        return {"unit": "tick", "n": 0, "minutes": 0, "resample_rule": ""}

    m = re.fullmatch(r"([MHD])(\d+)", tf)
    if not m:
        raise ValueError(
            f"Invalid timeframe '{tf}'. "
            f"Use M<n> (e.g. M1, M5, M30), H<n> (e.g. H1, H4), D1, or 'tick'."
        )
    unit, n = m.group(1), int(m.group(2))
    if n <= 0:
        raise ValueError(f"Timeframe period must be positive, got {n}")

    if unit == "M":
        return {"unit": "M", "n": n, "minutes": n, "resample_rule": f"{n}min"}
    elif unit == "H":
        return {"unit": "H", "n": n, "minutes": n * 60, "resample_rule": f"{n}h"}
    elif unit == "D":
        return {"unit": "D", "n": n, "minutes": n * _SESSION_MINUTES,
                "resample_rule": f"{n}D"}
    raise ValueError(f"Unknown unit '{unit}'")


def timeframe_mult(tf: str) -> int:
    """Compute indicator-period multiplier for a timeframe vs D1.

    The multiplier is how many bars in one trading session.
    D1=1, H1=7, M1=390, M5=78, etc.
    For tick data returns 1 (caller should aggregate first).
    """
    info = parse_timeframe(tf)
    if info["unit"] == "tick" or info["minutes"] == 0:
        return 1
    return max(1, _SESSION_MINUTES // info["minutes"])


# Default correlated tickers for MOEX local data
MOEX_CORRELATED_DEFAULT = (
    "GMKN", "LKOH", "YNDX", "TATNP", "RTSF",
    "SBRF", "AKRN", "MVID", "LSR", "RTKMP",
)


def _wf_defaults_for(tf: str) -> dict:
    """Compute walk-forward step/min_train for any timeframe (in bars).

    Based on D1 baseline: step=120 days, min_train=500 days.
    """
    mult = timeframe_mult(tf)
    return {"step": 120 * mult, "min_train": 500 * mult}


@dataclass
class Config:
    ticker: str = "GS"
    start: str = "2010-01-01"
    end: str = "2018-12-31"
    test_years: int = 2

    # Data source: "yfinance" for US stocks, "local" for MOEX from disk
    data_source: str = "yfinance"
    # Timeframe: M<n>, H<n>, D1, or "tick".
    # Examples: M1, M3, M5, M7, M14, M30, H1, H4, D1, tick
    timeframe: str = "D1"
    # Raw data source for local: "m1" (M1 bars) or "ticks" (tick data)
    # When "ticks", data is aggregated from raw tick stream.
    # When "m1" (default), M1 bars are aggregated to the target timeframe.
    local_raw_source: str = "m1"
    # Path to local data hierarchy (YEAR/MONTH/DAY/TICKER/M1|ticks)
    data_path: str = ""
    # Correlated tickers for local MOEX data (auto-discovered if empty)
    local_correlated: tuple = ()

    seq_len: int = 12       # Phase 1 best: 12 (1h M5 context, DirAcc 48.9%)
    batch_size: int = 256
    lr_g: float = 1e-3     # Phase 1 best: 1e-3 (G > D strategy)
    lr_d: float = 5e-5     # Phase 1 best: 5e-5
    n_epochs: int = 20
    critic_steps: int = 5
    hidden_size: int = 64
    num_layers: int = 1

    # Model type: "gan" | "supervised" | "classifier" | "cross_attn" | "tft"
    model_type: str = "gan"
    # Loss function for supervised models: "huber" | "mse"
    loss_fn: str = "huber"
    # Classification threshold (0.0 = binary up/down; >0 = 3-class up/flat/down)
    cls_threshold: float = 0.0
    # Number of classification classes (2 or 3)
    n_classes: int = 2

    # Generator choice: "lstm" | "tst"
    generator: str = "tst"  # Phase 1 best: tst (DirAcc 49.0% vs LSTM 48.4%)
    # TST (Transformer) parameters
    d_model: int = 64
    nhead: int = 4
    num_layers_tst: int = 2
    dropout_tst: float = 0.1
    # LSTM dropout
    dropout_lstm: float = 0.1

    # Multi-horizon prediction (D1): predict +h bars for each h in tuple
    forecast_horizons: tuple = (1,)
    # Volatility-adjusted target (D2): normalize delta by ATR
    use_atr_target: bool = False
    atr_period: int = 14

    # Loss weights (Phase 2 best: q=1.0 improves PinballLoss & MAE)
    adv_weight: float = 1.0
    l1_weight: float = 0.4
    cls_weight: float = 0.2
    q_weight: float = 1.0

    # Quantile regression
    quantiles: tuple = (0.1, 0.5, 0.9)

    # Fourier: multiple components for long/medium/short-term trends
    fourier_components: tuple = (3, 6, 9)
    arima_order: tuple = (5, 1, 0)

    # Feature toggles (auto-adjusted per timeframe)
    use_arima: bool = True
    use_volume_features: bool = False
    use_fourier: bool = False       # Round 1 ablation: neutral effect
    use_pca: bool = False           # Round 1 ablation: neutral effect
    use_indicators: bool = False    # Round 1 ablation: neutral (SMA/EMA/MACD/BB/RSI/Momentum)
    # Delta lag features: explicit price change lags to help learn momentum
    use_delta_lags: bool = False
    delta_lag_periods: tuple = (1, 2, 4)

    ae_hidden: int = 64
    ae_bottleneck: int = 32
    ae_epochs: int = 0          # Round 1 ablation: AE adds noise (MAE improves when off)

    pca_components: int = 12

    use_amp: bool = True

    # Gradient clipping (max norm for generator gradients; 0 = disabled)
    grad_clip: float = 1.0

    # Learning rate scheduler (Cyclical)
    use_lr_scheduler: bool = True
    lr_scheduler_min_factor: float = 0.1  # min_lr = lr * factor

    # Early stopping
    early_stopping_patience: int = 5

    # Walk-forward evaluation parameters
    wf_splits: int = 5
    wf_min_train: int = 500
    wf_step: int = 120  # test window size in bars (~6 months for D1)

    # GPU optimisation
    num_workers: int = 2          # DataLoader parallel workers (0 = main thread only)
    deterministic: bool = False   # True = reproducible but slower (disables TF32 & cuDNN benchmark)

    # Sentiment (opt-in, requires transformers model download)
    use_sentiment: bool = False

    @property
    def timeframe_mult(self) -> int:
        """Bars-per-session multiplier for the current timeframe."""
        return timeframe_mult(self.timeframe)

    def apply_timeframe_defaults(self):
        """Auto-adjust parameters that depend on timeframe.

        Call after construction when using non-D1 timeframes.
        Sets walk-forward params, disables slow features for fast timeframes, etc.
        """
        wf = _wf_defaults_for(self.timeframe)
        self.wf_step = wf["step"]
        self.wf_min_train = wf["min_train"]

        info = parse_timeframe(self.timeframe)
        # Disable ARIMA for anything faster than H1 (too slow)
        if info["unit"] in ("M", "tick"):
            self.use_arima = False

        if self.data_source == "local":
            self.use_sentiment = False
            self.use_volume_features = True
