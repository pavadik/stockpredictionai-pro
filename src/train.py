import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from .config import Config
from .data import build_panel, build_panel_auto
from .features.fourier import fit_fourier_multi, transform_fourier_multi
from .features.arima_feat import fit_arima, transform_arima
from .features.autoencoder import fit_autoencoder, transform_autoencoder
from .features.pca_eigen import fit_eigen, transform_eigen
from .utils.metrics import mae, mape, direction_accuracy, pinball_loss, multi_horizon_metrics
from .dataset import (train_test_split_by_years, make_sequences, walk_forward_splits,
                      make_sequences_multi, make_sequences_atr)
from .models.gan import WGAN_GP
from .models.supervised import SupervisedModel
from .models.classifier import DirectionClassifier
from .models.cross_attention import CrossAttentionModel
from .models.tft import SimplifiedTFT
from .utils.visualization import (
    plot_predictions, plot_training_curves,
    plot_technical_indicators, plot_fourier_components,
)

# Optional: WANDB
try:
    import wandb
except Exception:
    wandb = None

SEED = 42


def set_global_seed(seed: int = SEED, deterministic: bool = False):
    """Set random seeds for reproducibility across numpy, torch, and python.

    Args:
        seed: random seed value.
        deterministic: if True, forces fully reproducible (but slower) execution
            by disabling cuDNN benchmark, TF32, and enabling deterministic algorithms.
            If False, enables cuDNN benchmark and TF32 for maximum GPU throughput.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # cuDNN benchmark: auto-tunes convolution algorithms for fixed input sizes
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # TF32: ~2x faster matmul on Ampere+ GPUs (RTX 30xx / A100) with negligible accuracy loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _log_device_info():
    """Print GPU / CPU device information at startup."""
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"  Device : {dev.name}  (compute capability {dev.major}.{dev.minor})")
        print(f"  VRAM   : {total / 1024**3:.1f} GB total, {free / 1024**3:.1f} GB free")
        print(f"  PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}"
              f"  |  cuDNN: {torch.backends.cudnn.version()}")
        print(f"  TF32   : matmul={torch.backends.cuda.matmul.allow_tf32}"
              f"  cuDNN={torch.backends.cudnn.allow_tf32}")
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  AMP    : enabled")
    else:
        print(f"  Device : CPU  |  PyTorch: {torch.__version__}")
        print("  WARNING: CUDA not available -- training will be slow")


# ---------------------------------------------------------------------------
# Feature construction (leakage-safe: fit on train, transform on test)
# ---------------------------------------------------------------------------

def build_features_safe(train_panel: pd.DataFrame, test_panel: pd.DataFrame,
                        cfg: Config):
    """Build Fourier + ARIMA features without data leakage.

    Fourier: fit FFT components on train, extrapolate to test.
    ARIMA: fit on train, rolling one-step-ahead forecast on test.
    Respects cfg.use_arima (auto-disabled for M1 timeframe).
    """
    ticker = cfg.ticker

    # --- Fourier (leakage-safe, optional) ---
    if cfg.use_fourier:
        fft_states = fit_fourier_multi(train_panel[ticker], cfg.fourier_components)
        fft_train = transform_fourier_multi(train_panel.index, fft_states)
        fft_test = transform_fourier_multi(test_panel.index, fft_states)
        train_feat = train_panel.join(fft_train)
        test_feat = test_panel.join(fft_test)
    else:
        train_feat = train_panel.copy()
        test_feat = test_panel.copy()

    # --- ARIMA (leakage-safe, skipped for M1 due to performance) ---
    if cfg.use_arima:
        arima_state = fit_arima(train_panel[ticker], cfg.arima_order)
        ar_train, ar_test = transform_arima(
            train_panel[ticker], test_panel[ticker], arima_state
        )
        train_feat = train_feat.join(ar_train)
        test_feat = test_feat.join(ar_test)

    # Delta lag features: explicit price-change lags for momentum learning
    if getattr(cfg, 'use_delta_lags', False):
        lag_periods = getattr(cfg, 'delta_lag_periods', (1, 2, 4))
        for lag in lag_periods:
            col = f"{ticker}_dlag{lag}"
            train_feat[col] = train_feat[ticker].diff(lag)
            test_feat[col] = test_feat[ticker].diff(lag)

    train_feat = train_feat.dropna()
    test_feat = test_feat.dropna()

    # Optional sentiment feature
    if cfg.use_sentiment:
        try:
            from .features.sentiment import get_ticker_sentiment
            sent = get_ticker_sentiment(ticker, cfg.start, cfg.end)
            if sent is not None:
                train_feat = train_feat.join(sent, how="left").fillna(0.0)
                test_feat = test_feat.join(sent, how="left").fillna(0.0)
        except Exception:
            pass  # silently skip if unavailable

    return train_feat, test_feat


def fit_transforms(train_feat: pd.DataFrame, test_feat: pd.DataFrame, cfg: Config):
    """Autoencoder + PCA + StandardScaler: fit on train, transform both.

    AE and PCA are gated by cfg.ae_epochs > 0 and cfg.use_pca respectively.
    """
    train_all = train_feat.copy()
    test_all = test_feat.copy()

    # Autoencoder (optional -- disabled when ae_epochs=0)
    if cfg.ae_epochs > 0:
        ae_train, ae_model = fit_autoencoder(train_feat, hidden=cfg.ae_hidden,
                                             bottleneck=cfg.ae_bottleneck, epochs=cfg.ae_epochs)
        ae_test = transform_autoencoder(test_feat, ae_model)
        train_all = train_all.join(ae_train)
        test_all = test_all.join(ae_test)

    # PCA (optional)
    if cfg.use_pca:
        pca_train, pca_model = fit_eigen(train_all, n_components=cfg.pca_components)
        pca_test = transform_eigen(test_all, pca_model)
        train_all = train_all.join(pca_train)
        test_all = test_all.join(pca_test)

    train_all = train_all.dropna()
    test_all = test_all.dropna()

    # StandardScaler: fit only on train, transform both (prevents data leakage)
    scaler = StandardScaler()
    cols = train_all.columns
    train_all = pd.DataFrame(scaler.fit_transform(train_all), index=train_all.index, columns=cols)
    test_all = pd.DataFrame(scaler.transform(test_all), index=test_all.index, columns=cols)
    return train_all, test_all


def dl_from_xy(X, y, batch, num_workers=2):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    pin = torch.cuda.is_available()
    dl = DataLoader(ds, batch_size=batch, shuffle=True,
                    num_workers=num_workers, pin_memory=pin,
                    persistent_workers=num_workers > 0)
    return ds, dl


def _build_gan(cfg, input_size, seq_len):
    """Build WGAN_GP with all config parameters."""
    return WGAN_GP(
        input_size=input_size, seq_len=seq_len,
        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
        lr_g=cfg.lr_g, lr_d=cfg.lr_d, critic_steps=cfg.critic_steps,
        generator_type=cfg.generator,
        d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers_tst=cfg.num_layers_tst, dropout_tst=cfg.dropout_tst,
        dropout_lstm=cfg.dropout_lstm,
        num_quantiles=len(cfg.quantiles), quantiles=cfg.quantiles,
        adv_weight=cfg.adv_weight, l1_weight=cfg.l1_weight,
        cls_weight=cfg.cls_weight, q_weight=cfg.q_weight,
        use_lr_scheduler=cfg.use_lr_scheduler,
        lr_scheduler_min_factor=cfg.lr_scheduler_min_factor,
        n_epochs=cfg.n_epochs,
        grad_clip=cfg.grad_clip,
    )


def _build_model(cfg, input_size, seq_len, col_names=None):
    """Model factory: build any model type based on cfg.model_type.

    Args:
        col_names: list of column names in the feature DataFrame, used
                   by cross_attn to split target/correlate/volume streams.
    """
    mt = cfg.model_type
    if mt == "gan":
        return _build_gan(cfg, input_size, seq_len)
    elif mt == "supervised":
        return SupervisedModel(input_size, seq_len, cfg)
    elif mt == "classifier":
        return DirectionClassifier(input_size, seq_len, cfg)
    elif mt == "cross_attn":
        col_split = None
        if col_names is not None:
            ticker = cfg.ticker
            target_idx = [i for i, c in enumerate(col_names) if c == ticker]
            corr_idx = [i for i, c in enumerate(col_names)
                        if c != ticker and not c.startswith(f"{ticker}_")]
            other_idx = [i for i, c in enumerate(col_names)
                         if c.startswith(f"{ticker}_")]
            col_split = {"target_idx": target_idx or [0],
                         "corr_idx": corr_idx or list(range(1, input_size)),
                         "other_idx": other_idx}
        return CrossAttentionModel(input_size, seq_len, cfg, col_split=col_split)
    elif mt == "tft":
        return SimplifiedTFT(input_size, seq_len, cfg)
    else:
        raise ValueError(f"Unknown model_type: {mt}")


def evaluate_model(model, Xte, yte, batch, quantiles=(0.1, 0.5, 0.9)):
    """Run inference on any PredictionModel, return predictions, logits, and quantiles."""
    te_ds = TensorDataset(torch.tensor(Xte))
    te_dl = DataLoader(te_ds, batch_size=batch, shuffle=False)
    y_pred_list, y_logit_list, y_q_list = [], [], []
    with torch.no_grad():
        for (xb,) in te_dl:
            y_reg, y_cls, y_q = model.predict(xb)
            y_pred_list.append(y_reg.numpy())
            y_logit_list.append(y_cls.numpy())
            y_q_list.append(y_q.numpy())
    yp = np.concatenate(y_pred_list, axis=0)
    yl = np.concatenate(y_logit_list, axis=0)
    yq = np.concatenate(y_q_list, axis=0)
    return yp, yl, yq


# Backward compatibility alias
evaluate_gan = evaluate_model


def compute_metrics(yte, y_pred, y_q, quantiles=(0.1, 0.5, 0.9)):
    """Compute all evaluation metrics."""
    return {
        "MAE": mae(yte, y_pred),
        "MAPE": mape(yte, y_pred),
        "DirAcc": direction_accuracy(yte, y_pred),
        "PinballLoss": pinball_loss(yte, y_q, quantiles),
    }


# ---------------------------------------------------------------------------
# Unified training loop with early stopping (no duplication)
# ---------------------------------------------------------------------------

def train_with_early_stop(model, tr_dl, Xte, yte, cfg, verbose=True):
    """Train any PredictionModel with optional early stopping based on validation MAE.

    Works with GAN (WGAN_GP) and all Block C models (SupervisedModel, etc.).
    Returns (g_loss, d_loss, stopped_epoch).
    """
    patience = cfg.early_stopping_patience
    best_mae = float('inf')
    epochs_no_improve = 0
    best_state = None
    stopped_epoch = cfg.n_epochs

    is_gan = isinstance(model, WGAN_GP)

    for epoch in range(cfg.n_epochs):
        if is_gan:
            g_loss, d_loss = model.train_epoch(tr_dl)
        else:
            result = model.train_epoch(tr_dl)
            g_loss = result.get("g_loss", result.get("loss", 0.0))
            d_loss = result.get("d_loss", 0.0)

        model.step_schedulers()

        if wandb is not None and os.environ.get("WANDB_PROJECT"):
            wandb.log({"g_loss": g_loss, "d_loss": d_loss, "epoch": epoch + 1})

        marker = ""
        if patience > 0 and Xte is not None:
            yp, _, _ = evaluate_model(model, Xte, yte, cfg.batch_size, cfg.quantiles)
            if yp.ndim > 1:
                val_mae = mae(yte[:, 0], yp[:, 0])
            else:
                val_mae = mae(yte, yp)
            if val_mae < best_mae:
                best_mae = val_mae
                epochs_no_improve = 0
                if is_gan:
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.G.state_dict().items()}
                else:
                    best_state = model.get_state_dict()
                marker = " *"
            else:
                epochs_no_improve += 1

            if verbose:
                print(f"  epoch {epoch + 1}/{cfg.n_epochs}: G={g_loss:.4f} D={d_loss:.4f} "
                      f"val_MAE={val_mae:.4f}{marker}")

            if epochs_no_improve >= patience:
                stopped_epoch = epoch + 1
                if verbose:
                    print(f"  Early stopping at epoch {stopped_epoch} (patience={patience})")
                if best_state is not None:
                    if is_gan:
                        model.G.load_state_dict(best_state)
                    else:
                        model.load_state_dict(best_state)
                break
        else:
            if verbose:
                print(f"  epoch {epoch + 1}/{cfg.n_epochs}: G={g_loss:.4f} D={d_loss:.4f}")

    return g_loss, d_loss, stopped_epoch


# Backward compatibility alias
train_gan_with_early_stop = train_with_early_stop


def run_one_split(train_df, test_df, cfg, verbose=False,
                  raw_train_df=None, raw_test_df=None):
    """Build sequences, train model, evaluate -- for a single train/test split.

    Uses cfg.model_type to select the appropriate model (GAN, supervised, etc.).
    Supports multi-horizon (cfg.forecast_horizons) and ATR target (cfg.use_atr_target).

    Args:
        raw_train_df, raw_test_df: Optional unscaled DataFrames. When use_atr_target
            is True, ATR-normalized targets are computed from these raw values
            (to avoid dividing by near-zero values after StandardScaler).
            If not provided, falls back to scaled train_df/test_df.
    """
    col_names = list(train_df.columns)
    horizons = getattr(cfg, 'forecast_horizons', (1,))
    use_atr = getattr(cfg, 'use_atr_target', False)
    is_multi_out = len(horizons) > 1
    needs_custom_horizon = horizons != (1,)

    atr_col = f"{cfg.ticker}_atr"
    atr_test_vals = None

    if needs_custom_horizon:
        Xtr, _ = make_sequences_multi(train_df, cfg.ticker, cfg.seq_len, horizons)
        Xte, _ = make_sequences_multi(test_df, cfg.ticker, cfg.seq_len, horizons)
        raw_tr = raw_train_df if raw_train_df is not None else train_df
        raw_te = raw_test_df if raw_test_df is not None else test_df
        _, ytr = make_sequences_multi(raw_tr, cfg.ticker, cfg.seq_len, horizons)
        _, yte = make_sequences_multi(raw_te, cfg.ticker, cfg.seq_len, horizons)
        if not is_multi_out:
            ytr = ytr.squeeze(-1)
            yte = yte.squeeze(-1)
    elif use_atr and atr_col in train_df.columns:
        Xtr, _, _ = make_sequences_atr(train_df, cfg.ticker, atr_col, cfg.seq_len)
        Xte, _, _ = make_sequences_atr(test_df, cfg.ticker, atr_col, cfg.seq_len)
        raw_tr = raw_train_df if raw_train_df is not None else train_df
        raw_te = raw_test_df if raw_test_df is not None else test_df
        _, ytr, _ = make_sequences_atr(raw_tr, cfg.ticker, atr_col, cfg.seq_len)
        _, yte, atr_test_vals = make_sequences_atr(raw_te, cfg.ticker, atr_col, cfg.seq_len)
    else:
        Xtr, ytr = make_sequences(train_df, target_col=cfg.ticker, seq_len=cfg.seq_len)
        Xte, yte = make_sequences(test_df, target_col=cfg.ticker, seq_len=cfg.seq_len)

    # Auto-compute 3-class threshold from training deltas when needed
    if getattr(cfg, 'n_classes', 2) >= 3 and cfg.cls_threshold == 0.0:
        abs_y = np.abs(ytr) if ytr.ndim == 1 else np.abs(ytr[:, 0])
        nonzero = abs_y[abs_y > 0]
        if len(nonzero) > 0:
            cfg.cls_threshold = float(np.median(nonzero))
        if verbose:
            print(f"  Auto cls_threshold = {cfg.cls_threshold:.6f}")

    _, tr_dl = dl_from_xy(Xtr, ytr, cfg.batch_size, num_workers=cfg.num_workers)

    model = _build_model(cfg, input_size=Xtr.shape[-1], seq_len=Xtr.shape[1],
                         col_names=col_names)

    train_with_early_stop(model, tr_dl, Xte, yte, cfg, verbose=verbose)
    y_pred, y_logit, y_q = evaluate_model(model, Xte, yte, cfg.batch_size, cfg.quantiles)

    if is_multi_out:
        met = multi_horizon_metrics(yte, y_pred, horizons)
        met["MAE"] = met["MAE_avg"]
        met["DirAcc"] = met["DirAcc_avg"]
        met["MAPE"] = float('nan')
        met["PinballLoss"] = float('nan')
    elif use_atr and atr_test_vals is not None:
        y_pred_raw = y_pred * atr_test_vals
        yte_raw = yte * atr_test_vals
        met = compute_metrics(yte_raw, y_pred_raw, y_q, cfg.quantiles)
        met["DirAcc_norm"] = direction_accuracy(yte, y_pred)
    else:
        met = compute_metrics(yte, y_pred, y_q, cfg.quantiles)

    return met, yte, y_pred, y_logit, y_q, model


def _quantile_col_names(quantiles):
    """Generate column names like q10, q50, q90."""
    return [f"q{int(q * 100)}" for q in quantiles]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='GS')
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default='2018-12-31')
    parser.add_argument('--test_years', type=int, default=2)
    parser.add_argument('--walk_forward', action='store_true', help='Use walk-forward evaluation')
    parser.add_argument('--model_type', default='gan',
                        choices=['gan', 'supervised', 'classifier', 'cross_attn', 'tft'],
                        help='Model type')
    parser.add_argument('--loss_fn', default='huber',
                        choices=['huber', 'mse', 'focal', 'ce'],
                        help='Loss function for supervised/classifier')
    parser.add_argument('--generator', default='lstm', choices=['lstm', 'tst'],
                        help='Generator type: lstm or tst (Transformer)')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed for reproducibility')
    parser.add_argument('--data_source', default='yfinance', choices=['yfinance', 'local'],
                        help='Data source: yfinance (US stocks) or local (MOEX from disk)')
    parser.add_argument('--timeframe', default='D1',
                        help='Timeframe: M1, M3, M5, M7, M14, M30, H1, H4, D1, or tick')
    parser.add_argument('--data_path', default='', help='Path to local data hierarchy')
    parser.add_argument('--raw_source', default='m1', choices=['m1', 'ticks'],
                        help='Local raw data source: m1 (M1 bars) or ticks (tick data)')
    parser.add_argument('--seq_len', type=int, default=None, help='Override seq_len')
    parser.add_argument('--lr_g', type=float, default=None, help='Override lr_g')
    parser.add_argument('--lr_d', type=float, default=None, help='Override lr_d')
    parser.add_argument('--adv_weight', type=float, default=None, help='Override adv_weight')
    parser.add_argument('--l1_weight', type=float, default=None, help='Override l1_weight')
    parser.add_argument('--cls_weight', type=float, default=None, help='Override cls_weight')
    parser.add_argument('--q_weight', type=float, default=None, help='Override q_weight')
    args = parser.parse_args()

    # --- Reproducibility & GPU setup ---
    cfg = Config(ticker=args.ticker, start=args.start, end=args.end,
                 test_years=args.test_years, generator=args.generator,
                 data_source=args.data_source, timeframe=args.timeframe,
                 data_path=args.data_path, local_raw_source=args.raw_source,
                 model_type=args.model_type, loss_fn=args.loss_fn)
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.lr_g is not None:
        cfg.lr_g = args.lr_g
    if args.lr_d is not None:
        cfg.lr_d = args.lr_d
    if args.adv_weight is not None:
        cfg.adv_weight = args.adv_weight
    if args.l1_weight is not None:
        cfg.l1_weight = args.l1_weight
    if args.cls_weight is not None:
        cfg.cls_weight = args.cls_weight
    if args.q_weight is not None:
        cfg.q_weight = args.q_weight
    if cfg.data_source == "local" or cfg.timeframe != "D1":
        cfg.apply_timeframe_defaults()
    set_global_seed(args.seed, deterministic=cfg.deterministic)
    _log_device_info()

    if wandb is not None and os.environ.get("WANDB_PROJECT"):
        wandb.init(project=os.environ["WANDB_PROJECT"], config=vars(cfg))

    print(f"[1/7] Loading data panel (source={cfg.data_source}, "
          f"tf={cfg.timeframe}, generator={cfg.generator}, seed={args.seed})...")
    panel = build_panel_auto(cfg)

    # Optional statistical checks
    try:
        from .utils.stat_checks import run_all_checks
        run_all_checks(panel, cfg.ticker)
    except Exception as e:
        print(f"  (stat checks skipped: {e})")

    metrics_all = []
    preds_all = []
    os.makedirs('outputs', exist_ok=True)
    q_names = _quantile_col_names(cfg.quantiles)

    if args.walk_forward:
        print("[2/7] Walk-forward splits...")
        splits = walk_forward_splits(panel, n_splits=cfg.wf_splits,
                                     min_train=cfg.wf_min_train, step=cfg.wf_step)
        for i, (tr_idx, te_idx) in enumerate(splits, 1):
            tr_panel = panel.iloc[tr_idx]
            te_panel = panel.iloc[te_idx]
            print(f"  Split {i}: train={len(tr_panel)} test={len(te_panel)}")

            print(f"  [3/7] Building features (leakage-safe)...")
            tr_feat, te_feat = build_features_safe(tr_panel, te_panel, cfg)
            print(f"  [4/7] Fit transforms (AE/PCA) on train only...")
            tr_all, te_all = fit_transforms(tr_feat, te_feat, cfg)
            print(f"  [5/7] Training GAN...")
            met, yte, ypred, ylog, yq, _ = run_one_split(tr_all, te_all, cfg, verbose=False)
            metrics_all.append(met)
            fold_data = {'y_true': yte, 'y_pred': ypred}
            for j, qn in enumerate(q_names):
                fold_data[qn] = yq[:, j]
            fold_df = pd.DataFrame(fold_data)
            fold_df.to_csv(f'outputs/test_predictions_split{i}.csv', index=False)
            preds_all.append(fold_df)
            print(f"  Split {i} metrics: {met}")
        dfm = pd.DataFrame(metrics_all)
        dfm.to_csv('outputs/metrics_walk_forward.csv', index=False)
        print("Aggregated walk-forward metrics:\n", dfm.mean().to_dict())
    else:
        print("[2/7] Train/test split by years...")
        train_panel, test_panel = train_test_split_by_years(panel, cfg.test_years)

        print("[3/7] Building features (leakage-safe Fourier + ARIMA)...")
        tr_feat, te_feat = build_features_safe(train_panel, test_panel, cfg)

        print("[4/7] Fit transforms (AE/PCA) on train only...")
        tr_all, te_all = fit_transforms(tr_feat, te_feat, cfg)

        print("[5/7] Building sequences...")
        Xtr, ytr = make_sequences(tr_all, target_col=cfg.ticker, seq_len=cfg.seq_len)
        Xte, yte = make_sequences(te_all, target_col=cfg.ticker, seq_len=cfg.seq_len)
        _, tr_dl = dl_from_xy(Xtr, ytr, cfg.batch_size, num_workers=cfg.num_workers)

        col_names = list(tr_all.columns)
        print(f"[6/7] Training {cfg.model_type.upper()} ({cfg.generator} backbone)...")
        model = _build_model(cfg, input_size=Xtr.shape[-1], seq_len=Xtr.shape[1],
                             col_names=col_names)
        train_with_early_stop(model, tr_dl, Xte, yte, cfg, verbose=True)

        print("[7/7] Evaluating on test set...")
        y_pred, y_logit, y_q = evaluate_model(model, Xte, yte, cfg.batch_size, cfg.quantiles)

        met = compute_metrics(yte, y_pred, y_q, cfg.quantiles)
        print(f"  MAE={met['MAE']:.4f}  MAPE={met['MAPE']:.4f}  "
              f"DirAcc={met['DirAcc']:.3f}  PinballLoss={met['PinballLoss']:.4f}")

        out_data = {'y_true': yte, 'y_pred': y_pred, 'y_logit': y_logit}
        for j, qn in enumerate(q_names):
            out_data[qn] = y_q[:, j]
        pd.DataFrame(out_data).to_csv('outputs/test_predictions.csv', index=False)
        print("Saved: outputs/test_predictions.csv")

        # --- Visualization ---
        try:
            plot_predictions(yte, y_pred, y_q, cfg.quantiles)
            print("Saved: outputs/pred_vs_real.png")
            plot_technical_indicators(panel, cfg.ticker)
            print("Saved: outputs/technical_indicators.png")
            plot_fourier_components(panel[cfg.ticker], cfg.fourier_components)
            print("Saved: outputs/fourier_components.png")
        except Exception as e:
            print(f"  (visualization skipped: {e})")


if __name__ == '__main__':
    main()
