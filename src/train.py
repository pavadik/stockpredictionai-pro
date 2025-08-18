import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

from .config import Config
from .data import build_panel
from .features.fourier import fourier_approx
from .features.arima_feat import arima_in_sample
from .features.autoencoder import fit_autoencoder, transform_autoencoder
from .features.pca_eigen import fit_eigen, transform_eigen
from .utils.metrics import mae, mape, direction_accuracy
from .dataset import train_test_split_by_years, make_sequences, walk_forward_splits
from .models.gan import WGAN_GP

# Optional: WANDB
try:
    import wandb
except Exception:
    wandb = None

def base_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    fft = fourier_approx(df[cfg.ticker], cfg.fourier_k)
    ar = arima_in_sample(df[cfg.ticker], cfg.arima_order)
    feat = df.join(fft).join(ar).dropna()
    return feat

def fit_transforms(train_feat: pd.DataFrame, test_feat: pd.DataFrame, cfg: Config):
    # Autoencoder fit on train, transform both
    ae_train, ae_model = fit_autoencoder(train_feat, hidden=cfg.ae_hidden, bottleneck=cfg.ae_bottleneck, epochs=cfg.ae_epochs)
    ae_test = transform_autoencoder(test_feat, ae_model)
    # PCA fit on train, transform both
    pca_train, pca_model = fit_eigen(train_feat.join(ae_train), n_components=cfg.pca_components)
    pca_test = transform_eigen(test_feat.join(ae_test), pca_model)
    # Join all
    train_all = train_feat.join(ae_train).join(pca_train).dropna()
    test_all = test_feat.join(ae_test).join(pca_test).dropna()
    return train_all, test_all

def dl_from_xy(X, y, batch):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True)
    return ds, dl

def evaluate_gan(gan, Xte, yte, batch, out_logits=False):
    te_ds = TensorDataset(torch.tensor(Xte))
    te_dl = DataLoader(te_ds, batch_size=batch, shuffle=False)
    y_pred, y_logit = [], []
    with torch.no_grad():
        for (xb,) in te_dl:
            y_reg, y_cls = gan.predict(xb)
            y_pred.append(y_reg.numpy())
            y_logit.append(y_cls.numpy())
    import numpy as np
    yp = np.concatenate(y_pred, axis=0)
    yl = np.concatenate(y_logit, axis=0)
    if out_logits:
        return yp, yl
    return yp

def run_one_split(train_df, test_df, cfg, run_log=None):
    Xtr, ytr = make_sequences(train_df, target_col=cfg.ticker, seq_len=cfg.seq_len)
    Xte, yte = make_sequences(test_df, target_col=cfg.ticker, seq_len=cfg.seq_len)
    _, tr_dl = dl_from_xy(Xtr, ytr, cfg.batch_size)
    gan = WGAN_GP(input_size=Xtr.shape[-1], seq_len=Xtr.shape[1], hidden_size=cfg.hidden_size,
                  num_layers=cfg.num_layers, lr_g=cfg.lr_g, lr_d=cfg.lr_d, critic_steps=cfg.critic_steps)
    for epoch in range(cfg.n_epochs):
        g_loss, d_loss = gan.train_epoch(tr_dl)
        if run_log is not None and wandb is not None:
            wandb.log({"g_loss": g_loss, "d_loss": d_loss, "epoch": epoch+1})
    y_pred, y_logit = evaluate_gan(gan, Xte, yte, cfg.batch_size, out_logits=True)
    from .utils.metrics import mae, mape, direction_accuracy
    met = {
        "MAE": mae(yte, y_pred),
        "MAPE": mape(yte, y_pred),
        "DirAcc": direction_accuracy(yte, y_pred),
    }
    return met, yte, y_pred, y_logit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='GS')
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default='2018-12-31')
    parser.add_argument('--test_years', type=int, default=2)
    parser.add_argument('--walk_forward', action='store_true', help='Use walk-forward evaluation')
    args = parser.parse_args()

    cfg = Config(ticker=args.ticker, start=args.start, end=args.end, test_years=args.test_years)

    if wandb is not None and os.environ.get("WANDB_PROJECT"):
        wandb.init(project=os.environ["WANDB_PROJECT"], config=vars(cfg))

    print("[1/7] Загружаю и готовлю базовую панель...")
    panel = build_panel(cfg.ticker, cfg.start, cfg.end)

    print("[2/7] Базовые фичи (Фурье, ARIMA) ...")
    feats = base_features(panel, cfg)

    metrics_all = []
    preds_all = []

    if args.walk_forward:
        print("[3/7] Walk-forward сплиты ...")
        splits = walk_forward_splits(feats, n_splits=5, min_train=500, step=60)
        for i, (tr_idx, te_idx) in enumerate(splits, 1):
            tr_feat = feats.iloc[tr_idx]
            te_feat = feats.iloc[te_idx]
            print(f"  Split {i}: train={len(tr_feat)} test={len(te_feat)}")
            tr_all, te_all = fit_transforms(tr_feat, te_feat, cfg)
            met, yte, ypred, ylog = run_one_split(tr_all, te_all, cfg, run_log=True)
            metrics_all.append(met)
            fold_df = pd.DataFrame({'y_true': yte, 'y_pred': ypred})
            fold_df.to_csv(f'outputs/test_predictions_split{i}.csv', index=False)
            preds_all.append(fold_df)
            print(f"  Split {i} metrics: {met}")
        # aggregate
        dfm = pd.DataFrame(metrics_all)
        dfm.to_csv('outputs/metrics_walk_forward.csv', index=False)
        print("Агрегированные метрики walk-forward:
", dfm.mean().to_dict())
    else:
        print("[3/7] Трен/тест по годам ...")
        train_df, test_df = train_test_split_by_years(feats, cfg.test_years)
        print("[4/7] Fit transforms (AE/PCA) только на train...")
        tr_all, te_all = fit_transforms(train_df, test_df, cfg)
        print("[5/7] Последовательности ...")
        Xtr, ytr = make_sequences(tr_all, target_col=cfg.ticker, seq_len=cfg.seq_len)
        Xte, yte = make_sequences(te_all, target_col=cfg.ticker, seq_len=cfg.seq_len)
        _, tr_dl = dl_from_xy(Xtr, ytr, cfg.batch_size)
        print("[6/7] Тренирую WGAN-GP (multi-task)...")
        gan = WGAN_GP(input_size=Xtr.shape[-1], seq_len=Xtr.shape[1], hidden_size=cfg.hidden_size,
                      num_layers=cfg.num_layers, lr_g=cfg.lr_g, lr_d=cfg.lr_d, critic_steps=cfg.critic_steps)
        for epoch in range(cfg.n_epochs):
            g_loss, d_loss = gan.train_epoch(tr_dl)
            if wandb is not None and os.environ.get("WANDB_PROJECT"):
                wandb.log({"g_loss": g_loss, "d_loss": d_loss, "epoch": epoch+1})
            print(f"epoch {epoch+1}/{cfg.n_epochs}: G={g_loss:.4f} D={d_loss:.4f}")
        print("[7/7] Оценка на тесте ...")
        from .utils.metrics import mae, mape, direction_accuracy
        te_ds = TensorDataset(torch.tensor(Xte))
        te_dl = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False)
        y_pred, y_logit = [], []
        with torch.no_grad():
            for (xb,) in te_dl:
                y_reg, y_cls = gan.predict(xb)
                y_pred.append(y_reg.numpy()); y_logit.append(y_cls.numpy())
        import numpy as np
        y_pred = np.concatenate(y_pred, axis=0)
        y_logit = np.concatenate(y_logit, axis=0)

        m_mae = mae(yte, y_pred)
        m_mape = mape(yte, y_pred)
        m_dir = direction_accuracy(yte, y_pred)
        print(f"MAE={m_mae:.4f}  MAPE={m_mape:.4f}  DirAcc={m_dir:.3f}")
        os.makedirs('outputs', exist_ok=True)
        pd.DataFrame({'y_true': yte, 'y_pred': y_pred, 'y_logit': y_logit}).to_csv('outputs/test_predictions.csv', index=False)
        print("Сохранено: outputs/test_predictions.csv")

if __name__ == '__main__':
    main()
