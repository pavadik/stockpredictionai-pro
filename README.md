# StockPredictionAI Pro (2025 Edition)

**A full, modern reimplementation of the stock prediction project (Goldman Sachs, GS) based on Boris Banushev’s original notebook/repository — now upgraded with state-of-the-art PyTorch, quantile regression, Transformer generators, and automatic hyperparameter tuning.**

---

## ✨ Features

- **Data loading** via [Yahoo Finance](https://pypi.org/project/yfinance/): GS and correlated assets (SPY, XLF, JPM, BAC, MS, VIX, TNX, Dollar Index, Gold, EURUSD).
- **Technical indicators**: SMA, EMA, MACD, Bollinger Bands, RSI, Momentum.
- **Fourier & ARIMA features**: trend approximation with harmonics and ARIMA(5,1,0).
- **Autoencoder + PCA (Eigen-portfolio)**: extract nonlinear and linear latent features.
- **Feature Importance**: XGBoost regressor for feature relevance.
- **GAN (WGAN-GP)**:
  - Generator: LSTM or Transformer (TST).
  - Discriminator: temporal CNN.
  - Multi-task output: ∆P regression, direction classification, quantile regression (forecast intervals).
- **Metrics**: MAE, MAPE, directional accuracy, Pinball Loss.
- **Walk-forward evaluation**: time-consistent multi-split validation.
- **Optuna tuning**: automatic hyperparameter search.
- **Engineering improvements**:
  - Support for `torch.compile` and AMP (mixed precision).
  - Efficient DataLoaders (`num_workers`, `prefetch`, `pin_memory`).
  - WandB logging (optional).
  - Modular project structure (`src/features`, `src/models`, `scripts`).

---

## 📦 Installation

```bash
git clone https://github.com/pavadik/stockpredictionai-pro.git
cd stockpredictionai-pro

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🚀 Quick Start

Run a standard experiment (LSTM generator, train/test split by years):

```bash
python -m src.train --ticker GS --start 2010-01-01 --end 2018-12-31 --test_years 2
```

This launches the full pipeline: data → feature engineering → GAN training → evaluation → results in `outputs/test_predictions.csv`.

---

## 🔄 Walk-forward Evaluation

For time-consistent evaluation (default 5 splits):

```bash
python -m src.train --ticker GS --start 2010-01-01 --end 2018-12-31 --walk_forward
```

Metrics are saved to `outputs/metrics_walk_forward.csv`, split predictions in `outputs/test_predictions_split{i}.csv`.

---

## 🧠 Generator Choice

By default, **LSTM** is used. To switch to **Transformer (TST)** generator:

```bash
python -m src.train --ticker GS --start 2010-01-01 --end 2018-12-31 --generator tst
```

TST parameters (see `Config`):
- `d_model` — hidden size.
- `nhead` — number of attention heads.
- `num_layers_tst` — number of encoder layers.
- `dropout_tst` — dropout rate.

---

## 📊 Quantile Forecasts

The generator outputs quantile values (by default 10/50/90 percentiles).  
In `outputs/test_predictions.csv` you will find columns `q10`, `q50`, `q90`, which represent forecast intervals (uncertainty estimation).

---

## ⚙️ Optuna Hyperparameter Tuning

Run automatic hyperparameter search (example with 20 trials):

```bash
python scripts/optuna_tune.py --ticker GS --start 2010-01-01 --end 2018-12-31 --trials 20
```

Best results are stored in `outputs/optuna_best.json`.

---

## 📈 Feature Importance (XGBoost)

Run feature importance analysis:

```bash
python scripts/feature_importance.py --ticker GS --start 2010-01-01 --end 2018-12-31
```

Saved to `outputs/feature_importance_xgb.csv`.

---

## 🔧 Configuration (Config)

Key parameters (see `src/config.py`):

- **General**: `ticker`, `start`, `end`, `test_years`  
- **GAN**: `seq_len`, `batch_size`, `lr_g`, `lr_d`, `critic_steps`, `hidden_size`  
- **Generator**: `generator` (`'lstm'|'tst'`), `d_model`, `nhead`, `num_layers_tst`, `dropout_tst`  
- **Loss weights**: `adv_weight`, `l1_weight`, `cls_weight`, `q_weight`  
- **Quantile forecasts**: `quantiles` (default `(0.1, 0.5, 0.9)`)  
- **AE/PCA**: `ae_hidden`, `ae_bottleneck`, `ae_epochs`, `pca_components`  

---

## 🧾 Outputs

- `outputs/test_predictions.csv` — predictions (y_true, y_pred, y_logit, quantiles).  
- `outputs/metrics_walk_forward.csv` — aggregated metrics across splits.  
- `outputs/feature_importance_xgb.csv` — feature importance.  
- `outputs/optuna_best.json` — Optuna best params.  

---

## 🛠️ Technical Details

- PyTorch ≥2.1, AMP (mixed precision), `torch.compile` acceleration.  
- Optimizers: AdamW.  
- DataLoader: tuned for performance (`num_workers`, `persistent_workers`, `prefetch_factor`, `pin_memory`).  
- Modular design (`src/features`, `src/models`, `src/utils`, `scripts`).  
- WandB integration (if `WANDB_PROJECT` is set).

---

## 📜 License

MIT License.

---

## 🙌 Credits

- [Boris Banushev](https://github.com/borisbanushev/stockpredictionai) for the original notebook and pipeline idea.  
- PyTorch community for continued framework improvements (`torch.compile`, AMP).  
- Time series research community for TST, N-BEATS, N-HiTS inspiration.  
- Paul Dikaloff - 2025

---
"# stockpredictionai-pro" 
