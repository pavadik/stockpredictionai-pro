"""XGBoost feature importance analysis.

NOTE: This script uses in-sample Fourier/ARIMA (look-ahead bias) for
feature importance ranking only.  This is acceptable because the goal
here is exploratory (which features matter?), not prediction accuracy.

Usage:
    python scripts/feature_importance.py --ticker GS --start 2010-01-01 --end 2018-12-31
    python scripts/feature_importance.py --ticker SBRF --data_source local --data_path G:\\data2 --timeframe D1
"""
import argparse
import os
import sys
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.data import build_panel_auto
from src.features.fourier import fourier_multi
from src.features.arima_feat import arima_in_sample
from src.features.autoencoder import fit_autoencoder
from src.features.pca_eigen import eigen_portfolio


def assemble_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assemble full feature set for importance analysis (exploratory, in-sample)."""
    fft = fourier_multi(df[cfg.ticker], cfg.fourier_components)
    feat = df.join(fft)

    if cfg.use_arima:
        ar = arima_in_sample(df[cfg.ticker], cfg.arima_order)
        feat = feat.join(ar)

    feat = feat.dropna()
    ae_df, _ = fit_autoencoder(feat, hidden=cfg.ae_hidden,
                               bottleneck=cfg.ae_bottleneck, epochs=cfg.ae_epochs)
    feat = feat.join(ae_df).dropna()
    n_comp = min(cfg.pca_components, feat.shape[1] - 1)
    pca_df = eigen_portfolio(feat, n_components=n_comp)
    feat = feat.join(pca_df).dropna()
    return feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='GS')
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default='2018-12-31')
    parser.add_argument('--data_source', default='yfinance', choices=['yfinance', 'local'])
    parser.add_argument('--timeframe', default='D1',
                        help='M1, M3, M5, M7, M14, M30, H1, H4, D1, or tick')
    parser.add_argument('--data_path', default='')
    parser.add_argument('--raw_source', default='m1', choices=['m1', 'ticks'])
    args = parser.parse_args()

    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        data_source=args.data_source, timeframe=args.timeframe,
        data_path=args.data_path, local_raw_source=args.raw_source,
    )
    if cfg.data_source == "local" or cfg.timeframe != "D1":
        cfg.apply_timeframe_defaults()

    print(f"Loading panel for {cfg.ticker} "
          f"(source={cfg.data_source}, tf={cfg.timeframe})...")
    df = build_panel_auto(cfg)
    feat = assemble_features(df, cfg)
    X = feat.drop(columns=[cfg.ticker]).values
    y = feat[cfg.ticker].diff().shift(-1).dropna().values
    X = X[:len(y)]
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X, y)
    imp = model.feature_importances_
    s = pd.Series(
        imp, index=feat.drop(columns=[cfg.ticker]).columns, name="importance"
    ).sort_values(ascending=False)
    os.makedirs('outputs', exist_ok=True)
    s.to_csv('outputs/feature_importance_xgb.csv', header=True)
    print(f"Top 10 features:\n{s.head(10)}")
    print("Saved: outputs/feature_importance_xgb.csv")


if __name__ == '__main__':
    main()
