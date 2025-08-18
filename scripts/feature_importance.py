import argparse
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from src.config import Config
from src.data import build_panel
from src.features.fourier import fourier_approx
from src.features.arima_feat import arima_in_sample
from src.features.autoencoder import fit_autoencoder
from src.features.pca_eigen import eigen_portfolio

def assemble_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    fft = fourier_approx(df[cfg.ticker], cfg.fourier_k)
    ar = arima_in_sample(df[cfg.ticker], cfg.arima_order)
    feat = df.join(fft).join(ar).dropna()
    ae_df, _ = fit_autoencoder(feat, hidden=cfg.ae_hidden, bottleneck=cfg.ae_bottleneck, epochs=cfg.ae_epochs)
    feat = feat.join(ae_df).dropna()
    pca_df = eigen_portfolio(feat, n_components=cfg.pca_components)
    feat = feat.join(pca_df).dropna()
    return feat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='GS')
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default='2018-12-31')
    args = parser.parse_args()
    cfg = Config(ticker=args.ticker, start=args.start, end=args.end)
    df = build_panel(cfg.ticker, cfg.start, cfg.end)
    feat = assemble_features(df, cfg)
    X = feat.drop(columns=[cfg.ticker]).values
    y = feat[cfg.ticker].diff().shift(-1).dropna().values
    X = X[:len(y)]
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X, y)
    imp = model.feature_importances_
    s = pd.Series(imp, index=feat.drop(columns=[cfg.ticker]).columns).sort_values(ascending=False)
    s.to_csv('outputs/feature_importance_xgb.csv')
    print("Сохранено: outputs/feature_importance_xgb.csv")

if __name__ == '__main__':
    main()
