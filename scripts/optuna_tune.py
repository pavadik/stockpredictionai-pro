"""Optuna hyperparameter tuning for StockPredictionAI Pro.

Usage:
    python scripts/optuna_tune.py --ticker GS --start 2010-01-01 --end 2018-12-31 --trials 20
    python scripts/optuna_tune.py --ticker SBRF --data_source local --data_path G:\\data2 --timeframe D1 --trials 20
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import optuna
from src.config import Config
from src.data import build_panel_auto
from src.train import build_features_safe, fit_transforms, run_one_split, set_global_seed
from src.dataset import train_test_split_by_years


def objective(trial, panel, base_cfg):
    """Optuna objective: train a short GAN and return validation MAE."""
    set_global_seed(42)

    gen = trial.suggest_categorical("generator", ["lstm", "tst"])
    seq = trial.suggest_categorical("seq_len", [6, 10, 12, 16, 24])
    bs = trial.suggest_categorical("batch_size", [128, 256, 512])
    hs = trial.suggest_categorical("hidden_size", [32, 64, 128])
    lr_g = trial.suggest_float("lr_g", 5e-4, 5e-3, log=True)
    lr_d = trial.suggest_float("lr_d", 1e-5, 5e-4, log=True)
    cs = trial.suggest_int("critic_steps", 3, 10)
    dm = trial.suggest_categorical("d_model", [32, 64, 128])
    nh = trial.suggest_categorical("nhead", [2, 4])
    nl = trial.suggest_int("num_layers_tst", 1, 4)
    dr = trial.suggest_float("dropout_tst", 0.0, 0.3)
    l1w = trial.suggest_float("l1_weight", 0.1, 1.0)
    clsw = trial.suggest_float("cls_weight", 0.05, 0.5)
    qw = trial.suggest_float("q_weight", 0.3, 2.0)
    gc = trial.suggest_float("grad_clip", 0.5, 5.0)

    if dm % nh != 0:
        raise optuna.TrialPruned("d_model must be divisible by nhead")

    cfg = Config(
        ticker=base_cfg.ticker,
        start=base_cfg.start,
        end=base_cfg.end,
        test_years=base_cfg.test_years,
        data_source=base_cfg.data_source,
        timeframe=base_cfg.timeframe,
        data_path=base_cfg.data_path,
        local_raw_source=base_cfg.local_raw_source,
        use_arima=base_cfg.use_arima,
        use_sentiment=base_cfg.use_sentiment,
        use_volume_features=base_cfg.use_volume_features,
        generator=gen,
        seq_len=seq,
        batch_size=bs,
        hidden_size=hs,
        lr_g=lr_g,
        lr_d=lr_d,
        critic_steps=cs,
        n_epochs=10,
        d_model=dm,
        nhead=nh,
        num_layers_tst=nl,
        dropout_tst=dr,
        adv_weight=1.0,
        l1_weight=l1w,
        cls_weight=clsw,
        q_weight=qw,
        grad_clip=gc,
        ae_epochs=5,
    )
    if cfg.data_source == "local" or cfg.timeframe != "D1":
        cfg.apply_timeframe_defaults()

    try:
        train_panel, test_panel = train_test_split_by_years(panel, cfg.test_years)
        tr_feat, te_feat = build_features_safe(train_panel, test_panel, cfg)
        tr_all, te_all = fit_transforms(tr_feat, te_feat, cfg)
        met, _, _, _, _, _ = run_one_split(tr_all, te_all, cfg)
    except Exception as e:
        print(f"  Trial {trial.number} FAILED: {e}")
        raise optuna.TrialPruned(str(e))
    return met["MAE"]


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument('--ticker', default='GS')
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default='2018-12-31')
    parser.add_argument('--test_years', type=int, default=2)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--data_source', default='yfinance', choices=['yfinance', 'local'])
    parser.add_argument('--timeframe', default='D1',
                        help='M1, M3, M5, M7, M14, M30, H1, H4, D1, or tick')
    parser.add_argument('--data_path', default='')
    parser.add_argument('--raw_source', default='m1', choices=['m1', 'ticks'])
    args = parser.parse_args()

    cfg_base = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        test_years=args.test_years, data_source=args.data_source,
        timeframe=args.timeframe, data_path=args.data_path,
        local_raw_source=args.raw_source,
    )
    if cfg_base.data_source == "local" or cfg_base.timeframe != "D1":
        cfg_base.apply_timeframe_defaults()

    print(f"Loading data panel for {args.ticker} "
          f"(source={cfg_base.data_source}, tf={cfg_base.timeframe})...")
    panel = build_panel_auto(cfg_base)

    study = optuna.create_study(direction="minimize", study_name="stock_gan_tune")
    study.optimize(
        lambda trial: objective(trial, panel, cfg_base),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest MAE: {best.value:.6f}")
    print(f"Best params: {best.params}")

    os.makedirs('outputs/experiments', exist_ok=True)
    result = {"best_mae": best.value, "best_params": best.params,
              "n_trials": len(study.trials),
              "n_pruned": len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.PRUNED])}
    with open('outputs/optuna_best.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("Saved: outputs/optuna_best.json")

    import pandas as pd
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = {"trial": t.number, "MAE": t.value}
            row.update(t.params)
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows).sort_values("MAE")
        df.to_csv("outputs/experiments/optuna_trials.csv", index=False)
        print(f"Saved: outputs/experiments/optuna_trials.csv ({len(rows)} completed trials)")
        print("\nTop-5 trials:")
        print(df.head().to_string(index=False))


if __name__ == '__main__':
    main()
