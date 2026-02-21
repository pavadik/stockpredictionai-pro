"""Optuna-based parameter optimisation for the momentum-trend strategy.

Loads M1 data once, **pre-aggregates** to TF1/TF2 once, then runs many
trials that only recompute indicators + signals + backtest (fast inner
loop).

Objective: **maximise average trade** (total PnL / number of trades).
"""

import logging
from dataclasses import asdict
from typing import Callable, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd

from .momentum_trend import (
    StrategyParams,
    PreAggregatedData,
    pre_aggregate,
    apply_params_fast,
    prepare_strategy_data,
    generate_signals,
)
from .backtester import (
    simulate_trades, compute_metrics,
    simulate_trades_fast, compute_metrics_from_pnl,
)

logger = logging.getLogger(__name__)

_MIN_TRADES = 30


# -----------------------------------------------------------------------
# Parameter space
# -----------------------------------------------------------------------

def _suggest_params(trial: optuna.Trial,
                    base: Optional[StrategyParams] = None) -> StrategyParams:
    """Build a :class:`StrategyParams` from an Optuna trial."""
    if base is None:
        base = StrategyParams()

    lookback = trial.suggest_int("lookback", 50, 300, step=5)
    length = trial.suggest_int("length", 50, 300, step=5)
    lookback2 = trial.suggest_int("lookback2", 10, 100, step=5)
    length2 = trial.suggest_int("length2", 30, 200, step=5)
    min_s = trial.suggest_int("min_s", 100, 500, step=10)
    max_s = trial.suggest_int("max_s", 200, 700, step=10)
    koeff1 = trial.suggest_float("koeff1", 0.90, 1.10)
    koeff2 = trial.suggest_float("koeff2", 0.90, 1.10)
    mmcoff = trial.suggest_int("mmcoff", 5, 30)
    exitday = trial.suggest_categorical("exitday", [0, 1])
    sdel_day = trial.suggest_categorical("sdel_day", [0, 1])

    if max_s <= min_s + 60:
        raise optuna.TrialPruned("max_s must exceed min_s + 60")

    return StrategyParams(
        lookback=lookback, length=length,
        lookback2=lookback2, length2=length2,
        min_s=min_s, max_s=max_s,
        koeff1=koeff1, koeff2=koeff2,
        mmcoff=mmcoff, exitday=exitday, sdel_day=sdel_day,
        capital=base.capital,
        point_value_mult=base.point_value_mult,
        tf1_minutes=base.tf1_minutes,
        tf2_minutes=base.tf2_minutes,
        use_legacy_bug=base.use_legacy_bug,
    )


# -----------------------------------------------------------------------
# Objective (uses pre-aggregated data -- no per-trial aggregation)
# -----------------------------------------------------------------------

def make_objective(df_m1_train: pd.DataFrame,
                   base_params: Optional[StrategyParams] = None,
                   min_trades: int = _MIN_TRADES,
                   ) -> Callable[[optuna.Trial], float]:
    """Return a closure suitable for ``study.optimize(objective, ...)``.

    **Pre-aggregates** M1 data once; each trial only recomputes the
    time filter, ATR, indicators, signals, and backtest.
    """
    if base_params is None:
        base_params = StrategyParams()

    logger.info("Pre-aggregating M1 data to TF1=%d / TF2=%d ...",
                base_params.tf1_minutes, base_params.tf2_minutes)
    pre = pre_aggregate(df_m1_train, base_params)
    logger.info("Pre-aggregation done: %d TF2 bars.",
                len(pre.df_aligned))

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, base_params)

        df = apply_params_fast(pre, params)

        # Use Numba-JIT fast path for the backtest
        pnls = simulate_trades_fast(df, params)

        if len(pnls) < min_trades:
            raise optuna.TrialPruned(
                f"Only {len(pnls)} trades (min {min_trades})")

        metrics = compute_metrics_from_pnl(pnls)
        avg_trade = metrics["avg_trade"]

        trial.set_user_attr("num_trades", metrics["num_trades"])
        trial.set_user_attr("total_pnl", metrics["total_pnl"])
        trial.set_user_attr("win_rate", metrics["win_rate"])
        trial.set_user_attr("profit_factor", metrics["profit_factor"])
        trial.set_user_attr("max_drawdown", metrics["max_drawdown"])
        trial.set_user_attr("sharpe", metrics["sharpe"])

        return avg_trade

    return objective


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def run_optimization(
    df_m1_train: pd.DataFrame,
    n_trials: int = 2000,
    base_params: Optional[StrategyParams] = None,
    min_trades: int = _MIN_TRADES,
    seed: int = 42,
    study_name: str = "block_e_momentum_trend",
    n_jobs: int = 1,
) -> optuna.Study:
    """Run the full Optuna study and return the study object."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    objective = make_objective(df_m1_train, base_params, min_trades)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs,
                   show_progress_bar=True)

    logger.info(
        "Optimisation complete: %d trials, best avg_trade=%.2f",
        len(study.trials), study.best_value,
    )
    return study


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def study_to_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Convert completed study trials to a DataFrame for CSV export."""
    records: List[Dict] = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, "avg_trade": t.value}
        row.update(t.params)
        row.update(t.user_attrs)
        records.append(row)
    return pd.DataFrame(records)


def best_params_to_strategy(study: optuna.Study,
                            base: Optional[StrategyParams] = None,
                            ) -> StrategyParams:
    """Reconstruct a :class:`StrategyParams` from the best trial."""
    if base is None:
        base = StrategyParams()
    bp = study.best_params
    return StrategyParams(
        lookback=bp["lookback"], length=bp["length"],
        lookback2=bp["lookback2"], length2=bp["length2"],
        min_s=bp["min_s"], max_s=bp["max_s"],
        koeff1=bp["koeff1"], koeff2=bp["koeff2"],
        mmcoff=bp["mmcoff"], exitday=bp["exitday"], sdel_day=bp["sdel_day"],
        capital=base.capital,
        point_value_mult=base.point_value_mult,
        tf1_minutes=base.tf1_minutes,
        tf2_minutes=base.tf2_minutes,
        use_legacy_bug=base.use_legacy_bug,
    )
