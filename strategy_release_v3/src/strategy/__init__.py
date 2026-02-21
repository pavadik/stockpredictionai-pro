"""Block E: Rule-based momentum-trend strategy for MOEX futures.

Ports the MultiCharts EasyLanguage momentum scoring strategy to Python
with multi-timeframe analysis, session-aware aggregation, vectorised
signal generation, and Optuna-based parameter optimisation.
"""

from .momentum_trend import (  # noqa: F401
    StrategyParams,
    PreAggregatedData,
    compute_trend_score,
    align_timeframes,
    prepare_strategy_data,
    pre_aggregate,
    apply_params_fast,
    generate_signals,
    apply_expiration_filter,
    apply_time_filter,
)
from .backtester import (  # noqa: F401
    Trade,
    simulate_trades,
    simulate_trades_fast,
    compute_metrics,
    compute_metrics_from_pnl,
)
