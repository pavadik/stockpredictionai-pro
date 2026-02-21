"""GPU-accelerated batch parameter evaluation using PyTorch.

Instead of evaluating one parameter set at a time (Optuna), this module
moves the pre-aggregated price data to the GPU and evaluates **thousands
of random parameter combinations in parallel**.

Key optimisations:
    1. Pre-compute ``sign(close[t] - close[t-s])`` for ALL possible
       shift values -> (T, S_max) tensor, computed once.
    2. For any (lookback, lookback2), TrendScore is a slice-sum of
       that tensor -> pure indexing, no recomputation.
    3. Rolling means via ``F.avg_pool1d`` (GPU-native 1-D pooling).
    4. Multi-timeframe grid: pre-aggregate for several (tf1, tf2) combos.

Optimisation target: **per-1-contract avg_trade** (signal quality, no
position-sizing bias).

Requires: ``torch`` with CUDA.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .momentum_trend import (
    StrategyParams,
    PreAggregatedData,
    pre_aggregate,
    _minutes_from_day_start,
    apply_expiration_filter,
)

logger = logging.getLogger(__name__)

_MOMENTUM_OFFSETS = 10
_MIN_TRADES = 30

# Default tf grids to search over
TF1_GRID = (120, 150, 180, 210, 240, 300)
TF2_GRID = (45, 60, 90, 120)


# -----------------------------------------------------------------------
# GPU pre-computation
# -----------------------------------------------------------------------

@dataclass
class GPUData:
    """All tensors living on the GPU, pre-computed once for one (tf1, tf2)."""

    device: torch.device

    close_tf1: torch.Tensor       # (T,) float32
    close_tf2: torch.Tensor       # (T,) float32
    open_tf2: torch.Tensor        # (T,) float32
    high_tf2: torch.Tensor        # (T,) float32
    low_tf2: torch.Tensor         # (T,) float32
    true_range: np.ndarray        # (T,) float64 -- raw TR; ATR = rolling mean by mmcoff

    allow_trade: torch.Tensor     # (T,) bool
    elapsed_min: torch.Tensor     # (T,) int32

    sign_tf1: torch.Tensor        # (T, max_shift) int8
    sign_tf2: torch.Tensor        # (T, max_shift) int8

    max_shift: int
    T: int
    dates: torch.Tensor           # (T,) int32
    is_week_end: np.ndarray       # (T,) bool -- last bar of each week


def _build_sign_matrix(close: torch.Tensor, max_shift: int,
                       device: torch.device) -> torch.Tensor:
    T = close.shape[0]
    mat = torch.zeros(T, max_shift, dtype=torch.int8, device=device)
    for s in range(1, max_shift):
        shifted = torch.roll(close, s)
        diff = close - shifted
        diff[:s] = 0.0
        mat[:, s] = torch.sign(diff).to(torch.int8)
    return mat


def prepare_gpu_data(pre: PreAggregatedData,
                     max_lookback: int = 310,
                     device: Optional[torch.device] = None) -> GPUData:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pre.df_aligned
    T = len(df)

    close_tf1 = torch.tensor(df["close_tf1"].values, dtype=torch.float32,
                              device=device)
    close_tf2 = torch.tensor(df["close"].values, dtype=torch.float32,
                              device=device)
    open_tf2 = torch.tensor(df["open"].values, dtype=torch.float32,
                             device=device)
    high_tf2 = torch.tensor(df["high"].values, dtype=torch.float32,
                             device=device)
    low_tf2 = torch.tensor(df["low"].values, dtype=torch.float32,
                            device=device)

    true_range_np = np.asarray(pre.true_range.values, dtype=np.float64)

    allow = torch.tensor(df["allow_trade"].values, dtype=torch.bool,
                         device=device)
    elapsed = torch.tensor(pre.elapsed_minutes.values, dtype=torch.int32,
                           device=device)

    sign_tf1 = _build_sign_matrix(close_tf1, max_lookback + 1, device)
    sign_tf2 = _build_sign_matrix(close_tf2, max_lookback + 1, device)

    dates_ord = torch.tensor(
        np.array([d.toordinal() for d in df.index.date], dtype=np.int32),
        device=device)

    week_id = pre.week_id
    is_week_end = np.append(week_id[:-1] != week_id[1:], [True])

    return GPUData(
        device=device,
        close_tf1=close_tf1, close_tf2=close_tf2, open_tf2=open_tf2,
        high_tf2=high_tf2, low_tf2=low_tf2, true_range=true_range_np,
        allow_trade=allow, elapsed_min=elapsed,
        sign_tf1=sign_tf1, sign_tf2=sign_tf2,
        max_shift=max_lookback + 1, T=T, dates=dates_ord,
        is_week_end=is_week_end,
    )


# -----------------------------------------------------------------------
# Vectorised GPU operations
# -----------------------------------------------------------------------

def _trend_score_gpu(sign_matrix: torch.Tensor,
                     lookback: int) -> torch.Tensor:
    start = lookback + 1
    end = lookback + _MOMENTUM_OFFSETS + 1
    return sign_matrix[:, start:end].sum(dim=1).to(torch.int32)


def _rolling_mean_gpu(x: torch.Tensor, window: int) -> torch.Tensor:
    T = x.shape[0]
    x3d = x.reshape(1, 1, T).float()
    padded = F.pad(x3d, (window - 1, 0), mode="constant", value=0.0)
    pooled = F.avg_pool1d(padded, kernel_size=window, stride=1, padding=0)
    result = pooled.reshape(T)
    cumsum = torch.cumsum(x.float(), dim=0)
    counts = torch.arange(1, T + 1, device=x.device, dtype=torch.float32)
    expanding = cumsum / counts
    mask = torch.arange(T, device=x.device) < window
    result[mask] = expanding[mask]
    return result


# -----------------------------------------------------------------------
# Batch evaluation (per-1-contract objective)
# -----------------------------------------------------------------------

_SL_TYPES_ALL = ("none", "fixed", "atr", "trail_atr", "trail_pts",
                  "breakeven", "time")


def _sample_sl_params(rng: np.random.Generator,
                      sl_types: tuple = _SL_TYPES_ALL) -> dict:
    """Sample a random stop-loss configuration."""
    sl_type = str(rng.choice(sl_types))
    d: dict = {"sl_type": sl_type}
    if sl_type == "fixed":
        d["sl_pts"] = float(rng.integers(500, 10000, endpoint=True)) // 500 * 500
    elif sl_type == "atr":
        d["sl_atr_mult"] = round(float(rng.uniform(0.5, 5.0)), 2)
    elif sl_type == "trail_atr":
        d["trail_atr"] = round(float(rng.uniform(0.5, 5.0)), 2)
    elif sl_type == "trail_pts":
        d["trail_pts"] = float(rng.integers(500, 8000, endpoint=True)) // 500 * 500
    elif sl_type == "breakeven":
        d["be_target"] = float(rng.integers(500, 5000, endpoint=True)) // 500 * 500
        d["be_offset"] = float(rng.choice([0, 100, 200, 500]))
    elif sl_type == "time":
        d["time_bars"] = int(rng.integers(5, 50, endpoint=True))
        d["time_target"] = float(rng.integers(0, 2000, endpoint=True)) // 200 * 200
    return d


def _generate_params_batch(n: int,
                           rng: np.random.Generator,
                           tf1_grid: tuple,
                           tf2_grid: tuple,
                           sl_types: tuple = ("none",)) -> List[dict]:
    """Generate *n* random parameter dictionaries including tf1/tf2."""
    params = []
    for _ in range(n):
        min_s = int(rng.integers(60, 500, endpoint=True)) // 10 * 10
        max_s = int(rng.integers(min_s + 70, 700, endpoint=True)) // 10 * 10
        p = {
            "tf1": int(rng.choice(tf1_grid)),
            "tf2": int(rng.choice(tf2_grid)),
            "lookback": int(rng.integers(20, 300, endpoint=True)) // 5 * 5,
            "length": int(rng.integers(20, 300, endpoint=True)) // 5 * 5,
            "lookback2": int(rng.integers(5, 100, endpoint=True)) // 5 * 5,
            "length2": int(rng.integers(15, 200, endpoint=True)) // 5 * 5,
            "min_s": min_s,
            "max_s": max_s,
            "koeff1": float(rng.uniform(0.90, 1.10)),
            "koeff2": float(rng.uniform(0.90, 1.10)),
            "mmcoff": int(rng.integers(5, 30, endpoint=True)),
            "exitday": int(rng.integers(0, 1, endpoint=True)),
            "sdel_day": int(rng.integers(0, 1, endpoint=True)),
        }
        p.update(_sample_sl_params(rng, sl_types))
        params.append(p)
    return params


REFINE_TF1_GRID = (150, 165, 180, 195, 210)
REFINE_TF2_GRID = (45, 50, 55, 60, 70, 75)


def _generate_params_refined(n: int,
                              centre: dict,
                              rng: np.random.Generator,
                              tf1_grid: tuple = REFINE_TF1_GRID,
                              tf2_grid: tuple = REFINE_TF2_GRID,
                              sl_types: tuple = ("none",),
                              ) -> List[dict]:
    """Generate *n* parameter dicts sampled from a narrow neighbourhood."""

    def _clamp_int(val, lo, hi, step):
        v = int(rng.integers(lo, hi, endpoint=True)) // step * step
        return max(lo, min(hi, v))

    c = centre
    params = []
    for _ in range(n):
        lb = _clamp_int(0, max(20, c["lookback"] - 30), c["lookback"] + 30, 5)
        ln = _clamp_int(0, max(20, c["length"] - 40), c["length"] + 40, 5)
        lb2 = _clamp_int(0, max(5, c["lookback2"] - 25), c["lookback2"] + 25, 5)
        ln2 = _clamp_int(0, max(15, c["length2"] - 30), c["length2"] + 30, 5)
        ms = _clamp_int(0, max(60, c["min_s"] - 50), c["min_s"] + 50, 10)
        mx = _clamp_int(0, ms + 70, c["max_s"] + 60, 10)
        if mx <= ms + 60:
            mx = ms + 70

        p = {
            "tf1": int(rng.choice(tf1_grid)),
            "tf2": int(rng.choice(tf2_grid)),
            "lookback": lb,
            "length": ln,
            "lookback2": lb2,
            "length2": ln2,
            "min_s": ms,
            "max_s": mx,
            "koeff1": float(rng.uniform(
                max(0.90, c["koeff1"] - 0.03), min(1.10, c["koeff1"] + 0.03))),
            "koeff2": float(rng.uniform(
                max(0.90, c["koeff2"] - 0.03), min(1.10, c["koeff2"] + 0.03))),
            "mmcoff": int(rng.integers(
                max(5, c["mmcoff"] - 5), min(30, c["mmcoff"] + 5), endpoint=True)),
            "exitday": c.get("exitday", 0),
            "sdel_day": c.get("sdel_day", 1),
        }
        p.update(_sample_sl_params(rng, sl_types))
        params.append(p)
    return params


def _compute_atr_numpy(true_range: np.ndarray, period: int) -> np.ndarray:
    """Simple rolling mean ATR from pre-computed true range (CPU)."""
    out = np.empty_like(true_range)
    cumsum = 0.0
    for i in range(len(true_range)):
        cumsum += true_range[i]
        if i >= period:
            cumsum -= true_range[i - period]
            out[i] = cumsum / period
        else:
            out[i] = cumsum / (i + 1)
    return out


@torch.no_grad()
def evaluate_single_gpu(gd: GPUData, p: dict,
                        force_exit_week: bool = False,
                        min_trades: int = _MIN_TRADES) -> Optional[Dict[str, float]]:
    """Evaluate one parameter set.  Returns per-1-contract metrics.

    PnL is computed as ``exit_price - entry_price`` (1 contract) so
    that the optimizer rewards signal quality, not position sizing.
    """
    T = gd.T

    ts1 = _trend_score_gpu(gd.sign_tf1, p["lookback"]).float()
    tsa1 = _rolling_mean_gpu(ts1, p["length"])
    sma1 = _rolling_mean_gpu(gd.close_tf1, p["length"])

    ts2 = _trend_score_gpu(gd.sign_tf2, p["lookback2"]).float()
    tsa2 = _rolling_mean_gpu(ts2, p["length2"])
    sma2 = _rolling_mean_gpu(gd.close_tf2, p["length2"])

    k1, k2 = p["koeff1"], p["koeff2"]
    direction = p.get("direction", "long")
    is_short = direction == "short"

    if is_short:
        cond_tf1 = (ts1 < tsa1 * (2 - k1)) & (gd.close_tf1 < sma1 * (2 - k1))
        cond_tf2 = (ts2 < tsa2 * (2 - k2)) & (gd.close_tf2 < sma2 * (2 - k2))
    else:
        cond_tf1 = (ts1 > tsa1 * k1) & (gd.close_tf1 > sma1 * k1)
        cond_tf2 = (ts2 > tsa2 * k2) & (gd.close_tf2 > sma2 * k2)

    pnl_sign = -1.0 if is_short else 1.0

    in_window = (gd.elapsed_min >= p["min_s"]) & (gd.elapsed_min < p["max_s"])
    active = gd.allow_trade & in_window
    entry_flag = active & cond_tf1 & cond_tf2

    entry_np = entry_flag.cpu().numpy()
    allow_np = gd.allow_trade.cpu().numpy()
    in_window_np = in_window.cpu().numpy()
    close_np = gd.close_tf2.cpu().numpy()
    dates_np = gd.dates.cpu().numpy()
    week_end_np = gd.is_week_end

    sl_type = p.get("sl_type", "none")
    sl_pts = p.get("sl_pts", 0.0)
    sl_atr_mult = p.get("sl_atr_mult", 0.0)
    trail_atr = p.get("trail_atr", 0.0)
    trail_pts = p.get("trail_pts", 0.0)
    be_target = p.get("be_target", 0.0)
    be_offset = p.get("be_offset", 0.0)
    time_bars = p.get("time_bars", 0)
    time_target = p.get("time_target", 0.0)
    use_sl = sl_type != "none"

    atr_np: Optional[np.ndarray] = None
    if use_sl and sl_type in ("atr", "trail_atr"):
        atr_np = _compute_atr_numpy(gd.true_range, p.get("mmcoff", 17))

    exitday = p["exitday"]
    sdel_day = p["sdel_day"]
    exit_week = force_exit_week or p.get("exit_week", 0) == 1

    position = False
    entry_price = 0.0
    best_price = 0.0
    atr_at_entry = 1.0
    bars_in_trade = 0
    pnls: List[float] = []
    one_entry_today = False
    current_date = -1

    for i in range(T):
        d = dates_np[i]
        if d != current_date:
            current_date = d
            one_entry_today = False

        if position and not allow_np[i]:
            pnls.append(pnl_sign * (float(close_np[i]) - entry_price))
            position = False
            continue

        # --- stop-loss check ---
        if position and use_sl:
            bars_in_trade += 1
            c = float(close_np[i])
            if is_short:
                best_price = min(best_price, c)
            else:
                best_price = max(best_price, c)

            unrealised = pnl_sign * (c - entry_price)
            stop_hit = False
            if sl_type == "fixed" and unrealised < -sl_pts:
                stop_hit = True
            elif sl_type == "atr" and unrealised < -sl_atr_mult * atr_at_entry:
                stop_hit = True
            elif sl_type == "trail_atr":
                retrace = pnl_sign * (best_price - c)
                if retrace > trail_atr * atr_at_entry:
                    stop_hit = True
            elif sl_type == "trail_pts":
                retrace = pnl_sign * (best_price - c)
                if retrace > trail_pts:
                    stop_hit = True
            elif sl_type == "breakeven":
                if unrealised >= be_target:
                    stop_level = entry_price + pnl_sign * be_offset
                    if pnl_sign * (c - stop_level) < 0:
                        stop_hit = True
            elif sl_type == "time":
                if bars_in_trade >= time_bars and unrealised < time_target:
                    stop_hit = True

            if stop_hit:
                pnls.append(pnl_sign * (c - entry_price))
                position = False
                continue

        if allow_np[i] and in_window_np[i]:
            if not position:
                if entry_np[i]:
                    if sdel_day == 1 and one_entry_today:
                        continue
                    entry_price = float(close_np[i])
                    position = True
                    one_entry_today = True
                    best_price = entry_price
                    bars_in_trade = 0
                    if atr_np is not None:
                        atr_at_entry = max(float(atr_np[i]), 1.0)
                    else:
                        atr_at_entry = 1.0
            else:
                if not entry_np[i]:
                    pnls.append(pnl_sign * (float(close_np[i]) - entry_price))
                    position = False
        elif position and exitday == 1:
            pnls.append(pnl_sign * (float(close_np[i]) - entry_price))
            position = False

        if position and exit_week and week_end_np[i]:
            pnls.append(pnl_sign * (float(close_np[i]) - entry_price))
            position = False

    if position:
        pnls.append(pnl_sign * (float(close_np[-1]) - entry_price))

    n_trades = len(pnls)
    if n_trades < min_trades:
        return None

    pnl_arr = np.array(pnls, dtype=np.float64)
    total = float(pnl_arr.sum())
    avg = total / n_trades
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0

    cum = np.cumsum(pnl_arr)
    rmax = np.maximum.accumulate(cum)
    mdd = float((rmax - cum).max()) if n_trades else 0.0

    return {
        "avg_trade_1c": avg,
        "num_trades": n_trades,
        "total_pts": total,
        "win_rate": len(wins) / n_trades,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "max_dd_1c": mdd,
    }


# -----------------------------------------------------------------------
# Multi-timeframe pre-aggregation grid
# -----------------------------------------------------------------------

def _build_tf_grid(df_m1: pd.DataFrame,
                   tf1_grid: tuple,
                   tf2_grid: tuple,
                   device: torch.device,
                   max_lookback: int = 310,
                   ) -> Dict[Tuple[int, int], GPUData]:
    """Pre-aggregate for every (tf1, tf2) combo and move to GPU.

    Returns dict mapping ``(tf1, tf2) -> GPUData``.
    """
    grid: Dict[Tuple[int, int], GPUData] = {}
    for tf1 in tf1_grid:
        for tf2 in tf2_grid:
            if tf1 <= tf2:
                continue  # TF1 must be the slower timeframe
            key = (tf1, tf2)
            params = StrategyParams(tf1_minutes=tf1, tf2_minutes=tf2)
            pre = pre_aggregate(df_m1, params)
            gd = prepare_gpu_data(pre, max_lookback=max_lookback,
                                  device=device)
            grid[key] = gd
            logger.info("  TF grid (%d, %d): %d bars", tf1, tf2, gd.T)
    return grid


# -----------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------

def run_gpu_optimization(
    df_m1: pd.DataFrame,
    n_samples: int = 5000,
    base_params: Optional[StrategyParams] = None,
    min_trades: int = _MIN_TRADES,
    seed: int = 42,
    device: Optional[torch.device] = None,
    top_k: int = 20,
    tf1_grid: tuple = TF1_GRID,
    tf2_grid: tuple = TF2_GRID,
    force_exit_week: bool = False,
    refine_centre: Optional[dict] = None,
    direction: str = "long",
    sl_types: tuple = ("none",),
) -> Tuple[List[dict], pd.DataFrame]:
    """Evaluate *n_samples* random parameter sets on GPU.

    Optimises **per-1-contract avg_trade** (no position sizing bias).
    Also optimises tf1/tf2 timeframes from the provided grids.

    Args:
        force_exit_week: If True, always close positions at end of week.
        refine_centre: If given, use narrow refined sampling around this
            centre point instead of broad random search.

    Returns:
        (top_k_results, full_results_df)
    """
    if base_params is None:
        base_params = StrategyParams()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = "REFINE" if refine_centre else "BROAD"
    print(f"  [GPU] Building TF grid ({mode}): tf1={tf1_grid}, tf2={tf2_grid} ...")
    t0 = time.time()
    grid = _build_tf_grid(df_m1, tf1_grid, tf2_grid, device)
    combo_count = len(grid)
    print(f"  [GPU] {combo_count} TF combos pre-aggregated in {time.time()-t0:.1f}s")
    for (t1, t2), gd in grid.items():
        print(f"    ({t1}, {t2}): {gd.T} bars")

    rng = np.random.default_rng(seed)
    if refine_centre is not None:
        param_batch = _generate_params_refined(
            n_samples, refine_centre, rng, tf1_grid, tf2_grid, sl_types)
    else:
        param_batch = _generate_params_batch(n_samples, rng, tf1_grid, tf2_grid,
                                             sl_types)

    extra = " + exit_week" if force_exit_week else ""
    sl_info = f", sl_types={sl_types}" if sl_types != ("none",) else ""
    print(f"  [GPU] Evaluating {n_samples} parameter sets "
          f"(objective: per-1-contract avg_trade{extra}{sl_info}, "
          f"min_trades={min_trades}) ...")
    t0 = time.time()
    results: List[dict] = []
    valid = 0
    skipped_tf = 0

    for idx, p in enumerate(param_batch):
        p["direction"] = direction
        key = (p["tf1"], p["tf2"])
        if key not in grid:
            skipped_tf += 1
            continue
        gd = grid[key]
        metrics = evaluate_single_gpu(gd, p, force_exit_week=force_exit_week,
                                         min_trades=min_trades)
        if metrics is not None:
            row = {**p, **metrics}
            results.append(row)
            valid += 1

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"    {idx+1}/{n_samples}  valid={valid}  "
                  f"{rate:.0f} trials/s  elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  [GPU] Done: {valid}/{n_samples} valid in {elapsed:.1f}s "
          f"({n_samples/elapsed:.0f} trials/s)")
    if skipped_tf:
        print(f"  [GPU] Skipped {skipped_tf} trials (tf1 <= tf2)")

    if not results:
        return [], pd.DataFrame()

    results.sort(key=lambda r: r["avg_trade_1c"], reverse=True)
    df_all = pd.DataFrame(results)

    top = results[:top_k]
    print(f"  [GPU] Top-{min(top_k, len(top))} avg_trade (per 1 contract, pts):")
    for i, r in enumerate(top[:5]):
        sl_tag = f"  sl={r.get('sl_type','none')}" if r.get("sl_type", "none") != "none" else ""
        print(f"    #{i+1}: avg_1c={r['avg_trade_1c']:>10.1f} pts  "
              f"trades={r['num_trades']}  wr={r['win_rate']:.1%}  "
              f"PF={r['profit_factor']:.2f}  "
              f"tf=({r['tf1']},{r['tf2']}){sl_tag}")

    # Free GPU memory from all grid entries
    del grid
    torch.cuda.empty_cache()

    return top, df_all


def gpu_best_to_params(best: dict,
                       base: Optional[StrategyParams] = None,
                       exit_week: int = 0,
                       direction: str = "long") -> StrategyParams:
    """Convert a GPU result dict back to :class:`StrategyParams`."""
    if base is None:
        base = StrategyParams()
    return StrategyParams(
        lookback=best["lookback"], length=best["length"],
        lookback2=best["lookback2"], length2=best["length2"],
        min_s=best["min_s"], max_s=best["max_s"],
        koeff1=best["koeff1"], koeff2=best["koeff2"],
        mmcoff=best["mmcoff"], exitday=best["exitday"],
        sdel_day=best["sdel_day"],
        exit_week=exit_week,
        direction=direction,
        capital=base.capital,
        point_value_mult=base.point_value_mult,
        tf1_minutes=best.get("tf1", base.tf1_minutes),
        tf2_minutes=best.get("tf2", base.tf2_minutes),
        use_legacy_bug=base.use_legacy_bug,
        sl_type=best.get("sl_type", "none"),
        sl_pts=best.get("sl_pts", 0.0),
        sl_atr_mult=best.get("sl_atr_mult", 0.0),
        trail_atr=best.get("trail_atr", 0.0),
        trail_pts=best.get("trail_pts", 0.0),
        be_target=best.get("be_target", 0.0),
        be_offset=best.get("be_offset", 0.0),
        time_bars=best.get("time_bars", 0),
        time_target=best.get("time_target", 0.0),
    )
