import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import gymnasium as gym
    from gymnasium import spaces
    _IS_GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _IS_GYMNASIUM = False

# Reuse strategy core from strategy_release_v3.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../strategy_release_v3")))
from src.data_local import load_m1_bars, scan_available_tickers
from src.strategy.backtester import compute_metrics_from_pnl, simulate_trades_fast
from src.strategy.momentum_trend import StrategyParams, apply_params_fast, pre_aggregate


ACTION_LOW = np.array([0.5, 0.5, 10.0], dtype=np.float32)
ACTION_HIGH = np.array([2.0, 2.0, 500.0], dtype=np.float32)
ACTION_SHAPE = (3,)


def _clip_hp(hp: Dict[str, float]) -> Dict[str, float]:
    return {
        "koeff1": float(np.clip(hp["koeff1"], ACTION_LOW[0], ACTION_HIGH[0])),
        "koeff2": float(np.clip(hp["koeff2"], ACTION_LOW[1], ACTION_HIGH[1])),
        "sl_pts": float(np.clip(hp["sl_pts"], ACTION_LOW[2], ACTION_HIGH[2])),
    }


def _action_to_hp(action: np.ndarray) -> Dict[str, float]:
    vec = np.asarray(action, dtype=np.float32).reshape(-1)
    hp = {
        "koeff1": float(vec[0]),
        "koeff2": float(vec[1]),
        "sl_pts": float(vec[2]),
    }
    return _clip_hp(hp)


def _hp_key(hp: Dict[str, float]) -> Tuple[float, float, float]:
    return (round(hp["koeff1"], 4), round(hp["koeff2"], 4), round(hp["sl_pts"], 2))


def _build_params(base_params: StrategyParams, hp: Dict[str, float]) -> StrategyParams:
    p = StrategyParams(**asdict(base_params))
    p.koeff1 = hp["koeff1"]
    p.koeff2 = hp["koeff2"]
    p.sl_type = "fixed"
    p.sl_pts = hp["sl_pts"]
    return p


def evaluate_on_pre(pre_data, base_params: StrategyParams, hp: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    params = _build_params(base_params, hp)
    df_sig = apply_params_fast(pre_data, params)
    pnls = simulate_trades_fast(df_sig, params)
    metrics = compute_metrics_from_pnl(pnls)
    if metrics["num_trades"] < 10:
        return -100.0, metrics
    return float(metrics["avg_trade"]), metrics


def parse_konkop_oos(report_path: str) -> Dict[str, float]:
    if not os.path.exists(report_path):
        return {}
    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    marker = "COMBINED TEST / OOS"
    start_idx = text.find(marker)
    if start_idx < 0:
        return {}
    section = text[start_idx:start_idx + 5000]

    def _get_float(pattern: str):
        m = re.search(pattern, section)
        if not m:
            return None
        return float(m.group(1).replace(",", ""))

    return {
        "net_profit": _get_float(r"Net profit\s+([-\d,]+)"),
        "profit_factor": _get_float(r"Profit factor\s+([-\d.]+)"),
        "max_dd_pct": _get_float(r"Max DD\s+([-\d.]+)\s*%"),
        "avg_trade_1c": _get_float(r"Avg trade \(1c\)\s+([-\d.,]+)\s*pts"),
        "pf_1c": _get_float(r"Profit factor\(1c\)\s+([-\d.]+)"),
    }


def _load_train_oos_context(
    data_path: str,
    ticker: str,
    train_start: str,
    train_end: str,
    oos_start: str,
    oos_end: str,
) -> Tuple[StrategyParams, object, object]:
    available = scan_available_tickers(data_path, train_start, oos_end)
    if ticker not in available:
        raise RuntimeError(f"Ticker {ticker} is missing in {data_path} for {train_start}..{oos_end}")

    info = available[ticker]
    print(
        f"Ticker {ticker}: days={info['days']}, first={info['first_date']}, "
        f"last={info['last_date']}, has_m1={info['has_m1']}, has_ticks={info['has_ticks']}"
    )

    print("Loading M1 bars once for train+oos...")
    t0 = time.time()
    df_all = load_m1_bars(data_path, ticker, train_start, oos_end)
    print(f"Loaded {len(df_all):,} bars in {time.time() - t0:.1f}s")

    train_end_dt = np.datetime64(train_end)
    oos_start_dt = np.datetime64(oos_start)
    oos_end_dt = np.datetime64(oos_end)

    train_mask = df_all.index.values <= train_end_dt
    oos_mask = (df_all.index.values >= oos_start_dt) & (df_all.index.values <= oos_end_dt)
    df_train = df_all.loc[train_mask].copy()
    df_oos = df_all.loc[oos_mask].copy()
    if df_train.empty or df_oos.empty:
        raise RuntimeError("Train or OOS slice is empty. Check date ranges.")

    print(f"Train bars: {len(df_train):,}; OOS bars: {len(df_oos):,}")
    base_params = StrategyParams(direction="long", tf1_minutes=210, tf2_minutes=90)

    print("Pre-aggregating train and OOS data...")
    t1 = time.time()
    pre_train = pre_aggregate(df_train, base_params)
    pre_oos = pre_aggregate(df_oos, base_params)
    print(f"Pre-aggregation done in {time.time() - t1:.1f}s")
    return base_params, pre_train, pre_oos


class StrategyContinuousEnv(gym.Env):
    def __init__(self, pre_train, base_params: StrategyParams, episode_len: int = 10):
        super().__init__()
        self.pre_train = pre_train
        self.base_params = base_params
        self.current_step = 0
        self.max_steps = episode_len
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=ACTION_SHAPE,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = np.array([0.0], dtype=np.float32)
        if _IS_GYMNASIUM:
            return obs, {}
        return obs

    def step(self, action):
        hp = _action_to_hp(action)
        reward, _ = evaluate_on_pre(self.pre_train, self.base_params, hp)
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        obs = np.array([0.0], dtype=np.float32)
        if _IS_GYMNASIUM:
            return obs, reward, terminated, truncated, {}
        return obs, reward, terminated, {}


def _sample_candidates(model, seed: int, stochastic_samples: int, random_samples: int) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    obs = np.array([[0.0]], dtype=np.float32)
    candidates: List[Dict[str, float]] = []

    deterministic_action, _ = model.predict(obs, deterministic=True)
    candidates.append(_action_to_hp(np.asarray(deterministic_action).reshape(-1)))

    for _ in range(stochastic_samples):
        action, _ = model.predict(obs, deterministic=False)
        candidates.append(_action_to_hp(np.asarray(action).reshape(-1)))

    for _ in range(random_samples):
        vec = rng.uniform(ACTION_LOW, ACTION_HIGH).astype(np.float32)
        candidates.append(_action_to_hp(vec))

    # Add explicit baseline and boundary points.
    candidates.append({"koeff1": 1.0, "koeff2": 1.005, "sl_pts": 80.0})
    candidates.append({"koeff1": float(ACTION_LOW[0]), "koeff2": float(ACTION_LOW[1]), "sl_pts": float(ACTION_LOW[2])})
    candidates.append({"koeff1": float(ACTION_HIGH[0]), "koeff2": float(ACTION_HIGH[1]), "sl_pts": float(ACTION_HIGH[2])})

    uniq = {}
    for hp in candidates:
        hp2 = _clip_hp(hp)
        uniq[_hp_key(hp2)] = hp2
    return list(uniq.values())


def _save_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Long PPO optimization + OOS comparison vs konkop")
    ap.add_argument("--data_path", default=r"G:\data2")
    ap.add_argument("--ticker", default="RTSF")
    ap.add_argument("--train_start", default="2007-01-01")
    ap.add_argument("--train_end", default="2018-12-31")
    ap.add_argument("--oos_start", default="2019-01-01")
    ap.add_argument("--oos_end", default="2022-12-31")
    ap.add_argument("--timesteps", type=int, default=5000)
    ap.add_argument("--episode_len", type=int, default=8)
    ap.add_argument("--stochastic_candidates", type=int, default=80)
    ap.add_argument("--random_candidates", type=int, default=40)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--konkop_report",
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../strategy_release_v3/outputs/experiments/BLOCK_E_COMBINED_KONKOP.txt",
            )
        ),
    )
    ap.add_argument(
        "--output_dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs")),
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 90)
    print("PPO LONG OPTIMIZATION (TRAIN) + OOS TABLE + KONKOP COMPARISON")
    print("=" * 90)

    base_params, pre_train, pre_oos = _load_train_oos_context(
        data_path=args.data_path,
        ticker=args.ticker,
        train_start=args.train_start,
        train_end=args.train_end,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
    )

    baseline_hp = {"koeff1": 1.0, "koeff2": 1.005, "sl_pts": 80.0}
    base_train_reward, base_train_metrics = evaluate_on_pre(pre_train, base_params, baseline_hp)
    base_oos_reward, base_oos_metrics = evaluate_on_pre(pre_oos, base_params, baseline_hp)
    print(
        "Baseline:"
        f" train_avg_trade={base_train_reward:.2f}, train_pf={base_train_metrics['profit_factor']:.3f},"
        f" oos_avg_trade={base_oos_reward:.2f}, oos_pf={base_oos_metrics['profit_factor']:.3f}"
    )

    env = DummyVecEnv([lambda: StrategyContinuousEnv(pre_train, base_params, episode_len=args.episode_len)])
    rollout_steps = max(64, min(512, args.timesteps))
    model = PPO("MlpPolicy", env, verbose=1, device="cpu", n_steps=rollout_steps, seed=args.seed)

    print(f"\nTraining PPO for {args.timesteps} timesteps...")
    t_train = time.time()
    model.learn(total_timesteps=args.timesteps)
    print(f"Training done in {time.time() - t_train:.1f}s")

    candidates = _sample_candidates(
        model,
        seed=args.seed,
        stochastic_samples=args.stochastic_candidates,
        random_samples=args.random_candidates,
    )
    print(f"Candidate hyperparameter sets to evaluate: {len(candidates)}")

    konkop = parse_konkop_oos(args.konkop_report)
    konkop_net = konkop.get("net_profit")
    konkop_pf = konkop.get("profit_factor")
    konkop_mdd_pct = konkop.get("max_dd_pct")

    rows = []
    for i, hp in enumerate(candidates, start=1):
        train_reward, train_m = evaluate_on_pre(pre_train, base_params, hp)
        oos_reward, oos_m = evaluate_on_pre(pre_oos, base_params, hp)
        row = {
            "rank_oos": 0,
            "koeff1": round(hp["koeff1"], 6),
            "koeff2": round(hp["koeff2"], 6),
            "sl_pts": round(hp["sl_pts"], 4),
            "train_avg_trade": float(train_reward),
            "train_trades": int(train_m["num_trades"]),
            "train_total_pnl": float(train_m["total_pnl"]),
            "train_pf": float(train_m["profit_factor"]),
            "train_max_dd_abs": float(train_m["max_drawdown"]),
            "train_max_dd_pct": float(train_m["max_drawdown"] / base_params.capital * 100.0),
            "oos_avg_trade": float(oos_reward),
            "oos_trades": int(oos_m["num_trades"]),
            "oos_total_pnl": float(oos_m["total_pnl"]),
            "oos_pf": float(oos_m["profit_factor"]),
            "oos_max_dd_abs": float(oos_m["max_drawdown"]),
            "oos_max_dd_pct": float(oos_m["max_drawdown"] / base_params.capital * 100.0),
            "delta_oos_pnl_vs_konkop": (
                float(oos_m["total_pnl"] - konkop_net) if konkop_net is not None else np.nan
            ),
            "delta_oos_pf_vs_konkop": (
                float(oos_m["profit_factor"] - konkop_pf) if konkop_pf is not None else np.nan
            ),
            "delta_oos_max_dd_pct_vs_konkop": (
                float((oos_m["max_drawdown"] / base_params.capital * 100.0) - konkop_mdd_pct)
                if konkop_mdd_pct is not None
                else np.nan
            ),
        }
        rows.append(row)
        if i % 20 == 0 or i == len(candidates):
            print(f"Evaluated {i}/{len(candidates)} candidates")

    rows.sort(key=lambda r: (r["oos_total_pnl"], r["oos_avg_trade"]), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank_oos"] = idx

    top_rows = rows[: max(1, args.top_k)]
    best = top_rows[0]

    all_path = os.path.join(args.output_dir, "ppo_long_candidates_all.csv")
    top_path = os.path.join(args.output_dir, "ppo_long_candidates_top.csv")
    summary_path = os.path.join(args.output_dir, "ppo_long_summary.json")
    _save_csv(all_path, rows)
    _save_csv(top_path, top_rows)

    summary = {
        "run_args": vars(args),
        "baseline_hp": baseline_hp,
        "baseline_train": base_train_metrics,
        "baseline_oos": base_oos_metrics,
        "konkop_oos": konkop,
        "best_by_oos": best,
        "n_candidates": len(candidates),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTop candidates by OOS total PnL:")
    print(
        "rank | koeff1  koeff2  sl_pts | train_avg   train_pf | "
        "oos_avg    oos_pf    oos_pnl      oos_dd% | dPnL_vs_konkop"
    )
    for r in top_rows:
        print(
            f"{int(r['rank_oos']):>4} | "
            f"{r['koeff1']:>6.4f}  {r['koeff2']:>6.4f}  {r['sl_pts']:>7.2f} | "
            f"{r['train_avg_trade']:>9.2f}  {r['train_pf']:>8.3f} | "
            f"{r['oos_avg_trade']:>8.2f}  {r['oos_pf']:>8.3f}  {r['oos_total_pnl']:>10,.0f}  "
            f"{r['oos_max_dd_pct']:>7.2f}% | {r['delta_oos_pnl_vs_konkop']:>12,.0f}"
        )

    print("\nKONKOP OOS reference:", konkop)
    print("\nSaved:")
    print(f"  {all_path}")
    print(f"  {top_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
