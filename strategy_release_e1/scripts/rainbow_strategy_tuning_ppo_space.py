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
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gymnasium as gym
    from gymnasium import spaces

    _IS_GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces

    _IS_GYMNASIUM = False

from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import RainbowPolicy
from tianshou.trainer import offpolicy_trainer


# Reuse strategy core from strategy_release_v3.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../strategy_release_v3")))
from src.data_local import load_m1_bars, scan_available_tickers
from src.strategy.backtester import compute_metrics_from_pnl, simulate_trades_fast
from src.strategy.momentum_trend import StrategyParams, apply_params_fast, pre_aggregate


ACTION_LOW = np.array([0.5, 0.5, 10.0], dtype=np.float64)
ACTION_HIGH = np.array([2.0, 2.0, 500.0], dtype=np.float64)
BASELINE_HP = {"koeff1": 1.0, "koeff2": 1.005, "sl_pts": 80.0}


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


def _save_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_axis(low: float, high: float, bins: int, baseline: float) -> np.ndarray:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    vals = np.linspace(low, high, num=bins, dtype=np.float64)
    vals = np.concatenate([vals, np.array([baseline], dtype=np.float64)])
    vals = np.unique(np.round(vals, 6))
    vals.sort()
    return vals


def build_discrete_grid(k1_bins: int, k2_bins: int, sl_bins: int):
    k1_vals = _build_axis(float(ACTION_LOW[0]), float(ACTION_HIGH[0]), k1_bins, BASELINE_HP["koeff1"])
    k2_vals = _build_axis(float(ACTION_LOW[1]), float(ACTION_HIGH[1]), k2_bins, BASELINE_HP["koeff2"])
    sl_vals = _build_axis(float(ACTION_LOW[2]), float(ACTION_HIGH[2]), sl_bins, BASELINE_HP["sl_pts"])

    combos: List[Tuple[float, float, float]] = []
    for k1 in k1_vals:
        for k2 in k2_vals:
            for sl in sl_vals:
                combos.append((float(k1), float(k2), float(sl)))

    meta = {
        "k1_values": [float(x) for x in k1_vals],
        "k2_values": [float(x) for x in k2_vals],
        "sl_values": [float(x) for x in sl_vals],
    }
    return combos, meta


def nearest_action_idx(combos: List[Tuple[float, float, float]], hp: Dict[str, float]) -> int:
    arr = np.asarray(combos, dtype=np.float64)
    tgt = np.array([hp["koeff1"], hp["koeff2"], hp["sl_pts"]], dtype=np.float64)
    scale = ACTION_HIGH - ACTION_LOW
    d2 = np.sum(((arr - tgt) / scale) ** 2, axis=1)
    return int(np.argmin(d2))


class StrategyEvalContext:
    def __init__(
        self,
        data_path: str,
        ticker: str,
        train_start: str,
        train_end: str,
        oos_start: str,
        oos_end: str,
        combos: List[Tuple[float, float, float]],
    ):
        self.combos = combos

        available = scan_available_tickers(data_path, train_start, oos_end)
        if ticker not in available:
            raise RuntimeError(f"Ticker {ticker} not found in {data_path} for {train_start}..{oos_end}")

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
        self.df_train = df_all.loc[df_all.index.values <= train_end_dt].copy()
        self.df_oos = df_all.loc[
            (df_all.index.values >= oos_start_dt) & (df_all.index.values <= oos_end_dt)
        ].copy()
        if self.df_train.empty or self.df_oos.empty:
            raise RuntimeError("Train or OOS slice is empty. Check date ranges.")
        print(f"Train bars: {len(self.df_train):,}; OOS bars: {len(self.df_oos):,}")

        self.base_params = StrategyParams(direction="long", tf1_minutes=210, tf2_minutes=90)
        print("Pre-aggregating once (same TF as PPO: 210/90)...")
        t1 = time.time()
        self.pre_train = pre_aggregate(self.df_train, self.base_params)
        self.pre_oos = pre_aggregate(self.df_oos, self.base_params)
        print(f"Pre-aggregation done in {time.time() - t1:.1f}s")

        self.train_eval_cache: Dict[int, Tuple[float, Dict[str, float]]] = {}
        self.oos_eval_cache: Dict[int, Tuple[float, Dict[str, float]]] = {}

    def evaluate_idx(self, action_idx: int, split: str = "train") -> Tuple[float, Dict[str, float]]:
        cache = self.train_eval_cache if split == "train" else self.oos_eval_cache
        if action_idx in cache:
            return cache[action_idx]

        k1_v, k2_v, sl_v = self.combos[action_idx]
        params = StrategyParams(**asdict(self.base_params))
        params.koeff1 = k1_v
        params.koeff2 = k2_v
        params.sl_type = "fixed"
        params.sl_pts = sl_v

        pre_data = self.pre_train if split == "train" else self.pre_oos
        df_sig = apply_params_fast(pre_data, params)
        pnls = simulate_trades_fast(df_sig, params)
        metrics = compute_metrics_from_pnl(pnls)
        if metrics["num_trades"] < 10:
            out = (-100.0, metrics)
        else:
            out = (float(metrics["avg_trade"]), metrics)
        cache[action_idx] = out
        return out


class StrategyDiscreteEnv(gym.Env):
    def __init__(self, ctx: StrategyEvalContext, action_count: int, episode_len: int = 8):
        super().__init__()
        self.ctx = ctx
        self.max_step = episode_len
        self.cur_step = 0
        self.action_space = spaces.Discrete(action_count)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if _IS_GYMNASIUM:
            super().reset(seed=seed)
        self.cur_step = 0
        obs = np.array([0.0], dtype=np.float32)
        if _IS_GYMNASIUM:
            return obs, {}
        return obs

    def step(self, action):
        action_idx = int(action)
        reward, _ = self.ctx.evaluate_idx(action_idx, split="train")
        self.cur_step += 1
        terminated = self.cur_step >= self.max_step
        truncated = False
        obs = np.array([0.0], dtype=np.float32)
        if _IS_GYMNASIUM:
            return obs, reward, terminated, truncated, {}
        return obs, reward, terminated, {}


class RainbowCategoricalNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, num_atoms: int):
        super().__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        hidden = 64
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, action_dim * num_atoms)

    def forward(self, obs, state=None, info=None):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        elif not torch.is_tensor(obs):
            obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        obs = obs.float()
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        x = self.feature(obs)
        logits = self.head(x).view(-1, self.action_dim, self.num_atoms)
        probs = torch.softmax(logits, dim=2)
        return probs, state


def _policy_q_values(policy: RainbowPolicy) -> np.ndarray:
    obs_batch = Batch(obs=np.array([[0.0]], dtype=np.float32), info={})
    out = policy(obs_batch)
    logits = out.logits
    support = policy.support.reshape(1, 1, -1)
    q = (logits * support).sum(dim=2)
    return q.detach().cpu().numpy().reshape(-1)


def main():
    ap = argparse.ArgumentParser(description="Rainbow PPO-like grid optimization + OOS vs konkop")
    ap.add_argument("--data_path", default=r"G:\data2")
    ap.add_argument("--ticker", default="RTSF")
    ap.add_argument("--train_start", default="2007-01-01")
    ap.add_argument("--train_end", default="2018-12-31")
    ap.add_argument("--oos_start", default="2019-01-01")
    ap.add_argument("--oos_end", default="2022-12-31")
    ap.add_argument("--k1_bins", type=int, default=7)
    ap.add_argument("--k2_bins", type=int, default=7)
    ap.add_argument("--sl_bins", type=int, default=9)
    ap.add_argument("--max_epoch", type=int, default=10)
    ap.add_argument("--step_per_epoch", type=int, default=500)
    ap.add_argument("--step_per_collect", type=int, default=32)
    ap.add_argument("--episode_len", type=int, default=8)
    ap.add_argument("--episode_per_test", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--buffer_size", type=int, default=10000)
    ap.add_argument("--update_per_step", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--estimation_step", type=int, default=3)
    ap.add_argument("--target_update_freq", type=int, default=160)
    ap.add_argument("--num_atoms", type=int, default=51)
    ap.add_argument("--v_min", type=float, default=-5000.0)
    ap.add_argument("--v_max", type=float, default=5000.0)
    ap.add_argument("--train_eps", type=float, default=0.25)
    ap.add_argument("--test_eps", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_k", type=int, default=12)
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
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    combos, grid_meta = build_discrete_grid(args.k1_bins, args.k2_bins, args.sl_bins)
    print("=" * 90)
    print("RAINBOW PPO-LIKE OPTIMIZATION (koeff1/koeff2/sl_pts) + OOS VS KONKOP")
    print("=" * 90)
    print(
        f"Action space size: {len(combos)} "
        f"(k1={len(grid_meta['k1_values'])}, k2={len(grid_meta['k2_values'])}, sl={len(grid_meta['sl_values'])})"
    )

    ctx = StrategyEvalContext(
        data_path=args.data_path,
        ticker=args.ticker,
        train_start=args.train_start,
        train_end=args.train_end,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
        combos=combos,
    )

    baseline_idx = nearest_action_idx(combos, BASELINE_HP)
    base_train_reward, base_train_metrics = ctx.evaluate_idx(baseline_idx, split="train")
    base_oos_reward, base_oos_metrics = ctx.evaluate_idx(baseline_idx, split="oos")
    print(
        f"Baseline action idx={baseline_idx}, hp={combos[baseline_idx]} -> "
        f"train_avg_trade={base_train_reward:.2f}, train_pf={base_train_metrics['profit_factor']:.3f}, "
        f"oos_avg_trade={base_oos_reward:.2f}, oos_pf={base_oos_metrics['profit_factor']:.3f}"
    )

    env = StrategyDiscreteEnv(ctx, action_count=len(combos), episode_len=args.episode_len)
    train_envs = DummyVectorEnv(
        [lambda: StrategyDiscreteEnv(ctx, action_count=len(combos), episode_len=args.episode_len) for _ in range(4)]
    )
    test_envs = DummyVectorEnv(
        [lambda: StrategyDiscreteEnv(ctx, action_count=len(combos), episode_len=args.episode_len) for _ in range(2)]
    )

    net = RainbowCategoricalNet(obs_dim=1, action_dim=env.action_space.n, num_atoms=args.num_atoms)
    optim_ = optim.Adam(net.parameters(), lr=args.lr)
    policy = RainbowPolicy(
        model=net,
        optim=optim_,
        discount_factor=args.discount,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        estimation_step=args.estimation_step,
        target_update_freq=args.target_update_freq,
    )

    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(total_size=args.buffer_size, buffer_num=len(train_envs)),
    )
    test_collector = Collector(policy, test_envs)

    policy.set_eps(args.train_eps)
    train_collector.collect(n_step=max(256, args.step_per_collect * 4))

    def train_fn(epoch, env_step):
        frac = max(0.0, 1.0 - (epoch - 1) / max(1, args.max_epoch - 1))
        eps = args.test_eps + (args.train_eps - args.test_eps) * frac
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.test_eps)

    print(
        f"Training Rainbow for ~{args.max_epoch * args.step_per_epoch} steps "
        f"({args.max_epoch} x {args.step_per_epoch})..."
    )
    t_train = time.time()
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.max_epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
    )
    print(f"Training done in {time.time() - t_train:.1f}s")
    print("Trainer result:", result)

    q_values = _policy_q_values(policy)
    policy_best_action = int(np.argmax(q_values))
    print(f"Policy best action index: {policy_best_action}, hp={combos[policy_best_action]}")

    konkop = parse_konkop_oos(args.konkop_report)
    konkop_net = konkop.get("net_profit")
    konkop_pf = konkop.get("profit_factor")
    konkop_mdd_pct = konkop.get("max_dd_pct")

    rows: List[Dict[str, float]] = []
    for idx, (k1_v, k2_v, sl_v) in enumerate(combos):
        train_reward, train_m = ctx.evaluate_idx(idx, split="train")
        oos_reward, oos_m = ctx.evaluate_idx(idx, split="oos")
        row = {
            "rank_oos": 0,
            "action_idx": idx,
            "koeff1": k1_v,
            "koeff2": k2_v,
            "sl_pts": sl_v,
            "policy_q": float(q_values[idx]),
            "is_policy_best": int(idx == policy_best_action),
            "train_avg_trade": float(train_reward),
            "train_trades": int(train_m["num_trades"]),
            "train_total_pnl": float(train_m["total_pnl"]),
            "train_pf": float(train_m["profit_factor"]),
            "train_max_dd_abs": float(train_m["max_drawdown"]),
            "train_max_dd_pct": float(train_m["max_drawdown"] / ctx.base_params.capital * 100.0),
            "oos_avg_trade": float(oos_reward),
            "oos_trades": int(oos_m["num_trades"]),
            "oos_total_pnl": float(oos_m["total_pnl"]),
            "oos_pf": float(oos_m["profit_factor"]),
            "oos_max_dd_abs": float(oos_m["max_drawdown"]),
            "oos_max_dd_pct": float(oos_m["max_drawdown"] / ctx.base_params.capital * 100.0),
            "delta_oos_pnl_vs_konkop": (
                float(oos_m["total_pnl"] - konkop_net) if konkop_net is not None else np.nan
            ),
            "delta_oos_pf_vs_konkop": (
                float(oos_m["profit_factor"] - konkop_pf) if konkop_pf is not None else np.nan
            ),
            "delta_oos_max_dd_pct_vs_konkop": (
                float((oos_m["max_drawdown"] / ctx.base_params.capital * 100.0) - konkop_mdd_pct)
                if konkop_mdd_pct is not None
                else np.nan
            ),
        }
        rows.append(row)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(combos):
            print(f"Evaluated {idx + 1}/{len(combos)} action combos")

    rows.sort(key=lambda r: (r["oos_total_pnl"], r["oos_avg_trade"]), reverse=True)
    for rank, r in enumerate(rows, start=1):
        r["rank_oos"] = rank

    top_rows = rows[: max(1, args.top_k)]
    best = top_rows[0]

    all_path = os.path.join(args.output_dir, "rainbow_ppo_space_candidates_all.csv")
    top_path = os.path.join(args.output_dir, "rainbow_ppo_space_candidates_top.csv")
    summary_path = os.path.join(args.output_dir, "rainbow_ppo_space_summary.json")
    _save_csv(all_path, rows)
    _save_csv(top_path, top_rows)

    summary = {
        "run_args": vars(args),
        "action_grid": grid_meta,
        "baseline_hp": BASELINE_HP,
        "baseline_action": {
            "idx": baseline_idx,
            "hp": combos[baseline_idx],
            "train": base_train_metrics,
            "oos": base_oos_metrics,
        },
        "trainer_result": result,
        "policy_best_action": {
            "idx": policy_best_action,
            "hp": combos[policy_best_action],
            "policy_q": float(q_values[policy_best_action]),
        },
        "konkop_oos": konkop,
        "best_by_oos": best,
        "n_candidates": len(combos),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTop candidates by OOS total PnL:")
    print(
        "rank | idx | koeff1 koeff2 sl_pts | policy_q | train_avg   train_pf | "
        "oos_avg    oos_pf    oos_pnl      oos_dd% | dPnL_vs_konkop"
    )
    for r in top_rows:
        print(
            f"{int(r['rank_oos']):>4} | {int(r['action_idx']):>3} | "
            f"{r['koeff1']:>6.4f} {r['koeff2']:>6.4f} {r['sl_pts']:>7.2f} | "
            f"{r['policy_q']:>8.2f} | {r['train_avg_trade']:>9.2f}  {r['train_pf']:>8.3f} | "
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
