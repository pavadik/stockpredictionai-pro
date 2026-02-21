# Block E: Momentum-Trend Strategy (RTSF)

## Best Parameters (Refined Search)

| Parameter | Value | Description |
|-----------|-------|-------------|
| tf1_minutes | 150 | Trend timeframe (minutes per bar) |
| tf2_minutes | 60 | Trading timeframe (minutes per bar) |
| lookback | 95 | Momentum lookback on TF1 |
| length | 280 | Rolling window for TSA/SMA on TF1 |
| lookback2 | 85 | Momentum lookback on TF2 |
| length2 | 185 | Rolling window for TSA2/SMA2 on TF2 |
| min_s | 180 | Time window start (min from day open) |
| max_s | 500 | Time window end (min from day open) |
| koeff1 | 0.9609 | TF1 threshold multiplier |
| koeff2 | 1.0102 | TF2 threshold multiplier |
| mmcoff | 26 | ATR period for position sizing |
| exitday | 0 | No forced end-of-day exit |
| sdel_day | 1 | Max 1 entry per day |
| exit_week | 0 | No forced end-of-week exit |
| capital | 5,000,000 | Base capital (RUB) |
| point_value_mult | 300 | Volatility divisor |

Position sizing: `contracts = floor(capital / (300 * ATR(mmcoff)))`

Typical range: 5-65 contracts (mean ~24 train, ~28 test)

## Results

### Per 1 contract (signal quality)

|  | TRAIN (2007-2018) | TEST (2019-2022) |
|--|-------------------|------------------|
| Avg trade | 570.7 pts | **839.6 pts** |
| Total PnL | 179,775 pts | 62,130 pts |
| Num trades | 315 | 74 |
| Win rate | 31.43% | 37.84% |
| Profit factor | 1.65 | **2.40** |
| Sharpe | 2.15 | **2.23** |
| Max drawdown | 31,625 pts | 7,690 pts |

### With position sizing (capital=5M)

|  | TRAIN (2007-2018) | TEST (2019-2022) |
|--|-------------------|------------------|
| Avg trade | 13,417 | **20,044** |
| Total PnL | 4,226,200 | 1,483,250 |
| Profit factor | 1.77 | 2.26 |
| Sharpe | 2.29 | 2.07 |
| Max drawdown | 655,830 | 177,780 |

**OOS decay (1c): -47.1% (test better than train per-trade)**

## Walk-Forward Validation (5 folds)

| Fold | Train | Test | TR avg | TE avg | TR PF | TE PF | TE Sharpe | TE trades |
|------|-------|------|--------|--------|-------|-------|-----------|-----------|
| 1 | 2007-2012 | 2013-2014 | 968.8 | **+216.3** | 2.48 | 1.36 | 0.59 | 35 |
| 2 | 2009-2014 | 2015-2016 | 683.7 | -55.6 | 1.67 | 0.94 | -0.19 | 72 |
| 3 | 2011-2016 | 2017-2018 | 460.7 | -104.2 | 1.69 | 0.79 | -0.64 | 67 |
| 4 | 2013-2018 | 2019-2020 | 398.7 | **+343.5** | 1.51 | 1.30 | 0.67 | 55 |
| 5 | 2015-2020 | 2021-2022 | 618.9 | **+173.7** | 1.96 | 1.16 | 0.27 | 30 |

- **Mean OOS avg_trade**: 114.7 pts
- **Mean OOS PF**: 1.11
- **Mean OOS Sharpe**: 0.14
- **OOS consistency**: 3/5 folds profitable
- **Verdict**: NOT ROBUST (порог: 4/5 фолдов + mean PF > 1.3)

Стратегия прибыльна в трендовых режимах (2007-2014, 2019-2022), но теряет в боковиках (2015-2018).

## Command Lines

### Broad search (initial exploration)
```bash
python scripts/run_block_e.py --gpu --ticker RTSF --data_path G:\data2 --n_trials 30000 --min_trades 300
```

### Refined search (fine-tuning around best params)
```bash
python scripts/run_block_e.py --gpu --refine --ticker RTSF --data_path G:\data2 --n_trials 20000 --min_trades 300
```

### Walk-forward validation
```bash
python scripts/run_block_e_wf.py --n_trials 20000 --min_trades 150 --ticker RTSF --data_path G:\data2
```

### Evaluate specific params (no optimisation)
```bash
python scripts/run_block_e.py --params_json outputs/experiments/block_e_best_params.json --ticker RTSF --data_path G:\data2
```

## Output Files

| File | Description |
|------|-------------|
| `block_e_best_params.json` | Best parameters (JSON) |
| `block_e_results.csv` | Train/test summary metrics |
| `block_e_all_trades.csv` | All 389 trades (train+test), with period label |
| `block_e_trades_train.csv` | 315 trades on train set |
| `block_e_trades_test.csv` | 74 trades on test set |
| `block_e_gpu_trials.csv` | All broad search trials |
| `block_e_gpu_refined_trials.csv` | All refined search trials |
| `block_e_wf_summary.csv` | Walk-forward fold-by-fold results |

### Trades CSV columns

| Column | Description |
|--------|-------------|
| period | train / test (only in all_trades) |
| entry_time | Entry datetime |
| exit_time | Exit datetime |
| entry_price | Entry price (RTS points) |
| exit_price | Exit price (RTS points) |
| contracts | Number of contracts (floor(capital / (300 * ATR))) |
| pnl | PnL with position sizing (pnl_1c * contracts) |
| pnl_1c | PnL per 1 contract (exit_price - entry_price) |
| exit_reason | Why the trade was closed (signal_reverse) |
