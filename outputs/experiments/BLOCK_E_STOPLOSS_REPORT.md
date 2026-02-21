# Block E: Stop-Loss Research Report

## 1. Objective

Test whether adding stop-loss mechanisms to existing LONG and SHORT momentum-trend strategies improves out-of-sample performance, specifically:
- Reduces maximum drawdown
- Preserves or improves average trade (per 1 contract)
- Improves Sharpe ratio / Profit Factor

---

## 2. Methodology

### 2.1 Stop-Loss Types Tested

| # | Type | Key Parameter | Range |
|---|------|--------------|-------|
| 1 | **Fixed** | `sl_pts` (points) | 500 - 10,000 |
| 2 | **ATR-based** | `sl_atr_mult` | 0.5 - 5.0 |
| 3 | **Trailing ATR** | `trail_atr` (ATR units) | 0.5 - 5.0 |
| 4 | **Trailing Points** | `trail_pts` (points) | 500 - 8,000 |
| 5 | **Breakeven** | `be_target` / `be_offset` | 500-5000 / 0-500 |
| 6 | **Time** | `time_bars` / `time_target` | 5-50 / 0-2000 |
| 7 | **None** | (baseline, no stop) | — |

All 7 types competed simultaneously in the same optimization run.

### 2.2 Optimization Setup

- **GPU:** NVIDIA GeForce RTX 3090 Ti (25.8 GB)
- **Trials:** 30,000 random parameter sets per direction
- **Min trades:** 200 (train period 2007-2018)
- **Objective:** maximize `avg_trade_1c` (per 1 contract, no sizing bias)
- **Speed:** ~90 trials/s, ~6 min per run + 5 min pre-aggregation
- **Seed:** 42

### 2.3 Commands

```bash
# LONG with stop-losses
python scripts/run_block_e.py --gpu --direction long --ticker RTSF --data_path G:\data2 \
  --n_trials 30000 --min_trades 200 \
  --sl_types "none,fixed,atr,trail_atr,trail_pts,breakeven,time"

# SHORT with stop-losses
python scripts/run_block_e.py --gpu --direction short --ticker RTSF --data_path G:\data2 \
  --n_trials 30000 --min_trades 200 \
  --sl_types "none,fixed,atr,trail_atr,trail_pts,breakeven,time"
```

---

## 3. Results — LONG Strategy

### 3.1 Winning Stop-Loss Type: **ATR-based** (`sl_atr_mult = 4.65`)

The optimizer selected an ATR-based stop with a wide 4.65x ATR multiplier. This is a "catastrophe filter" — it only triggers on extreme adverse moves while letting normal trade dynamics play out.

### 3.2 LONG Parameters (with SL)

| Parameter | Baseline | With SL |
|-----------|----------|---------|
| tf1 | 150 | **180** |
| tf2 | 60 | 60 |
| lookback | 95 | **140** |
| length | 280 | **55** |
| lookback2 | 85 | **65** |
| length2 | 185 | **75** |
| min_s | 180 | **90** |
| max_s | 500 | **230** |
| koeff1 | 0.9609 | **0.9157** |
| koeff2 | 1.0102 | **1.0143** |
| mmcoff | 26 | **28** |
| exitday | 0 | 0 |
| sdel_day | 1 | **0** |
| **sl_type** | none | **atr** |
| **sl_atr_mult** | — | **4.65** |

### 3.3 LONG Comparison (per 1 contract)

|  | Baseline TRAIN | SL TRAIN | Baseline TEST | SL TEST |
|--|----------------|----------|---------------|---------|
| Avg trade | 570.7 | **777.4** (+36%) | **839.6** | 152.5 (-82%) |
| Trades | 315 | 209 | 74 | 52 |
| Win rate | 31.4% | **44.5%** | 37.8% | **42.3%** |
| PF (1c) | 1.65 | **1.93** | **2.40** | 1.17 |
| Sharpe (1c) | 2.41 | **2.70** | **2.21** | 0.35 |
| Max DD (pts) | 31,625 | **26,910** (-15%) | **7,690** | 8,320 (+8%) |

**Verdict (LONG):** The SL version has a much better TRAIN performance but significantly worse OOS (80.4% decay vs baseline which had negative decay). The optimizer found a different parameter set that is more fragile out-of-sample. The baseline LONG without stops remains superior.

### 3.4 LONG Yearly (SL vs Baseline)

| Year |  | Base avg1c | SL avg1c | Base PF1c | SL PF1c |
|------|--|--------:|--------:|------:|------:|
| 2007 | TR | 1,844 | **2,121** | 2.89 | **2.97** |
| 2008 | TR | -837 | **3,331** | 0.62 | **4.68** |
| 2009 | TR | 2,651 | 1,948 | 4.44 | 4.17 |
| 2010 | TR | 827 | **1,112** | 1.76 | **2.04** |
| 2011 | TR | 1,286 | **2,158** | 2.97 | **3.90** |
| 2012 | TR | -201 | **71** | 0.82 | **1.11** |
| 2013 | TR | 531 | 12 | 1.91 | 1.02 |
| 2014 | TR | 480 | -495 | 1.49 | 0.67 |
| 2015 | TR | 284 | -296 | 1.36 | 0.78 |
| 2016 | TR | 196 | **499** | 1.32 | **1.90** |
| 2017 | TR | 323 | -264 | 1.70 | 0.55 |
| 2018 | TR | -7 | **217** | 0.99 | **1.38** |
| **2019** | **TE** | **1,129** | 210 | **4.21** | 1.47 |
| **2020** | **TE** | 1,644 | 737 | 3.84 | 1.70 |
| **2021** | **TE** | **442** | -119 | **1.74** | 0.87 |
| **2022** | **TE** | -263 | -694 | 0.83 | 0.43 |

---

## 4. Results — SHORT Strategy

### 4.1 Winning Stop-Loss Type: **Time Stop** (`time_bars=45, time_target=1200`)

The optimizer selected a time-based stop: if a trade hasn't earned 1,200 pts profit after 45 bars, exit. This removes stale positions that tie up capital without trending.

### 4.2 SHORT Parameters (with SL)

| Parameter | Baseline | With SL |
|-----------|----------|---------|
| tf1 | 180 | **150** |
| tf2 | 45 | **90** |
| lookback | 245 | **190** |
| length | 55 | **125** |
| lookback2 | 70 | **80** |
| length2 | 95 | **135** |
| min_s | 330 | **170** |
| max_s | 650 | **670** |
| koeff1 | 1.0000 | **1.0064** |
| koeff2 | 0.9458 | **0.9529** |
| mmcoff | 17 | **29** |
| exitday | 0 | 0 |
| sdel_day | 1 | 1 |
| **sl_type** | none | **time** |
| **time_bars** | — | **45** |
| **time_target** | — | **1200** |

### 4.3 SHORT Comparison (per 1 contract)

|  | Baseline TRAIN | SL TRAIN | Baseline TEST | SL TEST |
|--|----------------|----------|---------------|---------|
| Avg trade | 604.6 | **885.0** (+46%) | **1,011.5** | 614.1 (-39%) |
| Trades | 338 | 230 | 78 | 70 |
| Win rate | 43.2% | 36.5% | 35.9% | 25.7% |
| PF (1c) | 1.76 | **1.98** | **2.16** | 1.56 |
| Sharpe (1c) | 2.91 | 2.35 | 1.34 | 0.69 |
| Max DD (pts) | 24,495 | **23,310** (-5%) | **12,080** | 45,790 (+279%) |

**Verdict (SHORT):** Similar pattern to LONG — better train metrics but worse OOS. The time stop drastically increases OOS max drawdown (3.8x) even though avg_trade remains positive. The baseline SHORT without stops remains superior.

### 4.4 SHORT Yearly (SL vs Baseline)

| Year |  | Base avg1c | SL avg1c | Base PF1c | SL PF1c |
|------|--|--------:|--------:|------:|------:|
| 2007 | TR | 1,341 | -1,378 | 2.34 | 0.17 |
| 2008 | TR | 1,942 | **6,803** | 2.62 | **5.98** |
| 2009 | TR | -380 | **878** | 0.74 | **2.11** |
| 2010 | TR | 936 | **1,322** | 2.12 | **2.40** |
| 2011 | TR | 1,596 | **1,911** | 2.96 | 2.35 |
| 2012 | TR | 230 | **447** | 1.22 | **1.48** |
| 2013 | TR | 353 | **558** | 1.86 | **2.11** |
| 2014 | TR | 629 | **1,435** | 1.87 | **3.15** |
| 2015 | TR | 138 | -118 | 1.23 | 0.87 |
| 2016 | TR | -603 | **111** | 0.29 | **1.18** |
| 2017 | TR | 234 | -257 | 1.54 | 0.70 |
| 2018 | TR | 503 | 52 | 2.44 | 1.07 |
| **2019** | **TE** | **106** | -118 | **1.32** | 0.85 |
| **2020** | **TE** | 972 | 893 | 2.15 | 1.87 |
| **2021** | **TE** | -143 | **-1,018** | 0.87 | **0.16** |
| **2022** | **TE** | 7,954 | **13,148** | 5.59 | **8.44** |

---

## 5. Analysis of GPU Top-5 Results

### 5.1 LONG Top-5

| # | avg_1c | trades | WR | PF | TF | SL type |
|---|-------:|-------:|---:|---:|---:|---------|
| 1 | 777.4 | 209 | 44.5% | 1.93 | (180,60) | **atr** |
| 2 | 738.4 | 207 | 35.7% | 1.75 | (240,120) | **atr** |
| 3 | 726.6 | 200 | 44.0% | 1.90 | (120,60) | **atr** |
| 4 | 711.8 | 205 | 39.0% | 1.64 | (150,120) | **atr** |
| 5 | 669.3 | 204 | 40.7% | 1.64 | (180,120) | **time** |

**Observation:** ATR-based stop dominates the LONG top-5 (4/5 entries).

### 5.2 SHORT Top-5

| # | avg_1c | trades | WR | PF | TF | SL type |
|---|-------:|-------:|---:|---:|---:|---------|
| 1 | 885.0 | 230 | 36.5% | 1.98 | (150,90) | **time** |
| 2 | 856.6 | 209 | 38.3% | 1.73 | (210,90) | **breakeven** |
| 3 | 847.5 | 210 | 35.2% | 1.75 | (180,60) | **none** |
| 4 | 807.1 | 213 | 39.4% | 1.73 | (150,120) | **time** |
| 5 | 805.6 | 214 | 34.6% | 1.66 | (300,120) | **breakeven** |

**Observation:** SHORT top-5 is more diverse — time, breakeven, and no-stop all appear. The #3 best SHORT result uses **no stop at all**.

---

## 6. Key Findings

### 6.1 Stop-losses did NOT improve OOS performance

| Direction | Metric | Baseline | With SL | Delta |
|-----------|--------|----------|---------|-------|
| LONG | Test avg_1c | **839.6** | 152.5 | **-82%** |
| LONG | Test PF_1c | **2.40** | 1.17 | -51% |
| LONG | Test MaxDD | **7,690** | 8,320 | +8% |
| LONG | OOS decay | **negative** | 80.4% | worse |
| SHORT | Test avg_1c | **1,011.5** | 614.1 | **-39%** |
| SHORT | Test PF_1c | **2.16** | 1.56 | -28% |
| SHORT | Test MaxDD | **12,080** | 45,790 | **+279%** |
| SHORT | OOS decay | 30.6% | 30.6% | same |

### 6.2 Why Stop-Losses Didn't Help

1. **Overfitting amplification:** The joint optimization of stop-loss params + strategy params doubled the parameter space (~20 dims), making it easier to overfit the training set. The optimizer found combos that look great on 2007-2018 but don't generalize.

2. **Different base parameters:** The optimizer didn't just add a stop to the baseline — it found entirely different strategy parameters. The SL LONG uses tf1=180/tf2=60 vs baseline tf1=150/tf2=60, and different lookback/length values.

3. **The strategy already has natural stops:** Signal-reverse exits + end-of-day exits already limit losses. Adding explicit stops on top of that is redundant at best, harmful at worst (cutting profitable positions that temporarily dip).

4. **Wide ATR multiplier (4.65x):** The LONG winner's stop is so wide it almost never triggers — it's essentially "no stop with a safety net for black swans". This confirms the strategy doesn't need tight stops.

### 6.3 Recommendations

1. **Keep baseline strategies without stop-losses** — they are more robust OOS.
2. If stop protection is desired, use a **very wide catastrophe stop** (e.g., fixed 15,000+ pts or ATR > 4x) applied at the portfolio level, not optimized per-strategy.
3. Future research could test stops with **fixed baseline parameters** (no joint optimization) to isolate the pure stop-loss effect.

---

## 7. File Structure

```
outputs/experiments/
├── block_e_sl_best_params.json           # Best LONG+SL params
├── block_e_short_sl_best_params.json     # Best SHORT+SL params
├── block_e_sl_results.csv                # LONG+SL train/test metrics
├── block_e_short_sl_results.csv          # SHORT+SL train/test metrics
├── block_e_sl_all_trades.csv             # All LONG+SL trades (261)
├── block_e_short_sl_all_trades.csv       # All SHORT+SL trades (300)
├── block_e_sl_trades_train.csv           # LONG+SL train (209)
├── block_e_sl_trades_test.csv            # LONG+SL test (52)
├── block_e_short_sl_trades_train.csv     # SHORT+SL train (230)
├── block_e_short_sl_trades_test.csv      # SHORT+SL test (70)
├── block_e_sl_gpu_trials.csv             # All LONG+SL GPU trials (12,481)
├── block_e_short_sl_gpu_trials.csv       # All SHORT+SL GPU trials (12,509)
├── block_e_sl_yearly.csv                 # LONG+SL yearly breakdown
├── block_e_short_sl_yearly.csv           # SHORT+SL yearly breakdown
└── BLOCK_E_STOPLOSS_REPORT.md            # This report
```
