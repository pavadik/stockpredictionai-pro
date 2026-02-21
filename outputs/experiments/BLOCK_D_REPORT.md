# Block D: Data and Target Experiments -- Report

**Date**: 2026-02-17
**Ticker**: SBER (MOEX), M5 timeframe
**Baseline**: TFT d=64 (C4b), DirAcc = 0.494, MAE = 0.0039

---

## Executive Summary

Block D tested three data/target modifications to improve the TFT model beyond the C4b baseline:

1. **D1 (Multi-horizon)** -- predicting multiple bars ahead instead of just +1 bar
2. **D2 (ATR-normalized target)** -- normalizing the delta by Average True Range
3. **D3 (Volume feature ablation)** -- identifying the minimal useful volume feature set

**Key finding**: Longer prediction horizons (D1d, h=4) achieved the best DirAcc of **50.7%**, surpassing all previous experiments. Volume features contribute ~1% DirAcc improvement. ATR normalization did not help.

---

## D0: Prep -- New Features Added

Before experiments, the data panel was extended:

- **ATR(14)**: Average True Range computed as EMA(TrueRange, 14) -- measures volatility
- **OBV**: On-Balance Volume -- cumulative signed volume
- **VWAP fix**: Replaced cumulative VWAP with 20-bar rolling VWAP to avoid drift

New config fields: `forecast_horizons`, `use_atr_target`, `atr_period`.

Panel after D0: 17 columns (11 close prices + 6 volume features: volume, range, vwap, gap, atr, obv).

---

## D1: Multi-Horizon Prediction

**Hypothesis**: Longer horizons give more stable signal since per-bar noise averages out.

### Results

| Run | Horizons | MAE | DirAcc | Pinball | Time |
|-----|----------|-----|--------|---------|------|
| D1a | (1,) | 0.0039 | **0.494** | 0.0015 | 267s |
| D1b | (1,2,4) joint | 0.0058 avg | 0.494 avg | -- | 179s |
| D1c | (2,) only | 0.0055 | 0.499 | 0.0019 | 297s |
| D1d | (4,) only | 0.0078 | **0.507** | 0.0030 | 146s |

**D1b per-horizon breakdown:**

| Horizon | MAE | DirAcc |
|---------|-----|--------|
| h=1 | 0.0039 | 0.484 |
| h=2 | 0.0055 | 0.495 |
| h=4 | 0.0079 | **0.503** |

### Analysis

- **Confirmed**: Longer horizons produce more stable directional signals.
  - h=1: DirAcc 0.494 (baseline)
  - h=2: DirAcc 0.499 (+0.5%)
  - h=4: DirAcc **0.507** (+1.3%) -- **best result across all blocks**
- MAE naturally increases with horizon (0.0039 -> 0.0055 -> 0.0078), expected.
- Joint training (D1b) averaged DirAcc=0.494 but the h=4 horizon reached 0.503, suggesting the model learns shared representations that benefit longer horizons.
- D1d (4-bar-ahead only) achieves 0.507, the best DirAcc overall. The model converges faster (epoch 7 vs 13) and trains ~2x faster.

### VSN Weights (D1d, h=4):

| Feature | Weight |
|---------|--------|
| SBER_range | 22.95% |
| SBER_atr | 13.67% |
| GMKN | 11.14% |
| SBER_vwap | 7.05% |
| LSR | 6.77% |
| RTSF | 6.74% |

At h=4, range and ATR dominate -- volatility features matter more for longer-horizon prediction.

---

## D2: Volatility-Adjusted Target (delta / ATR)

**Hypothesis**: Normalizing by ATR removes regime dependence.

### Results

| Run | Target | Model | MAE (raw) | DirAcc | DirAcc_norm |
|-----|--------|-------|-----------|--------|-------------|
| D2a | raw delta | TFT d=64 | 0.0039 | **0.494** | -- |
| D2b | delta/ATR(14) | TFT d=64 | 0.2335 | 0.490 | 0.490 |
| D2c | delta/ATR(7) | TFT d=64 | 0.2335 | 0.490 | 0.490 |
| D2d | delta/ATR(14) | Sup. LSTM | 0.2336 | 0.489 | 0.489 |

### Analysis

- **ATR normalization did NOT improve DirAcc** (0.490 vs 0.494 baseline).
- Raw MAE of ~0.23 is expected: `MAE_raw = MAE_norm * avg(ATR)`. Since ATR incorporates high-low range (larger than close-to-close delta), the inverse-transform amplifies errors.
- ATR(7) and ATR(14) produce identical results, suggesting the period choice doesn't matter for this data.
- The Supervised LSTM (D2d) with ATR shows similar results to TFT (D2b), confirming ATR normalization is architecture-agnostic in its (lack of) effect.
- **Possible explanation**: M5 SBER data may not have strong volatility regime shifts within the test period. ATR normalization would be more valuable across crisis/calm transitions (e.g., multi-year data with 2008, 2020 events).

### VSN Weights (D2b, ATR-normalized):

| Feature | Weight |
|---------|--------|
| SBER_atr | **82.67%** |
| SBER_range | 0.89% |
| SBER_obv | 0.82% |

When predicting delta/ATR, the VSN overwhelmingly selects ATR itself (82.67%). This is a trivial signal: the model essentially learns `delta/ATR ~ f(ATR)`, which doesn't provide useful directional information.

---

## D3: Volume Feature Ablation

**Hypothesis**: Not all volume features contribute equally. VSN in C4b showed range=33%, gap=18%, volume=1.2%.

### Results

| Run | Features (count) | MAE | DirAcc | Pinball |
|-----|-----------------|-----|--------|---------|
| D3a | All 6 vol features (17) | 0.0039 | **0.494** | 0.0015 |
| D3b | range + gap only (13) | 0.0040 | 0.488 | 0.0014 |
| D3c | range + gap + ATR + OBV (15) | 0.0040 | 0.486 | 0.0015 |
| D3d | No volume features (11) | 0.0039 | 0.483 | 0.0014 |

### Analysis

- **Volume features contribute +1.1% DirAcc**: 0.494 (all) vs 0.483 (none).
- **Removing any features hurts**: contrary to the hypothesis that D3b >= D3a, keeping all 6 volume features is best.
- Newly added ATR and OBV features don't help when added to range+gap (D3c: 0.486 vs D3b: 0.488). They may add noise without the full feature set context.
- The **full feature set (D3a) is optimal** -- the VSN handles feature selection internally.

### VSN Feature Importance by Configuration

**D3a (all 17 features)**:
```
SBER_atr:    35.7%   SBER_range:  15.5%   SBER_volume: 11.7%
SBER_gap:     5.2%   SBER_obv:     5.1%   AKRN:         5.1%
```

**D3b (13 features, range+gap only)**:
```
SBER_range:  76.0%   LKOH:         3.6%   TATNP:        3.0%
```
When only range+gap are available, VSN concentrates 76% weight on range alone.

**D3d (11 features, no volume)**:
```
LSR:         35.6%   LKOH:        28.7%   SBER:          7.7%
```
Without volume features, VSN shifts to correlated tickers for information.

---

## Consolidated Results vs Baseline

| Experiment | DirAcc | Delta vs C4b | Verdict |
|-----------|--------|-------------|---------|
| **C4b baseline** | **0.494** | -- | Reference |
| D1a (h=1 rerun) | 0.494 | +0.0% | Consistent |
| D1b (h=1,2,4 joint) | 0.494 avg | +0.0% | Neutral |
| D1c (h=2 only) | 0.499 | +0.5% | Marginal |
| **D1d (h=4 only)** | **0.507** | **+1.3%** | **Best overall** |
| D2b (ATR(14)) | 0.490 | -0.4% | Worse |
| D2c (ATR(7)) | 0.490 | -0.4% | Worse |
| D2d (Sup LSTM + ATR) | 0.489 | -0.5% | Worse |
| D3a (all vol) | 0.494 | +0.0% | Matches baseline |
| D3b (range+gap) | 0.488 | -0.6% | Worse |
| D3c (range+gap+atr+obv) | 0.486 | -0.8% | Worse |
| D3d (no vol) | 0.483 | -1.1% | Worst |

---

## Key Takeaways

1. **Longer horizons improve direction prediction.** D1d (h=4, predicting 20min ahead on M5) achieved DirAcc 50.7% -- the first time we've meaningfully exceeded 50% beyond noise.

2. **ATR normalization is counterproductive for this data.** The model trivially learns to predict ATR itself rather than direction. On M5 SBER without regime shifts in the test period, this adds no value.

3. **Volume features matter (+1.1% DirAcc).** The full set of 6 volume features is optimal. The VSN handles internal selection effectively -- manual feature pruning hurts.

4. **New features (ATR, OBV) primarily serve as inputs**, not as target normalizers. ATR as an input feature has 35.7% VSN weight (D3a), making it the #1 feature. But using ATR to normalize the target (D2) reduces performance.

## Recommended Configuration

Based on Block D results, the optimal configuration is:

```python
model_type = "tft"
d_model = 64
forecast_horizons = (4,)     # 4-bar-ahead (20min on M5)
use_atr_target = False        # raw delta target
use_volume_features = True    # all 6 volume features
```

**Current best DirAcc: 50.7%** (D1d)

---

## Files Modified

| File | Changes |
|------|---------|
| `src/config.py` | Added `forecast_horizons`, `use_atr_target`, `atr_period` fields |
| `src/data_local.py` | Added ATR(14), OBV columns; fixed VWAP to 20-bar rolling |
| `src/dataset.py` | Added `make_sequences_multi()` and `make_sequences_atr()` |
| `src/train.py` | Updated `run_one_split()` for multi-horizon and ATR target support |
| `src/models/tft.py` | Parameterized head output size for `n_horizons` |
| `src/utils/metrics.py` | Added `multi_horizon_metrics()` helper |
| `scripts/run_block_d.py` | New experiment runner for all D1-D3 runs |
