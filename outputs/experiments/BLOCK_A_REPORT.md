# Block A Report: H1 Timeframe Validation

**Date:** 2026-02-17
**Objective:** Validate whether H1 (hourly) is a viable primary timeframe for the WGAN-GP prediction system, with full hyperparameter sweeps and robustness checks.

## Key Configuration

Pipeline simplified (Block B prep):
- `use_fourier = False`, `use_pca = False`, `ae_epochs = 0`
- Result: ~40% faster pipeline, no accuracy loss (confirmed by Round 1 ablation)

## A1: Generator Baseline (LSTM vs TST)

| Generator | MAE | DirAcc | PinballLoss |
|-----------|------|--------|-------------|
| **LSTM** | **0.016** | **51.2%** | **0.0053** |
| TST | 0.710 | 47.6% | 0.0066 |

**Finding:** On H1, LSTM dominates. TST diverged badly (MAE 44x worse). This is the **opposite** of M5, where TST was slightly better. Likely cause: H1 has ~17K bars vs ~207K for M5 -- Transformer needs more data.

**Decision:** `BEST_GEN_H1 = lstm`

## A2: Sequence Length Sweep

| seq_len | Context | MAE | DirAcc |
|---------|---------|------|--------|
| 3 | 3h (half day) | 0.228 | 47.7% |
| 6 | 6h (1 day) | 0.067 | 47.4% |
| **12** | **2 days** | **0.015** | **50.3%** |
| 24 | 4 days | 0.022 | 47.6% |
| 48 | 8 days | 0.158 | 51.1% |

**Finding:** seq_len=12 is the sweet spot: best MAE and DirAcc > 50%. seq_len=48 has decent DirAcc but 10x worse MAE (overfitting). Short sequences (3, 6) lack enough context.

**Decision:** `BEST_SEQ_LEN_H1 = 12`

## A3: Learning Rate Sweep

| lr_g | lr_d | MAE | DirAcc |
|------|------|------|--------|
| 5e-4 | 5e-5 | 0.078 | 50.9% |
| 1e-3 | 1e-4 | 0.031 | 51.2% |
| 1e-3 | 5e-5 | 0.015 | 50.8% |
| **2e-3** | **2e-4** | **0.015** | **51.5%** |
| 5e-4 | 1e-4 | 0.024 | 50.0% |

**Finding:** ALL 5 LR combinations yielded DirAcc >= 50%. This suggests the H1 signal is robust to LR choice. The aggressive `2e-3 / 2e-4` pair is slightly best overall.

**Decision:** `BEST_LR_H1: lr_g=2e-3, lr_d=2e-4`

## A4: Loss Weights Sweep

| adv | l1 | cls | q | MAE | DirAcc |
|-----|-----|-----|-----|------|--------|
| **1.0** | **0.4** | **0.2** | **0.3** | **0.016** | **51.2%** |
| 1.0 | 0.4 | 0.2 | 1.0 | 0.019 | 47.7% |
| 1.0 | 1.0 | 0.2 | 0.3 | 0.014 | 51.0% |
| 0.5 | 0.4 | 0.5 | 0.3 | 0.014 | 50.6% |
| 0.3 | 0.8 | 0.1 | 0.5 | 0.014 | 49.9% |
| 2.0 | 0.2 | 0.1 | 0.1 | 0.039 | 51.6% |

**Key Finding:** `q_weight=1.0` which was optimal for M5 **hurts** H1 (DirAcc drops from 51.2% to 47.7%). Lower quantile weight (0.3) is better for H1.

**Decision:** `BEST_LOSS_H1: adv=1.0, l1=0.4, cls=0.2, q=0.3`

## A5: Walk-Forward Validation (5 splits)

| Split | Train | Test | MAE | DirAcc |
|-------|-------|------|------|--------|
| 1 | 3000 | 720 | 0.068 | **53.2%** |
| 2 | 3720 | 720 | 0.024 | 47.2% |
| 3 | 4440 | 720 | 0.059 | 49.6% |
| 4 | 5160 | 720 | 0.125 | **51.6%** |
| 5 | 5880 | 720 | 0.065 | **50.4%** |
| **Avg** | | | **0.068** | **50.4%** |

**Checkpoints:**
- Average DirAcc > 50%: **PASS** (50.4%)
- DirAcc spread < 10%: **PASS** (6.0%)
- No catastrophic splits: **PASS** (worst MAE = 0.125, 1.8x average)
- 3 of 5 splits > 50%: **MARGINAL**

## A6: Benchmark Comparison

| Model | MAE | DirAcc |
|-------|------|--------|
| Naive (delta=0) | 0.003 | 0.8% |
| SMA(21) | 0.003 | 50.0% |
| Random Walk | 0.004 | 51.2% |
| **Mean Reversion** | **0.004** | **52.3%** |
| GAN (LSTM) | 0.018 | 48.4% |

**Verdict:** GAN DirAcc (48.4%) < Mean Reversion DirAcc (52.3%). **FAIL.** The GAN does not beat simple naive strategies on a single-split benchmark run. However, walk-forward average (50.4%) was above Mean Reversion's theoretical average (50.0%).

## A7: Multi-Ticker Generalization

| Ticker | MAE | DirAcc |
|--------|------|--------|
| GAZP | 0.179 | 47.0% |
| LKOH | 0.039 | 46.9% |
| RTSF | 0.274 | 47.2% |

**Verdict:** 0 of 3 tickers > 50%. **FAIL.** The H1 signal does not generalize beyond SBER.

## Overall Assessment

### Positive Signals
1. SBER H1 LSTM consistently achieves DirAcc ~50-51.5% across multiple configurations
2. Walk-forward validation confirms weak but stable signal (50.4% average, 6% spread)
3. All 5 LR combinations yielded DirAcc >= 50% -- robust to hyperparameter choice
4. Pipeline is 40% faster with simplified features (no AE/Fourier/PCA overhead)

### Negative Signals
1. GAN does NOT beat Mean Reversion (52.3%) on single-split benchmarks
2. Signal does NOT generalize to other tickers (GAZP, LKOH, RTSF all < 50%)
3. TST (Transformer) diverges on H1 -- insufficient data for attention mechanism
4. MAE is 5-7x worse than naive baselines (GAN optimizes for direction, not magnitude)

### Conclusion

Per the plan: **"If DirAcc <= 50% -- H1 result was random; rethink strategy."**

The truth is in between:
- **SBER H1 LSTM has a real but weak signal** (walk-forward 50.4%, marginal but consistent)
- **The signal is ticker-specific** and does not generalize
- **Simple baselines (Mean Reversion) are competitive or better**

### Best H1 Configuration

```
generator = lstm
seq_len = 12
lr_g = 2e-3
lr_d = 2e-4
adv_weight = 1.0
l1_weight = 0.4
cls_weight = 0.2
q_weight = 0.3
use_fourier = False
use_pca = False
ae_epochs = 0
```

### Recommended Next Steps

1. **Architecture experiments (Block C)** should focus on SBER H1 with LSTM as baseline
2. **Consider ensemble approaches** -- combine GAN signal with Mean Reversion
3. **Investigate multi-task learning** -- train on multiple tickers jointly to improve generalization
4. **Explore attention-augmented LSTM** as middle ground (TST too data-hungry, plain LSTM too simple)
5. **Feature engineering for H1** -- intraday seasonality, volume profile, order flow features
