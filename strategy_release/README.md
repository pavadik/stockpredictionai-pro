# Momentum-Trend Strategy (RTSF) — Release Package

Комбинированная LONG+SHORT стратегия на фьючерсе RTS (RTSF) с momentum-скорингом на двух таймфреймах.

## Структура

```
strategy_release/
├── src/
│   ├── data_local.py                  # Загрузка M1 баров из локальных файлов
│   └── strategy/
│       ├── momentum_trend.py          # StrategyParams, generate_signals, prepare_strategy_data
│       ├── backtester.py              # simulate_trades, simulate_combined_trades, compute_metrics
│       ├── gpu_batch.py               # GPU-ускоренная batch-оценка параметров
│       └── optimizer.py               # Optuna-based parameter optimization
├── scripts/
│   ├── run_block_e.py                 # Оптимизация / оценка одного направления (LONG или SHORT)
│   ├── run_block_e_combined.py        # Объединённый бэктест LONG+SHORT (flip logic, caps, leverage)
│   ├── run_block_e_wf.py             # Walk-Forward валидация
│   └── konkop_analysis.py            # Konkop Xpress-стиль детальный анализ сделок
├── params/
│   ├── block_e_best_params.json       # Лучшие параметры LONG
│   └── block_e_short_best_params.json # Лучшие параметры SHORT
└── README.md
```

## Лучшие параметры

### LONG
| Параметр | Значение |
|----------|---------|
| tf1 | 150 мин |
| tf2 | 60 мин |
| lookback / length | 95 / 280 |
| lookback2 / length2 | 85 / 185 |
| min_s / max_s | 180 / 500 |
| koeff1 / koeff2 | 0.9609 / 1.0102 |
| mmcoff | 26 |
| sdel_day | 1 |

### SHORT
| Параметр | Значение |
|----------|---------|
| tf1 | 180 мин |
| tf2 | 45 мин |
| lookback / length | 245 / 55 |
| lookback2 / length2 | 70 / 95 |
| min_s / max_s | 330 / 650 |
| koeff1 / koeff2 | 1.0000 / 0.9458 |
| mmcoff | 17 |
| sdel_day | 1 |

## Рекомендуемые настройки запуска

### Объединённый бэктест (рекомендуемый)
```bash
python scripts/run_block_e_combined.py \
  --cap_long 100 --cap_short 40 \
  --flip_mode close_loss \
  --leverage 2.0 \
  --train_start 2006-01-01 --test_end 2026-02-28 \
  --commission 0.01
```

### Оптимизация (GPU)
```bash
# LONG
python scripts/run_block_e.py --gpu --direction long --ticker RTSF \
  --data_path G:\data2 --n_trials 30000 --min_trades 300

# SHORT
python scripts/run_block_e.py --gpu --direction short --ticker RTSF \
  --data_path G:\data2 --n_trials 30000 --min_trades 300
```

### Walk-Forward валидация
```bash
python scripts/run_block_e_wf.py --n_trials 20000 --min_trades 150 \
  --ticker RTSF --data_path G:\data2
```

## Ключевые параметры системы

| Параметр | Значение | Описание |
|----------|---------|----------|
| capital | 5,000,000 | Начальный капитал (RUB) |
| commission | 0.01% | Комиссия (round-trip) |
| cap_long | 100 | Макс. контрактов LONG (при leverage 2x) |
| cap_short | 40 | Макс. контрактов SHORT (при leverage 2x) |
| flip_mode | close_loss | Закрывать убыточную позицию при противоположном сигнале, прибыльную — игнорировать |
| leverage | 2.0 | Множитель позиции (1x base caps: L=50, S=20) |
| position sizing | floor(capital / (300 * ATR)) * leverage | Обратно-пропорционально волатильности |

## Результаты (leverage 2x, 2006-2026)

|  | TRAIN (2006-2018) | TEST / OOS (2019-2026) |
|--|-------------------|------------------------|
| Trades | 662 | 346 |
| Net Profit | 14,690,165 | 7,558,114 |
| Annual Return | 11.2% | 13.9% |
| Profit Factor | 1.73 | 1.74 |
| Max Drawdown | 13.7% | 13.0% |
| Ann.Ret / MaxDD | 0.82 | 1.07 |
| Sharpe (1c) | 3.94 | 2.67 |

**Total PnL (20 лет):** 22,248,279 RUB (+445%)

## Зависимости

- Python 3.10+
- numpy, pandas, numba
- PyTorch (CUDA) — для GPU-оптимизации
- optuna — для CPU-оптимизации
- matplotlib — для графиков
