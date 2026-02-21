# StockPredictionAI Pro (2025 Edition)

**Полноценная система прогнозирования цен акций на основе GAN, реализованная на PyTorch.**  
Основана на идеях [Boris Banushev](https://github.com/borisbanushev/stockpredictionai) — переработана с нуля, расширена Transformer-генератором, квантильной регрессией, автоматическим подбором гиперпараметров и комплексным feature engineering.

---

## Возможности

### Данные и Feature Engineering
- **40+ коррелированных активов** через Yahoo Finance: конкуренты (JPM, BAC, MS, C, WFC...), индексы (S&P 500, DJIA, NASDAQ, Russell 2000, FTSE, Nikkei, Hang Seng, DAX...), валюты (EUR/USD, GBP/USD, USD/JPY...), сырьё (золото, серебро, нефть, газ), волатильность (VIX), облигации (TNX, TLT)
- **13 технических индикаторов**: SMA(7/21), EMA(21), MACD + Signal + Histogram, Bollinger Bands (upper/mid/lower), RSI(14), Momentum(10), Log Momentum(10)
- **Мульти-Фурье** (k=3, 6, 9): разложение тренда на долгосрочный, среднесрочный и краткосрочный компоненты
- **ARIMA(5,1,0)**: in-sample аппроксимация как дополнительный признак
- **Stacked Autoencoder** (GELU): нелинейные латентные признаки
- **Eigen-портфель (PCA)**: линейные латентные признаки
- **FinBERT сентимент** (opt-in): анализ тональности новостей
- **XGBoost Feature Importance**: ранжирование признаков по значимости

### Модели (WGAN-GP)
- **Генераторы**: LSTM (с dropout) или Transformer (TST) — переключение одним флагом
- **Дискриминатор**: 3-слойный Conv1D (32→64→128) с BatchNorm, FC(220→220→1)
- **Multi-task выход**: регрессия ΔP, классификация направления, квантильная регрессия (q10/q50/q90)
- **Gradient Penalty** (WGAN-GP): стабильное обучение

### Обучение и оптимизация
- **CosineAnnealing LR Scheduler** для обоих оптимизаторов (G и D)
- **Early Stopping** по валидационному MAE с восстановлением лучшей модели
- **Optuna**: автоматический подбор гиперпараметров (генератор, lr, batch_size, loss weights и др.)
- **Walk-forward evaluation**: валидация с учётом хронологии
- **StandardScaler**: нормализация fit-on-train-only (без утечки данных)

### Диагностика и визуализация
- **Статистические тесты**: ADF (стационарность), VIF (мультиколлинеарность), Ljung-Box (автокорреляция), Breusch-Pagan (гетероскедастичность)
- **Графики**: прогнозы vs реальность с квантильными интервалами, кривые обучения (G/D loss), технические индикаторы, Фурье-декомпозиция
- **Метрики**: MAE, MAPE, Directional Accuracy, Pinball Loss

### Инженерные решения
- `torch.compile` + AMP (mixed precision) на CUDA
- WandB логирование (опционально)
- 45 автотестов (pytest)

---

## Структура проекта

```
stockpredictionai-pro/
├── src/
│   ├── config.py              # Все параметры (dataclass Config)
│   ├── data.py                # Загрузка данных, индикаторы, панель активов
│   ├── dataset.py             # Train/test split, make_sequences, walk-forward
│   ├── train.py               # Главный пайплайн обучения
│   ├── features/
│   │   ├── fourier.py         # Фурье-аппроксимация (single + multi)
│   │   ├── arima_feat.py      # ARIMA in-sample
│   │   ├── autoencoder.py     # Stacked Autoencoder
│   │   ├── pca_eigen.py       # PCA / Eigen-портфель
│   │   └── sentiment.py       # FinBERT сентимент-анализ
│   ├── models/
│   │   ├── generator.py       # LSTMGenerator + TransformerGenerator
│   │   ├── discriminator.py   # CNNDiscriminator (Conv1D + BatchNorm)
│   │   └── gan.py             # WGAN-GP + LR scheduler
│   └── utils/
│       ├── indicators.py      # Технические индикаторы (SMA, EMA, RSI, MACD, Bollinger, Momentum, Log Momentum)
│       ├── metrics.py         # MAE, MAPE, sMAPE, Direction Accuracy, Pinball Loss
│       ├── stat_checks.py     # Стат. тесты (ADF, VIF, Ljung-Box, Breusch-Pagan)
│       └── visualization.py   # Графики (прогнозы, кривые обучения, индикаторы, Фурье)
├── scripts/
│   ├── feature_importance.py  # XGBoost feature importance
│   ├── optuna_tune.py         # Автоподбор гиперпараметров
│   └── run_all.py             # Запуск полного цикла
├── tests/                     # 45 pytest тестов (unit + integration)
│   ├── test_data.py
│   ├── test_dataset.py
│   ├── test_features.py
│   ├── test_metrics.py
│   ├── test_models.py
│   └── test_pipeline.py
├── outputs/                   # Результаты (CSV, PNG, JSON)
├── requirements.txt
└── README.md
```

---

## Установка

```bash
git clone https://github.com/pavadik/stockpredictionai-pro.git
cd stockpredictionai-pro

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### Зависимости

| Пакет | Назначение |
|-------|-----------|
| `torch >= 2.1` | Нейросети, GAN, AMP |
| `numpy, pandas` | Данные, вычисления |
| `yfinance >= 0.2` | Загрузка котировок |
| `scikit-learn` | StandardScaler, метрики |
| `statsmodels >= 0.14` | ARIMA, стат. тесты |
| `xgboost >= 2.0` | Feature importance |
| `matplotlib >= 3.7` | Визуализация |
| `optuna >= 3.0` | Гиперпараметры |
| `transformers >= 4.40` | FinBERT сентимент |
| `pytest >= 7.0` | Тестирование |
| `wandb >= 0.16` | Логирование (opt.) |

---

## Быстрый старт

### Стандартный запуск (LSTM генератор)

```bash
python -m src.train --ticker GS --start 2010-01-01 --end 2018-12-31 --test_years 2
```

Полный пайплайн: загрузка данных → стат. проверки → feature engineering → обучение GAN с early stopping → оценка → графики и CSV в `outputs/`.

### Transformer (TST) генератор

```bash
python -m src.train --ticker GS --start 2010-01-01 --end 2018-12-31 --generator tst
```

### Walk-forward валидация (5 сплитов)

```bash
python -m src.train --ticker GS --start 2010-01-01 --end 2018-12-31 --walk_forward
```

### Автоподбор гиперпараметров (Optuna)

```bash
python scripts/optuna_tune.py --ticker GS --start 2010-01-01 --end 2018-12-31 --trials 50
```

### Feature Importance (XGBoost)

```bash
python scripts/feature_importance.py --ticker GS --start 2010-01-01 --end 2018-12-31
```

### Тесты

```bash
python -m pytest tests/ -v
```

---

## Конфигурация

Все параметры задаются в `src/config.py` (dataclass `Config`):

| Группа | Параметры | По умолчанию |
|--------|-----------|-------------|
| **Общие** | `ticker`, `start`, `end`, `test_years` | `GS`, `2010-01-01`, `2018-12-31`, `2` |
| **GAN** | `seq_len`, `batch_size`, `lr_g`, `lr_d`, `n_epochs`, `critic_steps`, `hidden_size` | `17`, `64`, `1e-3`, `1e-4`, `20`, `5`, `64` |
| **LSTM** | `num_layers`, `dropout_lstm` | `1`, `0.1` |
| **Transformer** | `generator='tst'`, `d_model`, `nhead`, `num_layers_tst`, `dropout_tst` | `64`, `4`, `2`, `0.1` |
| **Loss weights** | `adv_weight`, `l1_weight`, `cls_weight`, `q_weight` | `1.0`, `0.4`, `0.2`, `0.3` |
| **Квантили** | `quantiles` | `(0.1, 0.5, 0.9)` |
| **Фурье** | `fourier_components` | `(3, 6, 9)` |
| **ARIMA** | `arima_order` | `(5, 1, 0)` |
| **AE / PCA** | `ae_hidden`, `ae_bottleneck`, `ae_epochs`, `pca_components` | `64`, `32`, `10`, `12` |
| **LR Scheduler** | `use_lr_scheduler`, `lr_scheduler_min_factor` | `True`, `0.1` |
| **Early Stopping** | `early_stopping_patience` | `5` |
| **Сентимент** | `use_sentiment` | `False` |

---

## Выходные файлы (`outputs/`)

| Файл | Описание |
|------|----------|
| `test_predictions.csv` | `y_true`, `y_pred`, `y_logit`, `q10`, `q50`, `q90` |
| `test_predictions_split{i}.csv` | Прогнозы по каждому walk-forward сплиту |
| `metrics_walk_forward.csv` | Агрегированные метрики по сплитам |
| `feature_importance_xgb.csv` | Важность признаков (XGBoost) |
| `optuna_best.json` | Лучшие гиперпараметры (Optuna) |
| `pred_vs_real.png` | График прогнозов с квантильными интервалами |
| `training_curves.png` | Кривые обучения (G loss, D loss) |
| `technical_indicators.png` | Дашборд технических индикаторов |
| `fourier_components.png` | Фурье-декомпозиция (k=3, 6, 9) |

---

## Архитектура GAN

```
         ┌─────────────────────────────────────────────┐
         │  Генератор (LSTM или Transformer)           │
         │  Вход: [B, T, F+1]  (features + noise)     │
         │  Выходы:                                    │
         │    y_reg     [B]     — регрессия ΔP          │
         │    y_cls     [B]     — направление (logit)  │
         │    y_q       [B, 3]  — квантили (q10/50/90) │
         └───────────────┬─────────────────────────────┘
                         │
         ┌───────────────▼─────────────────────────────┐
         │  Дискриминатор (3× Conv1D + BatchNorm)      │
         │  Вход: [B, T, F] concat y → [B, T, F+1]    │
         │  Conv1D: 32→64→128, FC: 220→220→1           │
         │  Выход: WGAN score (без sigmoid)             │
         └─────────────────────────────────────────────┘

 Loss_G = adv_weight * (-D(G(x))) + l1_weight * L1 + cls_weight * BCE + q_weight * Pinball
 Loss_D = -(D(real) - D(fake)) + λ * GP
```

---

## Пайплайн обработки данных

```
 Yahoo Finance (40+ тикеров)
       │
       ▼
 Технические индикаторы (13 шт.)
       │
       ▼
 Мульти-Фурье (k=3, 6, 9) + ARIMA
       │
       ▼
 Стат. проверки (ADF, VIF, Ljung-Box, Breusch-Pagan)
       │
       ▼
 Autoencoder → латентные признаки
       │
       ▼
 PCA / Eigen-портфель
       │
       ▼
 StandardScaler (fit on train only)
       │
       ▼
 Sliding window sequences [B, T, F]
       │
       ▼
 WGAN-GP обучение (LR scheduler + early stopping)
       │
       ▼
 Оценка (MAE, MAPE, DirAcc, Pinball) + визуализация
```

---

## Тестирование

Проект покрыт **45 автотестами** (pytest):

| Модуль | Тесты | Что покрывают |
|--------|-------|---------------|
| `test_data` | 4 | Загрузка данных, индикаторы, NaN |
| `test_dataset` | 4 | Train/test split, sequences, walk-forward |
| `test_features` | 7 | Fourier (single + multi), ARIMA, AE, PCA, сентимент, log_momentum |
| `test_metrics` | 10 | MAE, MAPE, sMAPE, DirAcc, Pinball |
| `test_models` | 15 | LSTM, Transformer, Discriminator, BatchNorm, GP, WGAN-GP, LR scheduler |
| `test_pipeline` | 5 | Полный пайплайн (LSTM + TST), walk-forward, CSV-выход |

---

## Лицензия

MIT License.

---

## Благодарности

- [Boris Banushev](https://github.com/borisbanushev/stockpredictionai) — оригинальная идея и notebook
- PyTorch — фреймворк (`torch.compile`, AMP)
- Сообщество Time Series — вдохновение (TST, N-BEATS, N-HiTS)
- Paul Dikaloff — 2025
