# Архитектура C# Консольного Приложения

## 1. Режимы работы (Modes)

Приложение должно поддерживать запуск через аргументы командной строки.
Пример:
* `RTSF_Strategy_ML.exe --mode backtest --data "G:\data2\RTSF.csv"`
* `RTSF_Strategy_ML.exe --mode live --data "G:\data2\RTSF.csv" --signalr "http://host:port/hub"`

### Режим 1: Backtest (Историческое тестирование)
1. Загрузка минутных баров (M1) из CSV.
2. Прогон симулятора стратегии (LONG и SHORT параллельно).
3. Объединение сделок с логикой `FlipMode.CloseLoss`.
4. Расчет ML фичей для всех потенциальных сделок.
5. Инференс XGBoost модели (предсказание вероятности).
6. Фильтрация сделок по `threshold >= 0.5`.
7. Генерация детального Konkop Xpress отчета (включая разбивку по годам и месяцам).
8. Экспорт итоговых сделок в CSV файл.

### Режим 2: Live (Реальная торговля через SignalR)
1. Инициализация состояния: загрузка "хвоста" исторических M1 баров (например, за последние 3-4 месяца) из папки, чтобы прогреть индикаторы (SMA, D1 ADX и т.д.).
2. Подключение к SignalR хабу для получения стрима новых баров/тиков.
3. OnNewBar (или OnTick):
   - Добавление нового бара в память.
   - Пересчет индикаторов.
   - Проверка условий входа (LONG/SHORT).
   - Если есть сигнал входа -> Расчет 14 ML-фичей прямо в рантайме.
   - Вызов `XGBoost.Predict(features)`.
   - Если `prob >= 0.5` -> Вывод алерта на экран/отправка команды брокеру.
   - Если позиция открыта -> Проверка условий выхода (Signal Reverse / End of Day).

---

## 2. Структура проекта (C#)

```text
RTSF_Strategy_ML/
├── Program.cs                 # Точка входа, парсинг аргументов
├── Core/
│   ├── Config.cs              # Параметры (TF, koeff, lookback)
│   ├── Enums.cs               # Direction, FlipMode, TradeReason
│   └── Models/
│       ├── Bar.cs             # M1 и D1 бары
│       └── Trade.cs           # Структура сделки
├── Data/
│   ├── CsvDataLoader.cs       # Чтение истории
│   └── SignalRClient.cs       # Подписка на live-поток
├── Strategy/
│   ├── Indicators.cs          # SMA, ATR, ADX, TrendScore
│   ├── SignalGenerator.cs     # Поиск точек входа/выхода на основе TF1/TF2
│   └── Backtester.cs          # Симуляция портфеля, Flip logic
├── ML/
│   ├── FeatureBuilder.cs      # Сбор 14 фичей из баров (D1 и M1)
│   └── XGBoostScorer.cs       # Инференс модели (через Microsoft.ML.OnnxRuntime)
└── Reporting/
    └── KonkopAnalyzer.cs      # Расчет метрик, печать таблиц (Годы, Месяцы)
```

## 3. Интеграция ML модели (XGBoost в C#)
Чтобы использовать модель, обученную на Python (XGBoost), в C#, мы экспортируем ее в формат **ONNX** (Open Neural Network Exchange).
В C# будем использовать пакет `Microsoft.ML.OnnxRuntime`.
Это обеспечит миллисекундный инференс в Live-режиме.

## 4. Зависимости (NuGet Packages)
* `CsvHelper` (быстрое чтение истории)
* `Microsoft.AspNetCore.SignalR.Client` (Live режим)
* `Microsoft.ML.OnnxRuntime` (Inference XGBoost)
* `CommandLineParser` (удобный парсинг аргументов запуска)
