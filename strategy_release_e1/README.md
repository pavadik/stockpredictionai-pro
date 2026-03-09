# Strategy Release E1: Reinforcement Learning Hyperparameter Tuning

Эта директория содержит реализацию подбора гиперпараметров торговой стратегии (и/или ML-модели) с помощью алгоритмов обучения с подкреплением (Reinforcement Learning), таких как **PPO** и **Rainbow**, базируясь на предыдущих итерациях стратегии (`strategy_release_v3`).

## Требования

Для работы скриптов необходимо установить следующие библиотеки:
```bash
pip install gym stable-baselines3 tianshou torch numpy pandas
```

*Обратите внимание: скрипты зависят от модулей предыдущей итерации стратегии (в частности `strategy_release_v3/src/strategy`).*

## Структура

1. `scripts/ppo_strategy_tuning.py`
   - **Метод**: PPO (Proximal Policy Optimization) через `stable-baselines3`.
   - **Тип действия**: Непрерывное (Continuous).
   - **Окружение**: Агент подбирает коэффициенты `koeff1`, `koeff2` и величину стоп-лосса `sl_pts`.
   - **Награда**: Средняя прибыль со сделки (`avg_trade`) на отложенной (или валидационной) выборке. Данные пред-агрегируются для ускорения обучения.

2. `scripts/rainbow_strategy_tuning.py`
   - **Метод**: Rainbow (DQN со всеми улучшениями) через `tianshou`.
   - **Тип действия**: Дискретное (Discrete).
   - **Окружение**: Агент выбирает комбинацию гиперпараметров из заданного набора (`tf1`, `tf2`, `lookback`).
   - **Награда**: Также средняя прибыль со сделки (`avg_trade`).

3. `scripts/rainbow_strategy_tuning_ppo_space.py`
   - **Метод**: Rainbow (DQN со всеми улучшениями) через `tianshou`.
   - **Тип действия**: Дискретное (Discrete).
   - **Окружение**: Агент выбирает комбинацию параметров из пространства, эквивалентного PPO (`koeff1`, `koeff2`, `sl_pts`) на тех же данных и с той же reward-функцией.
   - **Назначение**: прямое сравнение `PPO vs Rainbow` в формате "один в один".

## Как запустить

Перейдите в корень проекта (там, где лежат папки `strategy_release_e1` и `strategy_release_v3`) и запустите скрипты:

```bash
# Запуск оптимизации через PPO (непрерывные параметры)
python GIT/strategy_release_e1/scripts/ppo_strategy_tuning.py

# Запуск оптимизации через Rainbow (дискретные параметры)
python GIT/strategy_release_e1/scripts/rainbow_strategy_tuning.py

# Rainbow с PPO-like action space (koeff1/koeff2/sl_pts)
python GIT/strategy_release_e1/scripts/rainbow_strategy_tuning_ppo_space.py
```

## Как это работает с реальной логикой GAN / Стратегии

В скриптах реализованы функции `evaluate_strategy(hp)` и `evaluate_strategy_discrete(hp_idx)`. 
Если вам нужно будет переключить это на подбор параметров **GAN** (вместо параметров торговой системы), вы можете заменить вызов `simulate_trades_fast(...)` на тренировку вашей GAN и возвращать в качестве reward значение `-(validation_loss)`. 
Текущая реализация тесно интегрирована с вашим торговым бэктестером из `strategy_release_v3`, что позволяет подбирать `koeff1`, `koeff2`, таймфреймы и размер стоп-лосса, оптимизируя итоговый `PnL` или `Sharpe`.
