import sys; sys.path.insert(0, '.')
from src.config import Config
from src.data import build_panel_auto

cfg = Config(ticker='SBER', start='2006-01-01', end='2022-12-31',
             data_source='local', timeframe='H1', data_path=r'G:\data2')
cfg.apply_timeframe_defaults()
panel = build_panel_auto(cfg)
p = panel['SBER']
print('Price stats by year:')
for y in range(2006, 2023):
    sub = p[p.index.year == y]
    if len(sub) > 0:
        print(f'  {y}: {len(sub):>6} bars, '
              f'min={sub.min():>12.2f}, max={sub.max():>12.2f}, '
              f'last={sub.iloc[-1]:>12.2f}')
    else:
        print(f'  {y}: no data')

# Find the split boundary
for i in range(1, len(p)):
    ratio = p.iloc[i] / p.iloc[i-1]
    if ratio < 0.01 or ratio > 100:
        print(f'\nSPLIT detected at index {i}: '
              f'{p.index[i-1]} -> {p.index[i]}: '
              f'{p.iloc[i-1]:.2f} -> {p.iloc[i]:.2f} (ratio {ratio:.6f})')
