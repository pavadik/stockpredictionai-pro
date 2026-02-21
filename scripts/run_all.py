import argparse, os, subprocess, sys


def run(cmd):
    print('>>>', ' '.join(cmd))
    res = subprocess.run(cmd, check=True)
    return res.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='GS')
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default='2018-12-31')
    parser.add_argument('--test_years', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_source', default='yfinance', choices=['yfinance', 'local'])
    parser.add_argument('--timeframe', default='D1',
                        help='M1, M3, M5, M7, M14, M30, H1, H4, D1, or tick')
    parser.add_argument('--data_path', default='')
    parser.add_argument('--raw_source', default='m1', choices=['m1', 'ticks'])
    args = parser.parse_args()

    common = ['--ticker', args.ticker, '--start', args.start,
              '--end', args.end, '--seed', str(args.seed),
              '--data_source', args.data_source,
              '--timeframe', args.timeframe,
              '--raw_source', args.raw_source]
    if args.data_path:
        common += ['--data_path', args.data_path]

    run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

    # 1. Train GAN (LSTM)
    run([sys.executable, '-m', 'src.train'] + common +
        ['--test_years', str(args.test_years)])

    # 2. Feature importance
    run([sys.executable, 'scripts/feature_importance.py',
         '--ticker', args.ticker, '--start', args.start, '--end', args.end,
         '--data_source', args.data_source, '--timeframe', args.timeframe,
         '--raw_source', args.raw_source]
        + (['--data_path', args.data_path] if args.data_path else []))

    # 3. Baseline benchmarks
    run([sys.executable, 'scripts/baselines.py'])


if __name__ == '__main__':
    main()
