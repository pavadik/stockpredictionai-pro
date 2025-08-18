import argparse, os, subprocess, sys

def run(cmd):
    print('>>>', ' '.join(cmd)); 
    res = subprocess.run(cmd, check=True)
    return res.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='GS')
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default='2018-12-31')
    parser.add_argument('--test_years', type=int, default=2)
    args = parser.parse_args()

    run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    run([sys.executable, '-m', 'src.train', '--ticker', args.ticker, '--start', args.start, '--end', args.end, '--test_years', str(args.test_years)])
    run([sys.executable, 'scripts/feature_importance.py', '--ticker', args.ticker, '--start', args.start, '--end', args.end])

if __name__ == '__main__':
    main()
