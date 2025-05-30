# # main.py
import subprocess
import argparse
import sys
import yaml
import multiprocessing
import os
from configs.syspath import (BASE_PATH, FACTOR_CODE_PATH)

FACTOR_CONFIG_PATH = os.path.join(BASE_PATH, 'configs', 'factor.yaml')
def run_scripts(factor_name):
    subprocess.check_call([sys.executable, 'run_factor.py', '--name', factor_name])
    subprocess.check_call([sys.executable, 'run_backtest.py', '--name', factor_name])
    subprocess.check_call([sys.executable, 'run_plot.py', '--name', factor_name])

def run_scripts2(factor_name):
    subprocess.check_call([sys.executable, 'factor', '--name', factor_name])
    subprocess.check_call([sys.executable, 'backtest', '--name', factor_name])
    subprocess.check_call([sys.executable, 'plot', '--name', factor_name])

def tmain(names):
    print("names:"+names)
    factors = names.split(",") 
    processes = []
    for factor in factors:
        p = multiprocessing.Process(target=run_scripts2, args=(factor,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

def main():
    parser = argparse.ArgumentParser(description='Run factor computations, backtest, and plotting for multiple factors.')
    # parser.add_argument('--config', type=str, default=FACTOR_CONFIG_PATH, help='Path to the config YAML file.')
    parser.add_argument('--names', nargs='+', type=str, required=True,help='输入的数组参数，用空格分隔')
    args = parser.parse_args()
    factors = args.names

    processes = []
    for factor in factors:
        print(factor)
        p = multiprocessing.Process(target=run_scripts, args=(factor,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()

