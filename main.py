# # main.py
import subprocess
import argparse
import sys
import yaml
import multiprocessing
import os
from configs.syspath import (BASE_PATH, DATA_PATH, UNIVERSE_PATH, FACTOR_VALUES_PATH,
                                BACKTEST_PATH, IMAGE_PATH, INTERMEDIATE_PATH, STATS_PATH)
FACTOR_CONFIG_PATH = os.path.join(BASE_PATH, 'configs', 'factor.yaml')
def run_scripts(factor_name, config_path):
    subprocess.check_call([sys.executable, 'run_factor.py', '--name', factor_name, '--config', config_path])
    subprocess.check_call([sys.executable, 'run_backtest.py', '--name', factor_name, '--config', config_path])
    subprocess.check_call([sys.executable, 'run_plot.py', '--name', factor_name, '--config', config_path])

def main():
    parser = argparse.ArgumentParser(description='Run factor computations, backtest, and plotting for multiple factors.')
    parser.add_argument('--config', type=str, default=FACTOR_CONFIG_PATH, help='Path to the config YAML file.')
    args = parser.parse_args()

    with open(args.config, 'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
    factors = config.get('factors', [])

    processes = []
    for factor in factors:
        p = multiprocessing.Process(target=run_scripts, args=(factor['name'], args.config))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()

