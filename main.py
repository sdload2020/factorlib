# # main.py
import subprocess
import argparse
import sys
import yaml
import multiprocessing
import os
import time
from calc_alpha import AlphaCalc
from configs.syspath import (BASE_PATH, WORK_PATH, LOGS_PATH)
from loguru import logger
from utils.logger_setup import setup_execution_logger


def run_scripts(factor_name):
    # logger.info(BASE_PATH)
    subprocess.check_call([sys.executable, os.path.join(WORK_PATH, 'factorlib/')+'run_factor.py', '--name', factor_name])
    subprocess.check_call([sys.executable, os.path.join(WORK_PATH, 'factorlib/')+'run_backtest.py', '--name', factor_name])
    subprocess.check_call([sys.executable, os.path.join(WORK_PATH, 'factorlib/')+'run_plot.py', '--name', factor_name])

def run_scripts2(factor_name):
    subprocess.check_call([sys.executable, 'factor', '--name', factor_name])
    subprocess.check_call([sys.executable, 'backtest', '--name', factor_name])
    subprocess.check_call([sys.executable, 'plot', '--name', factor_name])

def tmain(names):
    # logger.info("names:"+names)
    factors = names.split(",") 
    processes = []
    for factor in factors:
        p = multiprocessing.Process(target=run_scripts2, args=(factor,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

def main():
    setup_execution_logger(LOGS_PATH)
    parser = argparse.ArgumentParser(description='Run factor computations, backtest, and plotting for multiple factors.')
    # parser.add_argument('--config', type=str, default=FACTOR_CONFIG_PATH, help='Path to the config YAML file.')
    # parser.add_argument('--names', nargs='+', type=str, default=['xy3'],help='输入的数组参数，用空格分隔') ##用于debug调试 added 250530
    parser.add_argument('--name', nargs='+', type=str, required=True,help='输入的数组参数，用空格分隔')
    args = parser.parse_args()
    factors = args.name

    processes = []
    for factor in factors:
        logger.info("-" * 150)
        logger.info(f"主模块启动, 准备运行因子: {factor}")
        p = multiprocessing.Process(target=run_scripts, args=(factor,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()

