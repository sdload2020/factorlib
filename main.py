# # main.py
import subprocess
import argparse
import ast,sys
import yaml
import multiprocessing
import os
import time
from calc_alpha import AlphaCalc
from configs.syspath import (BASE_PATH, WORK_PATH, LOGS_PATH, FACTOR_CODE_PATH)
from loguru import logger
from utils.logger_setup import setup_execution_logger
from datetime import datetime
from pathlib import Path

def write_flag_file(factor_name):
    today = datetime.now().strftime("%Y-%m-%d")
    # today = "2025-06-18"
    flag_dir = os.path.join(WORK_PATH, 'flags', today)
    os.makedirs(flag_dir, exist_ok=True)
    open(os.path.join(flag_dir, f"{factor_name}.flag"), 'w').close()

def get_factor_params(factor_name):
    fileName = factor_name + '.py'
    path = Path(FACTOR_CODE_PATH)
    for file_path in path.rglob('*.py'):  # 使用 rglob 递归匹配所有文件
        if file_path.is_file():
            if(file_path.name == fileName):
                with open(file_path, 'r',encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source)
                arrays = {}

                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == 'config':
                                arrays = ast.literal_eval(node.value)
                                break
    factor_params = next((f for f in arrays['factors'] if f['name'] == factor_name), None)
    return factor_params

def run_scripts(factor_name):
    # logger.info(BASE_PATH)
    # subprocess.check_call([sys.executable, os.path.join(WORK_PATH, 'factorlib/')+'run_factor.py', '--name', factor_name])
    p = subprocess.Popen(
        [sys.executable, os.path.join(WORK_PATH, 'factorlib/') + 'run_factor.py', '--name', factor_name]
    )
    p.wait()
    if p.returncode != 0:
        logger.info("因子运行被跳过，终止后续流程")
        return
    subprocess.check_call([sys.executable, os.path.join(WORK_PATH, 'factorlib/')+'run_backtest.py', '--name', factor_name])
    subprocess.check_call([sys.executable, os.path.join(WORK_PATH, 'factorlib/')+'run_plot.py', '--name', factor_name])
    factor_params = get_factor_params(factor_name)
    if factor_params.get('if_crontab') and factor_params.get('run_mode') == 'online':
        write_flag_file(factor_name)
        logger.success(f"因子 {factor_name} 运行完成, 已写入flag文件...")
    logger.success(f"因子 {factor_name} 运行完成...")
def run_scripts2(factor_name):
    # subprocess.check_call([sys.executable, 'factor', '--name', factor_name])
    p = subprocess.Popen([sys.executable, 'factor', '--name', factor_name])
    p.wait()
    if p.returncode != 0:
        logger.info("因子运行被跳过，终止后续流程")
        return
    subprocess.check_call([sys.executable, 'backtest', '--name', factor_name])
    subprocess.check_call([sys.executable, 'plot', '--name', factor_name])
    factor_params = get_factor_params(factor_name)
    if factor_params.get('if_crontab') and factor_params.get('run_mode') == 'online':
        write_flag_file(factor_name)
        logger.success(f"因子 {factor_name} 运行完成, 已写入flag文件...")
    logger.success(f"因子 {factor_name} 运行完成...")
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
    # parser.add_argument('--name', nargs='+', type=str, default=['vpfs'], help='输入的数组参数，用空格分隔') ##用于debug调试 added 250530
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

