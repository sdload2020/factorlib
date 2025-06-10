# run_factor.py
import time
import yaml
import argparse
from calc_alpha import AlphaCalc
import os
from pathlib import Path
import ast,sys
from configs.syspath import (BASE_PATH,FACTOR_CODE_PATH, LOGS_PATH)
from loguru import logger
from utils.logger_setup import setup_execution_logger


def run_factor(params):
    simulator = AlphaCalc(params)
    indicator_dict = simulator.run()
    return indicator_dict

def main(factor_name):
    setup_execution_logger(LOGS_PATH)
    start_time = time.time()
    logger.info("-" * 150)
    logger.info("开始运行 run_factor")
    logger.info("运行因子计算")
    try:
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
        logger.info(factor_params)
        if factor_params is None:
            raise ValueError(f"Factor {factor_name} not found in the config file.")
        indicator_dict_all = run_factor(factor_params)

        logger.info(indicator_dict_all)
        end_time = time.time()
        total_time = end_time - start_time
        # logger.info(f"run_factor.py Total script runtime: {total_time:.2f} seconds")
        logger.info(f"执行因子运行时长: {total_time:.2f} seconds")
    except Exception:
        logger.exception("run_backtest 执行失败")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run factor computation for a specific factor.')
    parser.add_argument('--name', type=str, required=True, help='Name of the factor to compute.')
    # parser.add_argument('--config', type=str, default= factor_config_path, help='Path to the config YAML file.')
    args = parser.parse_args()
    name = args.name
    main(args.name)
    # parser = argparse.ArgumentParser(...)
    # args = parser.parse_args()
    
    # # start_time = time.time()

    # fileName = name + '.py'
    # path = Path(FACTOR_CODE_PATH)
    # for file_path in path.rglob('*.py'):  # 使用 rglob 递归匹配所有文件
    #     if file_path.is_file():
    #         if(file_path.name == fileName):
    #             with open(file_path, 'r',encoding='utf-8') as f:
    #                 source = f.read()
    #             tree = ast.parse(source)
    #             arrays = {}

    #             for node in tree.body:
    #                 if isinstance(node, ast.Assign):
    #                     for target in node.targets:
    #                         if isinstance(target, ast.Name) and target.id == 'config':
    #                             arrays = ast.literal_eval(node.value)
    #                             break


    # factor_params = next((f for f in arrays['factors'] if f['name'] == name), None)
    # logger.info(factor_params)
    # if factor_params is None:
    #     raise ValueError(f"Factor {name} not found in the config file.")
    # start_time = time.time()
    # indicator_dict_all = run_factor(factor_params)
    # logger.info(indicator_dict_all)
    # end_time = time.time()
    # total_time = end_time - start_time
    # logger.info(f"执行因子运行时长: {total_time:.2f} seconds")
