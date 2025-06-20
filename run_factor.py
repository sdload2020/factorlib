# run_factor.py
import time
import yaml
import argparse
from calc_alpha import AlphaCalc
import os
from pathlib import Path
import ast,sys
from configs.syspath import (BASE_PATH,FACTOR_CODE_PATH, LOGS_PATH, WORK_PATH)
from loguru import logger
from utils.logger_setup import setup_execution_logger
from utils.get_params import get_factor_params
from datetime import datetime

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
        factor_params = get_factor_params(factor_name, logger)
        if factor_params.get('if_crontab') and factor_params.get('run_mode') == 'online':
            logger.info(f"因子--{factor_name}: run_mode为online,且if_crontab为True, 判断为定时增量更新, 进入检查前置数据flag文件是否就绪")
            today = datetime.now().strftime("%Y-%m-%d")
            # today = "2025-06-18"  # 测试用
            flag_dir = os.path.join(WORK_PATH, 'flags', today)
            flag_file = os.path.join(flag_dir, f"{factor_name}.flag")
            if os.path.exists(flag_file):
                logger.info(f"因子--{factor_name}: flag文件已存在, 今日已完成执行, 直接退出")
                sys.exit(1)
            logger.info(f"因子--{factor_name}: flag文件不存在, 继续检查前置数据flag文件是否就绪")
            f1 = f"/data-platform/shared/kline/flags/futures/{today}.flag"
            f2 = f"/data-platform/shared/kline/flags/universe/{today}.flag"
            if not (os.path.exists(f1) and os.path.exists(f2)):
                logger.info("前置数据flag文件未就绪,行情数据未准备好, 终止后续流程")
                sys.exit(1)
            logger.info("前置数据flag文件就绪, 继续执行")

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