# # run_backtest.py
import time
import yaml
import argparse
from calc_alpha import AlphaCalc
import os
import pandas as pd
from pathlib import Path
import ast,sys
from configs.syspath import (BASE_PATH, FACTOR_CODE_PATH,SHARED_PATH, FACTOR_VALUES_PATH,LOGS_PATH)

from loguru import logger
from utils.get_params import get_factor_params
from utils.logger_setup import setup_execution_logger
setup_execution_logger(LOGS_PATH)



def run_backtest(params):

    simulator = AlphaCalc(params)

    stats = simulator.report_stats()

    return stats

def main(name):
    setup_execution_logger(LOGS_PATH)
    logger.info("-" * 150)
    logger.info("开始运行 run_backtest")
    logger.info("运行回测")
    start_time = time.time()
    try: 

        factor_params = get_factor_params(name, logger)

        logger.info(f"因子配置参数:\n {factor_params}")

        if factor_params is None:
            raise ValueError(f"Factor {name} not found in the config file.")

        author = factor_params['author']
        factor_values_path_new = os.path.join(FACTOR_VALUES_PATH, f"{name}.parquet")
        if not os.path.exists(factor_values_path_new):
            raise FileNotFoundError(f"Parquet file not found: {factor_values_path_new}")
        indicator_df = pd.read_parquet(factor_values_path_new)

        if indicator_df.empty or not indicator_df.index.is_unique:
            raise ValueError(f"Indicator DataFrame is empty or has non-unique index for factor {name}.")
        

        try:

            start_date = indicator_df.index[0][0] if isinstance(indicator_df.index, pd.MultiIndex) else indicator_df.index[0]
            end_date = indicator_df.index[-1][0] if isinstance(indicator_df.index, pd.MultiIndex) else indicator_df.index[-1]
        except (IndexError, TypeError) as e:
            raise ValueError(f"Error extracting dates from index: {e}")
        
        # 更新 factor_params
        factor_params['start_date'] = str(start_date)  # 转换为字符串
        factor_params['end_date'] = str(end_date)
        factor_params['run_mode'] = 'all'

        performance_stats = run_backtest(factor_params)

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"回测运行时长: {total_time:.2f} seconds")
    except Exception:
        logger.exception("run_backtest 执行失败")
        sys.exit(1)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run backtest for a specific factor.')
    parser.add_argument('--name', type=str, required=True, help='Name of the factor to run backtest.')
    # parser.add_argument('--config', type=str, default=FACTOR_CONFIG_PATH, help='Path to the config YAML file.')
    args = parser.parse_args()
    name = args.name
    main(args.name)