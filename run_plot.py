# # run_plot.py
import time
import yaml
import argparse
import os
from calc_alpha import AlphaCalc
import pandas as pd
from pathlib import Path
import ast,sys
from configs.syspath import (BASE_PATH, SHARED_PATH, IMAGE_PATH, FACTOR_CODE_PATH, FACTOR_VALUES_PATH, LOGS_PATH)
from loguru import logger

from utils.db_connector import fetch_latest_stats_from_db
from loguru import logger
from utils.logger_setup import setup_execution_logger
from utils.get_params import get_factor_params

def run_plot(params):

    simulator = AlphaCalc(params)

    factor_name = params['name']
    author = params['author']
    # 从数据库中获取最新记录
    latest_stats = fetch_latest_stats_from_db(factor_name)
    if not latest_stats:
        logger.info(f"未找到因子 '{factor_name}' 的记录")
        return

    # 调用 report_plot 方法
    simulator.report_plot(
        stats=latest_stats,
        author = author,
        plot=True,
        savefig=True,
        path=IMAGE_PATH,
        full_title=f"{params['name']}_{params['frequency']}"
    )


def main(name):
    setup_execution_logger(LOGS_PATH)
    start_time = time.time()
    logger.info("-" * 150)
    logger.info(" 开始运行 run_plot")
    logger.info("running plot") 
    try:

        factor_params = get_factor_params(name, logger)
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
        factor_params['start_date'] = str(start_date)  
        factor_params['end_date'] = str(end_date)
        factor_params['run_mode'] = 'all'
        
        # logger.info(f"Updated factor_params: {factor_params}")
        run_plot(factor_params)
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"画图运行时长: {total_time:.2f} seconds")
    except Exception:
        logger.exception("run_backtest 执行失败")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run plotting for a specific factor.')
    parser.add_argument('--name', type=str, required=True, help='Name of the factor to plot.')
    args = parser.parse_args()
    name = args.name
    main(args.name)