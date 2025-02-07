# # run_backtest.py
import time
import yaml
import argparse
from xalpha import Xalpha
import os
import pandas as pd
from configs.syspath import (BASE_PATH, DATA_PATH, UNIVERSE_PATH, FACTOR_VALUES_PATH,
                                BACKTEST_PATH, IMAGE_PATH, INTERMEDIATE_PATH, STATS_PATH)
FACTOR_CONFIG_PATH = os.path.join(BASE_PATH, 'configs', 'factor.yaml')


def run_backtest(params):
    print("getting params in run_backtest")
    simulator = Xalpha(params)
    print ("running backtest")
    stats = simulator.report_stats()

    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run backtest for a specific factor.')
    parser.add_argument('--name', type=str, required=True, help='Name of the factor to run backtest.')
    parser.add_argument('--config', type=str, default=FACTOR_CONFIG_PATH, help='Path to the config YAML file.')
    args = parser.parse_args()

    start_time = time.time()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    factor_params = next((f for f in config['factors'] if f['name'] == args.name), None)
    if factor_params is None:
        raise ValueError(f"Factor {args.name} not found in the config file.")
    factor_values_path_new = os.path.join(FACTOR_VALUES_PATH, f"{args.name}.parquet")
    if not os.path.exists(factor_values_path_new):
        raise FileNotFoundError(f"Parquet file not found: {factor_values_path_new}")


    indicator_df = pd.read_parquet(factor_values_path_new)

    if indicator_df.empty or not indicator_df.index.is_unique:
        raise ValueError(f"Indicator DataFrame is empty or has non-unique index for factor {args.name}.")
    

    try:

        start_date = indicator_df.index[0][0] if isinstance(indicator_df.index, pd.MultiIndex) else indicator_df.index[0]
        end_date = indicator_df.index[-1][0] if isinstance(indicator_df.index, pd.MultiIndex) else indicator_df.index[-1]
    except (IndexError, TypeError) as e:
        raise ValueError(f"Error extracting dates from index: {e}")
    
    # 更新 factor_params
    factor_params['start_date'] = str(start_date)  # 转换为字符串
    factor_params['end_date'] = str(end_date)
    factor_params['run_mode'] = 'all'
    
    # print(f"Updated factor_params: {factor_params}")
    performance_stats = run_backtest(factor_params)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run_backtest.py Total script runtime: {total_time:.2f} seconds")

