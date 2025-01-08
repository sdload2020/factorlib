import os

BASE_PATH = "/home/yangzhilin/backtest/backtest_light/code/" # 定义项目根目录 只需要修改这个路径，其他的不用改
RAWDATA_PATH = "/data-platform/crypto/output_parquet/"
DATA_PATH = os.path.join(RAWDATA_PATH, "pv_")
UNIVERSE_PATH = os.path.join(RAWDATA_PATH, "pv_universe.parquet")
FACTOR_VALUES_PATH = os.path.join(BASE_PATH, "factor_values")
BACKTEST_PATH = os.path.join(BASE_PATH,  "backtest")
IMAGE_PATH = os.path.join(BACKTEST_PATH, "image")
INTERMEDIATE_PATH = os.path.join(BACKTEST_PATH, "intermediate")
STATS_PATH = os.path.join(BACKTEST_PATH, "stats.csv")
CONFIG_FILE = os.path.join(BASE_PATH, "configs", "factor.yaml")


print(f"FACTOR_VALUES_PATH: {FACTOR_VALUES_PATH}")