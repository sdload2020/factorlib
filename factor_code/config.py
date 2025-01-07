import os

BASE_PATH = "/home/yangzhilin/backtest/backtest_light/"
RAWDATA_PATH = "/data-platform/crypto/output_parquet/"
DATA_PATH = os.path.join(RAWDATA_PATH, "pv_")
UNIVERSE_PATH = os.path.join(RAWDATA_PATH, "pv_universe.parquet")
FACTOR_VALUES_PATH = os.path.join(BASE_PATH, "factor_values")
BACKTEST_PATH = os.path.join(BASE_PATH,  "backtest")
IMAGE_PATH = os.path.join(BACKTEST_PATH, "image")
INTERMEDIATE_PATH = os.path.join(BACKTEST_PATH, "intermediate")
STATS_PATH = os.path.join(BACKTEST_PATH, "stats.csv")
CONFIG_FILE = os.path.join(BASE_PATH, "configs", "config.yaml")


print(f"FACTOR_VALUES_PATH: {FACTOR_VALUES_PATH}")