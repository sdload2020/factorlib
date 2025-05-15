import os

# BASE_PATH = "/home/yangzhilin/backtest/backtest_light/code/" # 自定义项目根目录 只需要修改这个路径，其他的不用改
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 自动定义项目根目录
SHARED_PATH = "/data-platform/shared/"  # 共享路径
RAWDATA_PATH = "/data-platform/crypto/output_parquet/"

LOGS_PATH = os.path.join(BASE_PATH, 'logs')  # 日志文件路径
FACTOR_CODE_PATH = os.path.join(BASE_PATH, 'factor_code')  # 因子计算代码路径
DATA_PATH = os.path.join(RAWDATA_PATH, "pv_")
UNIVERSE_PATH = os.path.join(RAWDATA_PATH, "pv_universe.parquet")


BACKTEST_PATH = os.path.join(BASE_PATH,  "result")
FACTOR_VALUES_PATH = os.path.join(BACKTEST_PATH, "indicator")
IMAGE_PATH = os.path.join(BACKTEST_PATH, "report","image")
INTERMEDIATE_PATH = os.path.join(BACKTEST_PATH,"report", "intermediate")
STATS_PATH = os.path.join(BACKTEST_PATH, "stats.csv")
CONFIG_FILE = os.path.join(BASE_PATH, "configs", "factor.yaml")


# print(f"BASE_PATH: {BASE_PATH}")