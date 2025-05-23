import os
import sys

# 判断是否打包运行
if getattr(sys, 'frozen', False):
    # PyInstaller 打包后的路径
    BASE_PATH = sys._MEIPASS
else:
    # 正常开发环境路径
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# 组员工作目录
WORK_PATH = "/data-platform/gux/factor_manage"

# 相对目录处理
SHARED_PATH = "/data-platform/"
RAWDATA_PATH =  "/data-platform/crypto/output_parquet/"
DATA_PATH = os.path.join(RAWDATA_PATH, "pv_")
UNIVERSE_PATH = os.path.join(RAWDATA_PATH, "pv_universe.parquet")

LOGS_PATH = "/data-platform/gux/lianghua/factorlib/logs"  # 日志文件路径
FACTOR_CODE_PATH = "/data-platform/gux/factor_manage/factor"  # 因子计算代码路径
BACKTEST_PATH = "/data-platform/gux/factor_manage/result"
FACTOR_VALUES_PATH = "/data-platform/gux/factor_manage/result/indicator"
IMAGE_PATH = "/data-platform/gux/factor_manage/result/report/image"
INTERMEDIATE_PATH = "/data-platform/gux/factor_manage/result/report/intermediate"
STATS_PATH = "/data-platform/gux/factor_manage/result/stats.csv"
CONFIG_FILE = "/data-platform/gux/factor_manage/result/configs/factor.yaml"


# print(f"BASE_PATH: {BASE_PATH}")
