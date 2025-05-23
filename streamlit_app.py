import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import errorcode
import os
import logging
from glob import glob
from configs.syspath import IMAGE_PATH, SHARED_PATH
from utils.db_connector import db_config
from streamlit_autorefresh import st_autorefresh
import pytz
from datetime import datetime
from configs.syspath import LOGS_PATH

# 设置日志
LOG_FILE = os.path.join(LOGS_PATH, "streamlit.log")
os.makedirs(LOGS_PATH, exist_ok=True)

logger = logging.getLogger("streamlit")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

logger.info("Streamlit 页面启动。")

# 页面标题
st.title("因子回测结果")

# 自动刷新，每600秒刷新一次
count = st_autorefresh(interval=600000, key="auto_refresh")

# 手动刷新按钮
if st.button("手动刷新"):
    logger.info("用户点击手动刷新按钮。")
    st.rerun()

# 显示上次刷新时间（北京时间）
tz = pytz.timezone('Asia/Shanghai')
last_refresh_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
st.write(f"上次刷新时间: {last_refresh_time}")

# 从数据库获取回测数据
# @st.cache_data(ttl=300)
def get_backtest_data():
    try:
        cnx = mysql.connector.connect(**db_config)
        cursor = cnx.cursor(dictionary=True)
        cursor.execute("SELECT * FROM backtest_result")
        records = cursor.fetchall()
        df = pd.DataFrame(records)
        logger.info("成功从数据库获取回测数据。")
        return df
    except mysql.connector.Error as err:
        logger.error(f"数据库错误: {err}")
        st.error(f"数据库错误: {err}")
        return pd.DataFrame()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'cnx' in locals() and cnx.is_connected():
            cnx.close()

df = get_backtest_data()

st.subheader("数据库中的回测结果")
if not df.empty:
    st.dataframe(df)
else:
    st.info("暂无回测数据。")


IMAGE_PATH_xubo = os.path.join(SHARED_PATH, "xubo", "factor_manage", "result","report","image")
IMAGE_PATH_yzl = os.path.join(SHARED_PATH, "yzl", "factor_manage", "result","report","image")
IMAGE_PATH_gt = os.path.join(SHARED_PATH, "gt", "factor_manage", "result","report","image")
# 获取因子图像
# @st.cache_data(ttl=300)
def get_factor_images():
    factor_imgs = {}
    for image_path in [IMAGE_PATH_xubo, IMAGE_PATH_yzl, IMAGE_PATH_gt]:
        ic_files = glob(os.path.join(image_path, "*_ic_pnl.png"))
        for ic in ic_files:
            base = os.path.basename(ic)
            factor_title = base.replace("_ic_pnl.png", "")
            gmv_path = os.path.join(image_path, f"{factor_title}_gmv_benchmark.png")
            if os.path.exists(gmv_path):
                factor_imgs[factor_title] = (ic, gmv_path)
    logger.info(f"找到 {len(factor_imgs)} 个因子的图像。")
    return factor_imgs

factor_images = get_factor_images()

st.subheader("因子图展示")

if factor_images:
    factor_selected = st.selectbox("选择因子", sorted(factor_images.keys()))
    ic_img, gmv_img = factor_images[factor_selected]
    st.image(ic_img, caption=f"{factor_selected} - IC & PnL", use_container_width=True)
    st.image(gmv_img, caption=f"{factor_selected} - GMV & Benchmark", use_container_width=True)

else:
    st.info("未找到图像文件，请确认 IMAGE_PATH 中存在对应的 PNG 文件。")
