import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import errorcode
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from glob import glob
from configs.syspath import IMAGE_PATH, SHARED_PATH, LOGS_PATH as LOGS_PATH_1
from utils.db_connector import db_config
from streamlit_autorefresh import st_autorefresh
import pytz
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
sys.path.append('/home/yangzhilin/backtest/')

LOGS_PATH_2 = '/data-platform/gux/factor_manage/logs'

logger = logging.getLogger("streamlit")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
if not logger.handlers:
    tz = pytz.timezone('Asia/Shanghai')
    today_str = datetime.now(tz).strftime("%Y-%m-%d")
    streamlit_path1 = os.path.join(LOGS_PATH_1, "streamlit")
    os.makedirs(streamlit_path1, exist_ok=True)
    fh1 = logging.FileHandler(os.path.join(streamlit_path1, f"{today_str}.log"), encoding="utf-8")
    fh1.setFormatter(fmt)
    logger.addHandler(fh1)
    streamlit_path2 = os.path.join(LOGS_PATH_2, "streamlit")
    os.makedirs(streamlit_path2, exist_ok=True)
    fh2 = logging.FileHandler(os.path.join(streamlit_path2, f"{today_str}.log"), encoding="utf-8")
    fh2.setFormatter(fmt)
    logger.addHandler(fh2)

logger.info("Streamlit 页面启动。")

st.title("因子回测结果")
count = st_autorefresh(interval=600000, key="auto_refresh")
if st.button("手动刷新"):
    logger.info("用户点击手动刷新按钮。")
    st.rerun()

tz = pytz.timezone('Asia/Shanghai')
last_refresh_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
st.write(f"上次刷新时间: {last_refresh_time}")

def filter_parquet_data(df, start_date, end_date, start_label, end_label):
    mask = ((df.index.get_level_values('date') > start_date) & (df.index.get_level_values('date') < end_date)) | ((df.index.get_level_values('date') == start_date) & (df.index.get_level_values('Label') >= start_label)) | ((df.index.get_level_values('date') == end_date) & (df.index.get_level_values('Label') <= end_label))
    return df.loc[mask]

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
    except Exception as e:
        logger.error(f"未预期错误: {e}")
        st.error(f"发生未预期错误: {e}")
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

IMAGE_PATH_xubo = os.path.join(SHARED_PATH, "xubo", "factor_manage", "result", "report", "image")
IMAGE_PATH_yzl = os.path.join(SHARED_PATH, "yzl", "factor_manage", "result", "report", "image")
IMAGE_PATH_gt = os.path.join(SHARED_PATH, "gt", "factor_manage", "result", "report", "image")
IMAGE_PATH_gx = os.path.join(SHARED_PATH, "gux", "factor_manage", "result", "report", "image")

def get_factor_images():
    factor_imgs = {}
    for image_path in [IMAGE_PATH_xubo, IMAGE_PATH_yzl, IMAGE_PATH_gt, IMAGE_PATH_gx]:
        if not os.path.isdir(image_path):
            logger.error(f"图像目录不存在: {image_path}")
            continue
        ic_files = glob(os.path.join(image_path, "*_ic_pnl.png"))
        for ic in ic_files:
            base = os.path.basename(ic)
            factor_title = base.replace("_ic_pnl.png", "")
            gmv_path = os.path.join(image_path, f"{factor_title}_gmv_benchmark.png")
            if os.path.exists(gmv_path):
                factor_imgs[factor_title] = (ic, gmv_path)
            else:
                logger.error(f"缺少 GMV 图像文件: {gmv_path}")
    logger.info(f"找到 {len(factor_imgs)} 个因子的图像。")
    return factor_imgs

factor_images = get_factor_images()

st.subheader("因子图展示")
if factor_images:
    factor_selected = st.selectbox("选择因子", sorted(factor_images.keys()))
    ic_img, gmv_img = factor_images[factor_selected]
    try:
        st.image(ic_img, caption=f"{factor_selected} - IC & PnL", use_container_width=True)
    except Exception as e:
        logger.error(f"显示 IC 图像时出错: {ic_img}，错误: {e}")
        st.error(f"无法显示 IC 图像: {e}")
    try:
        st.image(gmv_img, caption=f"{factor_selected} - GMV & Benchmark", use_container_width=True)
    except Exception as e:
        logger.error(f"显示 GMV 图像时出错: {gmv_img}，错误: {e}")
        st.error(f"无法显示 GMV 图像: {e}")
else:
    st.info("未找到图像文件，请确认 IMAGE_PATH 中存在对应的 PNG 文件。")

st.subheader("因子回测累积Pnl对比")

tz = pytz.timezone('Asia/Shanghai')
today = datetime.now(tz).date()
default_start = datetime(2025,1,1).date()
start_date_all_input = st.date_input("全周期开始日期", value=default_start)
last_n_days = st.number_input("最近天数", min_value=1, value=60)
end_date = today.strftime('%Y-%m-%d')
start_date_all = start_date_all_input.strftime('%Y-%m-%d')
start_date_recent = (today - timedelta(days=last_n_days)).strftime('%Y-%m-%d')
start_label, end_label = 1, 288

users = ['xubo','yzl','gt','gux']
pnl_all = {}
pnl_recent = {}
for user in users:
    intermediate_dir = os.path.join(SHARED_PATH, user, 'factor_manage', 'result', 'report', 'intermediate')
    try:
        files = glob(os.path.join(intermediate_dir, '*.parquet'))
    except Exception as e:
        logger.error(f"读取目录 {intermediate_dir} 出错: {e}")
        continue
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            df_inter = pd.read_parquet(f)
        except Exception as e:
            logger.error(f"加载文件 {f} 出错: {e}")
            continue
        df_all = filter_parquet_data(df_inter, start_date_all, end_date, start_label, end_label)
        if 'pnl' in df_all.columns:
            pnl_all[name] = df_all['pnl'].cumsum()
        df_recent = filter_parquet_data(df_inter, start_date_recent, end_date, start_label, end_label)
        if 'pnl' in df_recent.columns:
            pnl_recent[name] = df_recent['pnl'].cumsum()

all_factors = sorted(set(list(pnl_all.keys()) + list(pnl_recent.keys())))
selected_factors = st.multiselect("选择要展示的因子", options=all_factors, default=all_factors)

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,18))
num_xticks = 10
if pnl_all:
    ref_idx = max(pnl_all.values(), key=lambda s: len(s)).index
    x = list(range(len(ref_idx)))
    labels = [f"{d.date().isoformat()}_{lbl}" for d, lbl in ref_idx]
    for name, series in pnl_all.items():
        if name in selected_factors:
            vals = series.reindex(ref_idx).fillna(method='ffill').fillna(0).values
            ax1.plot(x, vals, label=name)
    step = max(1, len(x)//num_xticks)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels([labels[i] for i in x][::step], rotation=45)
ax1.set_title(f"{start_date_all} - now PnL")
ax1.set_ylabel("Cumulative PnL")
ax1.legend(loc='upper left', fontsize='small', ncol=2)

if pnl_recent:
    ref_idx2 = max(pnl_recent.values(), key=lambda s: len(s)).index
    x2 = list(range(len(ref_idx2)))
    labels2 = [f"{d.date().isoformat()}_{lbl}" for d, lbl in ref_idx2]
    for name, series in pnl_recent.items():
        if name in selected_factors:
            vals2 = series.reindex(ref_idx2).fillna(method='ffill').fillna(0).values
            ax2.plot(x2, vals2, label=name)
    step2 = max(1, len(x2)//num_xticks)
    ax2.set_xticks(x2[::step2])
    ax2.set_xticklabels([labels2[i] for i in x2][::step2], rotation=45)
ax2.set_title(f"last {last_n_days} days PnL")
ax2.set_ylabel("Cumulative PnL")
ax2.legend(loc='upper left', fontsize='small', ncol=2)

st.pyplot(fig)
save_dir = os.path.join(SHARED_PATH, 'yzl', 'factor_manage', 'result', 'report', 'image')
os.makedirs(save_dir, exist_ok=True)
fig.savefig(os.path.join(save_dir, 'pnl_overall.png'), bbox_inches='tight')
