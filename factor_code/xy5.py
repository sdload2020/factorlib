import pandas as pd
import numpy as np

config = {'factors': [
    {
    "name": "xy5",
    "pre_lag": 10,  # 第一个数据点需要的时间长度, 默认为1000, 单位是frequency参数频率下的, bar数量，建议不要于bar_lag差距过大，会降低计算效率
    "bar_lag": 10,  # rolling时每一个数据点需要的时间长度
    "frequency": '8h',  # '10m', '15m', '30m', '1h',  '4h', '6h', '8h', '1d'
    "run_mode": 'all',  # 'all', 'recent', 'online'
    "start_date": '2022-01-01',  # 开始时间, 只对run_mode='all'有效
    "end_date": '2025-05-29',  # 结束时间, 只对run_mode='all'有效
    # "if_addition": True,       # online模式下需要为true, 其他无所谓，填不填都行
    "bar_fields": [  # 需要的原始数据的字段，首字母大写
        'Open',
        'Last'
    ],
    "start_label": 0,  # start_date对应的label start_date + start_label = 因子的第一个数据点的时间，默认0
    "end_label": 288,  # end_date对应的label end_date + end_label = 因子的最后一个数据点的时间, 默认288
    # "composite_method": True,  # 是否需要依赖前置因子
    # "depend_factor_field": [   # 以来的前置因子的名称，使用时首先要确保前置因子已经计算完毕
    #     'xy',
    #     'xy1'
    # ],
    "factortype": 'pv',
    "factortype2": 'cs',  # 用于区分ts和cs因子
    "author": 'yzl',  ## boxu, gt, yzl 必须按这样填写, 会去区分路径, 正确填写不然会报错
    "if_prod": False,  ## 是否上线
    "level": 1,  ## 层级，依赖原始数据的为1，依赖1级因子的为2，以此类推
    "if_crontab": False,  ## 是否配置cron定时任务
    "out_sample_date": '2024-01-01'  ## out_sample的开始时间
}]
}

def initialize():
    pass
    
def normalize_cs(indicator, tail_l=4):
    indi_std = indicator.replace([np.inf, -np.inf], np.nan).std(axis=1).replace(0, np.nan).ffill().to_frame().values
    indi = np.clip((indicator - indicator.mean(1).to_frame().values) / indi_std, -1 * tail_l, tail_l)
    indi = indi - indi.mean(1).to_frame().values
    sig = indi / indi.abs().sum(1).to_frame().values
    return sig

def preprocess(bar_dict):

    indicator_dict = {'indicator':pd.DataFrame()}
    return indicator_dict
 
def handle_all(bar_dict):
    indicator = bar_dict['Open']/bar_dict['Open'].rolling(5).mean() -1 
    indicator_dict = {'indicator':indicator}
    return indicator_dict

def handle_bar(bar_dict,indicator_dict=None):
    indicator = (bar_dict['Open']/bar_dict['Open'].rolling(5).mean() -1).iloc[-1:]
    indicator_dict = {'indicator':indicator}
    return indicator_dict
