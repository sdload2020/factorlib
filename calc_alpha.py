# calc_alpha.py
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import os
import time
from configs.syspath import (BASE_PATH, DATA_PATH, UNIVERSE_PATH,WORK_PATH,LOGS_PATH,
                                BACKTEST_PATH, IMAGE_PATH, STATS_PATH, FACTOR_CODE_PATH, SHARED_PATH, INTERMEDIATE_PATH, FACTOR_VALUES_PATH)
import importlib
import mysql.connector
from configs.dbconfig import db_config
from configs.tablecreator import create_backtest_result_table
import datetime
from mysql.connector import errorcode
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from loguru import logger
import warnings
warnings.filterwarnings("ignore")
from utils.logger_setup import setup_execution_logger

def load_external_module(module_name: str, path_to_py: str):
    spec = importlib.util.spec_from_file_location(module_name, path_to_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def rescale(dft, fre, bar_fields, require_last=True):
    fre2n = {
        '10m': 2,
        '15m': 3,
        '30m': 6,
        '1h': 12,
        '4h': 48,
        '6h': 72,
        '8h': 96,
        '1d': 288
    }
    if fre not in fre2n:
        raise ValueError(f"Unsupported frequency: {fre}")
    n = fre2n[fre]
    resd = {}
    agg_funcs = {
        'Dolvol': 'sum',
        'TradeCount': 'sum',
        'Volume': 'sum',
        'SellDolvol': 'sum',
        'BuyDolvol': 'sum',
        'SellVolume': 'sum',
        'BuyVolume': 'sum',
        'Open': 'first',
        'Last': 'last',
        'High': 'max',
        'Low': 'min'
    }
    for key in bar_fields:
        if key in agg_funcs:
            tmp = dft[key].reset_index().copy()
            tmp['Label'] = ((tmp['Label'] - 1) // n + 1) * n
            resd[key] = tmp.pivot_table(index=['date', 'Label'], aggfunc=agg_funcs[key])
            if tmp['Label'].iloc[-1] % n != 0:
                resd[key] = resd[key].iloc[0:-1]
            else:
                resd[key] = resd[key].iloc[0:]

    if require_last:
        if 'Last' not in resd:
            raise ValueError("'Last' field is required to compute 'Last_next'. Please include it in bar_fields.")
        tmp = dft['Last'].shift(-1).reset_index().copy()
        tmp['Label'] = ((tmp['Label'] - 1) // n + 1) * n
        resd['Last_next'] = tmp.pivot_table(index=['date', 'Label'], aggfunc='last')
        if tmp['Label'].iloc[-1] % n != 0:
            resd['Last_next'] = resd['Last_next'].iloc[1:-1]
        else:
            resd['Last_next'] = resd['Last_next'].iloc[1:]

    return resd
def preprocess(bar_dict):

    indicator_dict = {'indicator':pd.DataFrame()}
    return indicator_dict


class AlphaCalc:
    freq_hours_map = {
        '5m':  5 / 60,
        '10m': 10 / 60,
        '15m': 15 / 60,
        '30m': 30 / 60,
        '1h':  1,
        '4h':  4,
        '6h':  6,
        '8h':  8,
        '1d':  24
    }


    def __init__(self, prams: dict):
        required_keys = ['name', 'pre_lag', 'bar_lag', 'frequency', 'run_mode']
        for key in required_keys:
            if key not in prams:
                raise KeyError(f"Missing required parameter: {key}")
        self.name = prams['name']

        module_path = os.path.join(WORK_PATH, 'factor',self.name+'.py')
        # logger.info("module_path:"+module_path)
        logger.info(f"执行因子:{self.name}")
        logger.info(f"执行因子脚本路径:{module_path}")

        factor_module = load_external_module(self.name, module_path)

        #factor_module = importlib.import_module(f"factorlib.factor_code.{self.name}")
        self.initialize = getattr(factor_module, 'initialize')

        self.handle_all = getattr(factor_module, 'handle_all')
        self.handle_bar = getattr(factor_module, 'handle_bar')
        self.normalize_cs = getattr(factor_module, 'normalize_cs')
        self.run_mode = prams['run_mode']
        self.composite_method = prams.get('composite_method', False)
        self.depend_factor_field = prams.get('depend_factor_field', None)
        self.author = prams.get('author', 'Unknown')

        self.factor_values_path = FACTOR_VALUES_PATH
        os.makedirs(self.factor_values_path, exist_ok=True)
        self.logs_path = LOGS_PATH
        os.makedirs(self.logs_path, exist_ok=True)
        setup_execution_logger(self.logs_path)

        self.intermediate_path = INTERMEDIATE_PATH
        os.makedirs(self.intermediate_path, exist_ok=True)
        self.factortype = prams.get('factortype', None)
        self.factortype2 = prams.get('factortype2', 'cs')  # Default to 'cs' if not provided
        self.if_prod = prams.get('if_prod', False)
        self.level = prams.get('level', 1)
        self.if_crontab = prams.get('if_crontab', True)
        current_date_str = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        self.out_sample_date = prams.get('out_sample_date', None)
        if self.composite_method:
            if not self.depend_factor_field:
                raise ValueError("depend_factor_field must be provided when composite_method is True.")
            if not isinstance(self.depend_factor_field, list):
                raise ValueError("depend_factor_field must be a list of factor names.")
        self.bar_fields = prams.get('bar_fields', ['Open', 'High', 'Low', 'Last', 'Dolvol',
                                                  'TradeCount', 'Volume', 'SellDolvol',
                                                  'BuyDolvol', 'SellVolume', 'BuyVolume'])
        if 'Last' not in self.bar_fields:
            raise ValueError("'Last' field is required for computations. Please include it in bar_fields.")
        self.if_addition = prams.get('if_addition', False)
        fre2n = {
            '5m': 1,
            '10m': 2,
            '15m': 3,
            '30m': 6,
            '1h': 12,
            '2h': 24,
            '4h': 48,
            '6h': 72,
            '8h': 96,
            '1d': 288
        }
        self.fre = prams['frequency']
        if self.fre not in fre2n:
            raise ValueError(f"Unsupported frequency: {self.fre}")
        self.bar_num = fre2n[self.fre]
        self.dpath = DATA_PATH
        if self.run_mode in ['online', 'recent']:
            self._set_dates_online(prams)
        else:
            self._set_dates_manual(prams)
        self.pre_lag = prams['pre_lag']
        self.bar_lag = prams['bar_lag']
        self.fre = prams['frequency']
        self.if_addition = prams.get('if_addition', False)
        self.dr = pd.date_range(start=self.start_date - timedelta(days=(self.pre_lag * self.bar_num) // 288 ),
                                end=self.end_date)
        bars = np.arange(1, 289)
        self.mindex = pd.MultiIndex.from_product([self.dr, bars], names=['date', 'Label'])
        if not os.path.exists(UNIVERSE_PATH):
            raise FileNotFoundError(f"Universe file not found at {UNIVERSE_PATH}")
        # self.mask = pd.read_parquet(UNIVERSE_PATH).reindex(self.mindex).ffill()
        self.funding_rate_path = f"{SHARED_PATH}/crypto/funding_rate/{self.fre}/funding_rate.parquet"
        self.mask_path = f"{SHARED_PATH}/shared/kline/output_parquet_{self.fre}/univ_mask.parquet"
        if prams.get('bar_dict') is None:
            data = {}
            for key in self.bar_fields:
                if key == 'funding_rate' or key == 'mask' or key == 'ret':
                    continue
                file_path = f"{self.dpath}{key.lower()}.parquet"
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Data file for {key} not found at {file_path}")
                df = pd.read_parquet(file_path)
                data[key] = df.reindex(self.mindex)
            bar_rescaled = rescale(data, self.fre, [k for k in self.bar_fields if k != 'funding_rate' and k != 'mask' and k != 'ret'])
            df_mask = pd.read_parquet(self.mask_path)
            df_mask = df_mask.reindex(bar_rescaled['Last'].index)
            self.mask = df_mask
            if 'funding_rate' in self.bar_fields:
                df_fr = pd.read_parquet(self.funding_rate_path)
                df_fr = df_fr.reindex(bar_rescaled['Last'].index)
                bar_rescaled['funding_rate'] = df_fr
            if 'mask' in self.bar_fields:
                bar_rescaled['mask'] = df_mask
            if 'ret' in self.bar_fields:
                bar_rescaled['ret'] = bar_rescaled['Last'].pct_change()
            
            # else:
            #     bar_rescaled = rescale(data, self.fre, self.bar_fields)
            del data
            self.bar_dict = bar_rescaled
        
            if self.composite_method:
                factor_data = {}
                for key in self.depend_factor_field:
                    factor_path = os.path.join(self.factor_values_path, f"{key}.parquet")
                    if not os.path.exists(factor_path):
                        raise FileNotFoundError(f"Dependent factor file '{key}' not found at {factor_path}")
                    df_factor = pd.read_parquet(factor_path)
                    df_factor_filtered = df_factor[df_factor.index.get_level_values('date').isin(self.dr)]
                    factor_data[key] = df_factor_filtered
                self.bar_dict.update(factor_data)
        else:
            self.bar_dict = prams['bar_dict']
            if self.composite_method:
                missing_factors = [factor for factor in self.depend_factor_field if factor not in self.bar_dict]
                if missing_factors:
                    for factor in missing_factors:
                        factor_path = os.path.join(self.factor_values_path, f"{factor}.parquet")
                        if not os.path.exists(factor_path):
                            raise FileNotFoundError(f"Dependent factor file '{factor}' not found at {factor_path}")
                        df_factor = pd.read_parquet(factor_path)
                        self.bar_dict[factor] = df_factor

            if 'funding_rate' in self.bar_dict:
                df_fr = pd.read_parquet(self.funding_rate_path)
                self.bar_dict['funding_rate'] = df_fr
            if 'mask' in self.bar_dict:
                df_mask = pd.read_parquet(self.mask_path)
                self.bar_dict['mask'] = df_mask
            if 'ret' in self.bar_dict:
                self.bar_dict['ret'] = self.bar_dict['Last'].pct_change()

        if 'Last_next' not in self.bar_dict:
            raise KeyError("'Last_next' key not found in bar_dict.")
        self.ret_1lag = self.bar_dict['Last_next'].pct_change().shift(-1)
        # self.ret_1lag = self.bar_dict['Last'].pct_change().shift(-1)
        if self.ret_1lag.empty:
            raise ValueError("ret_1lag is empty. Please check if 'Last_next' data is correct.")
        # self.mask = self.univ.loc[self.ret_1lag.index].ffill().replace(False, np.nan).astype(float)
        self.indicator_dict = {'indicator': None}
        self.run_result = None

    def _set_dates_online(self, prams):
        if self.run_mode == 'online':
            factor_name = self.name
            save_path = os.path.join(self.factor_values_path, f"{factor_name}.parquet")
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"Factor file not found at {save_path} for 'online' run_mode.")
            existing_indicator_df = pd.read_parquet(save_path)
            if not isinstance(existing_indicator_df.index, pd.MultiIndex):
                raise ValueError("Existing indicator_df is not a MultiIndex.")
            if existing_indicator_df.index.names != ['date', 'Label']:
                existing_indicator_df.index.set_names(['date', 'Label'], inplace=True)
            last_date, last_label = existing_indicator_df.index[-1]
            next_label = last_label + self.bar_num
            self.start_date = last_date
            self.start_label = next_label
            last_field_path = f"{self.dpath}last.parquet"
            if not os.path.exists(last_field_path):
                raise FileNotFoundError(f"Last field data file not found at {last_field_path}")
            df_last = pd.read_parquet(last_field_path)
            latest_index = df_last.index[-1]
            self.end_date, self.end_label = latest_index
            self.pre_lag = prams.get('pre_lag', 1000)
            self.rolling_start = pd.to_datetime(prams.get('rolling_start', prams['start_date']))
            self.rolling_end = pd.to_datetime(prams.get('rolling_end', prams['end_date']))
        elif self.run_mode == 'recent':
            last_field_path = f"{self.dpath}last.parquet"
            if not os.path.exists(last_field_path):
                raise FileNotFoundError(f"Last field data file not found at {last_field_path}")
            df_last = pd.read_parquet(last_field_path)
            latest_index = df_last.index[-1]
            self.end_date, self.end_label = latest_index
            self.start_date = self.end_date - timedelta(days=7)
            self.start_label = self.bar_num
            self.rolling_start = pd.to_datetime(prams.get('rolling_start', prams['start_date']))
            self.rolling_end = pd.to_datetime(prams.get('rolling_end', prams['end_date']))

    def _set_dates_manual(self, prams):
        required_keys = ['start_date', 'end_date']
        for key in required_keys:
            if key not in prams:
                raise KeyError(f"Missing required parameter: {key}")
        try:
            self.start_date = pd.to_datetime(prams['start_date'])
            self.end_date = pd.to_datetime(prams['end_date'])
        except ValueError:
            raise ValueError("start_date and end_date must be in 'YYYY-MM-DD' format.")
        if self.start_date > self.end_date:
            raise ValueError("start_date must be earlier than or equal to end_date.")
        self.rolling_start = pd.to_datetime(prams.get('rolling_start', prams['start_date']))
        self.rolling_end = pd.to_datetime(prams.get('rolling_end', prams['end_date']))
        self.start_label = prams.get('start_label', 0)
        self.end_label = prams.get('end_label', 288)

    @staticmethod
    def get_bar_dict(bar_dict, index_list):
        res = {}
        for key in bar_dict.keys():
            res[key] = bar_dict[key].reindex(index_list)
        return res

    def handle_window(self, tindex, indicator_dict=None):
        if indicator_dict is None:
            tloc = self.mindex.get_loc(tindex[0])
            preprocess_index = self.mindex[max(tloc - self.pre_lag, 0):tloc]
            preprocess_bar_dict = self.get_bar_dict(self.bar_dict, preprocess_index)
            indicator_dict = preprocess(preprocess_bar_dict)
        for i in tqdm(tindex, desc="Handling window"):
            if (i[0] < self.start_date) or (i[0] == self.start_date and i[1] < self.start_label):
                continue
            tloc = self.mindex.get_loc(i)
            unit1 = (self.bar_lag * self.bar_num)
            window_start = tloc - unit1 + 1
            window_end = tloc + 1
            if window_start < 0:
                window_start = 0
            step_indices = list(range(window_start + self.bar_num - 1, window_end, self.bar_num))
            window_index = self.mindex[step_indices]
            bar_dict = self.get_bar_dict(indicator_dict, window_index)
            bar_indicator_dict = self.handle_bar(bar_dict, indicator_dict)
            for key in bar_indicator_dict.keys():
                bar_indicator_dict[key] = bar_indicator_dict[key].iloc[-1:]
                if key not in indicator_dict:
                    indicator_dict[key] = bar_indicator_dict[key]
                else:
                    indicator_dict[key] = pd.concat([indicator_dict[key], bar_indicator_dict[key].loc[[i]]])
        return indicator_dict

    def run(self):
        # if self.composite_method:
        #     logger.info("进入复合因子计算:")
            # pass
        if self.run_mode == 'all':
            logger.info("运行模式=all：开始全量计算")
            indicator_dict = self.handle_all(self.bar_dict)
            logger.info(f"批量计算完成：indicator 行数={len(indicator_dict['indicator'])}")
            self.indicator_dict = indicator_dict
            self.save_indicator_dict()
            logger.info(f"因子结果已保存：日期从 {self.start_date} 到 {self.end_date}")
            epsilon = 1e-5
            logger.info(f"头部尾部校验差值阈值：{epsilon}")
            # 头部增量校验，检查头部start_point行, 第一个非nan/0的列的差异
            start_point = indicator_dict['indicator'].index[self.bar_lag + 1]
            logger.info(f"开始检查检查增量头部校验时点：{start_point}, 第一个非nan/0的列值的差异")
            tloc = self.mindex.get_loc(start_point)
            window_size = self.bar_lag * self.bar_num
            ws = max(tloc - window_size + 1, 0)
            we = tloc + 1
            steps = list(range(ws + self.bar_num - 1, we, self.bar_num))
            window_idx = self.mindex[steps]
            bar_dict_head = self.get_bar_dict(self.bar_dict, window_idx)
            head_dict = self.handle_window(window_idx, indicator_dict=bar_dict_head)
            diff_head = head_dict['indicator'].loc[start_point] - indicator_dict['indicator'].loc[start_point]
            head_failed = False
            for uid, d in diff_head.items():
                if pd.notna(d) and abs(d) > epsilon:
                    batch_v = indicator_dict['indicator'].loc[start_point, uid]
                    inc_v   = head_dict['indicator'].loc[start_point, uid]
                    logger.warning(f"头部校验失败, 第一个差异点: UID={uid}, handle_all={batch_v}, handle_bar={inc_v}, 差值={d}")
                    head_failed = True
                    break
            if not head_failed:
                logger.success("头部校验通过: 无差异点")

            # 尾部增量校验 检验尾部7个时间点总和的差异
            tail_idx = indicator_dict['indicator'].index[-7:]
            bar_dict_tail = self.get_bar_dict(self.bar_dict, tail_idx)
            tail_dict = self.handle_window(tail_idx, indicator_dict=bar_dict_tail)
            # 聚合差值总和
            diff_tail = tail_dict['indicator'].loc[tail_idx] - indicator_dict['indicator'].loc[tail_idx]
            chk2 = diff_tail.sum().sum()
            logger.info(f"开始检验尾部7个时间点{tail_idx}总和的差异")
            logger.info(f"尾部校验差值总和：{chk2}")
            tail_failed = chk2 > epsilon
            if tail_failed:
                # 定位第一个差异点用于调试
                for point in tail_idx:
                    for uid, d in diff_tail.loc[point].items():
                        if pd.notna(d) and abs(d) > epsilon:
                            batch_v = indicator_dict['indicator'].loc[point, uid]
                            inc_v   = tail_dict['indicator'].loc[point, uid]
                            logger.warning(f"尾部校验失败, 第一个差异点：时间={point}, UID={uid}, handle_all={batch_v}, handle_bar={inc_v}, 差值={d}")
                            break
                    if tail_failed:
                        break
            else:
                logger.success("尾部校验通过: 差值总和小于阈值")

            # 汇总返回结果,如果有问题，返回有问题的部分
            # if head_failed or tail_failed:
            #     result = {'all': indicator_dict}
            #     if head_failed:
            #         result['head'] = head_dict
            #     if tail_failed:
            #         result['tail'] = tail_dict
            #     self.run_result = result
            # else:
            #     logger.info("校验全部通过，返回全量结果")
            #     self.run_result = indicator_dict

            # 最终返回，并记录失败部分
            failures = []
            if head_failed:
                failures.append('头部')
                logger.warning(f"头部校验失败，返回头部结果和计算使用到的数据以供调试 : {head_dict}")
            if tail_failed:
                failures.append('尾部')
                logger.warning(f"尾部校验失败，返回尾部结果和计算使用到的数据以供调试 : {tail_dict}")
            if failures:
                logger.info(f"校验失败部分: {','.join(failures)}, 仍返回全量indicator_dict")
            else:
                logger.success("所有校验通过, 返回全量indicator_dict")
            logger.info(f"使用到的数据类别: {self.bar_dict.keys()}")
            logger.info(f"使用到的数据: {self.bar_dict}")
            self.run_result = indicator_dict
            return self.run_result

        elif self.run_mode == 'online':
            factor_name = self.name
            indicator_path = os.path.join(self.factor_values_path, f"{factor_name}.parquet")
            if not os.path.exists(indicator_path):
                return None
            try:
                existing_indicator_df = pd.read_parquet(indicator_path)
                if not isinstance(existing_indicator_df.index, pd.MultiIndex):
                    raise ValueError("Existing indicator_df is not a MultiIndex.")
                if existing_indicator_df.index.names != ['date', 'Label']:
                    existing_indicator_df.index.set_names(['date', 'Label'], inplace=True)
                indicator_dict = {'indicator': existing_indicator_df}
                new_indices = self.bar_dict['Last'].index
                if new_indices.empty:
                    self.run_result = indicator_dict
                    return indicator_dict
                # if self.depend_factor_field:
                #     handle_dict = {
                #         key: self.bar_dict[key]
                #         for key in self.depend_factor_field
                #         if key in self.bar_dict
                #     }
                # else:
                #     handle_dict = self.bar_dict
                addition_dict = self.get_bar_dict(self.bar_dict, new_indices)
                new_indicator_dict = self.handle_window(new_indices, indicator_dict=addition_dict)
                for key in new_indicator_dict.keys():
                    if key not in indicator_dict:
                        indicator_dict[key] = new_indicator_dict[key]
                    else:
                        indicator_dict[key] = pd.concat([indicator_dict[key], new_indicator_dict[key]])
                self.indicator_dict = indicator_dict
                self.save_indicator_dict()
                self.run_result = new_indicator_dict
                return self.run_result
            except Exception:
                return None
        elif self.run_mode == 'recent':
            indicator_dict = self.handle_all(self.bar_dict)
            self.indicator_dict = indicator_dict
            self.save_indicator_dict()
            start_point = indicator_dict['indicator'].index[self.bar_lag*2]
            tloc = self.mindex.get_loc(start_point)
            unit1 = (self.bar_lag * self.bar_num)
            window_start = tloc - unit1 + 1
            window_end = tloc + 1
            if window_start < 0:
                window_start = 0
            step_indices = list(range(window_start + self.bar_num - 1, window_end, self.bar_num))
            window_index = self.mindex[step_indices]
            bar_dict = self.get_bar_dict(self.bar_dict, window_index)
            bar_indicator_dict_head = self.handle_window(window_index, indicator_dict=bar_dict)
            chk1 = (bar_indicator_dict_head['indicator'].loc[start_point] - indicator_dict['indicator'].loc[start_point]).values[0]
            epsilon = 1e-10
            if abs(chk1) < epsilon:
                mindex = indicator_dict['indicator'].index[-7:]
                bar_dict = self.get_bar_dict(self.bar_dict, mindex)
                bar_indicator_dict_tail = self.handle_window(mindex, indicator_dict=bar_dict)
                chk2 = (bar_indicator_dict_tail['indicator'].loc[mindex] - indicator_dict['indicator'].loc[mindex]).sum().sum()
                if abs(chk2) < epsilon:
                    self.run_result = indicator_dict
                    return self.run_result
                else:
                    self.run_result = {'all': indicator_dict, 'tail': bar_indicator_dict_tail}
                    return self.run_result
            else:
                self.run_result = {'all': indicator_dict, 'head': bar_indicator_dict_head}
                return self.run_result
        else:
            raise ValueError(f"Invalid run_mode: {self.run_mode}. Choose from 'all', 'online', 'recent'.")

    def save_indicator_dict(self):
        from pandas import IndexSlice as idx
        factor_name = self.name
        save_path = os.path.join(self.factor_values_path, f"{factor_name}.parquet")
        indicator_df = self.indicator_dict['indicator']
        if not isinstance(indicator_df.index, pd.MultiIndex):
            raise ValueError("indicator_dict['indicator'] must have a MultiIndex with 'date' and 'Label'.")
        if indicator_df.index.names != ['date', 'Label']:
            indicator_df = indicator_df.rename_axis(['date', 'Label'])
        start_date = self.start_date
        start_date_plus1 = start_date + timedelta(days=1)
        end_date = self.end_date
        start_label = self.start_label
        end_label = self.end_label
        idx_slice = idx[start_date:end_date, :]
        try:
            filtered_df = indicator_df.loc[idx_slice, :]
        except KeyError:
            raise KeyError("Error in slicing indicator_df.")

        if self.run_mode == 'all':
            mask = (
                ((filtered_df.index.get_level_values('date') > start_date) & (filtered_df.index.get_level_values('date') < end_date)) |
                ((filtered_df.index.get_level_values('date') == start_date) & (filtered_df.index.get_level_values('Label') >= start_label)) |
                ((filtered_df.index.get_level_values('date') == end_date) & (filtered_df.index.get_level_values('Label') <= end_label))
            )
            filtered_df = filtered_df.loc[mask]
            if not filtered_df.index.is_monotonic_increasing:
                filtered_df = filtered_df.sort_index()
        else:
            filtered_df = indicator_df.copy()
        if self.run_mode == 'recent':
            mask = (
                ((filtered_df.index.get_level_values('date') > start_date_plus1) & (filtered_df.index.get_level_values('date') < end_date)) |
                ((filtered_df.index.get_level_values('date') == start_date_plus1) & (filtered_df.index.get_level_values('Label') >= start_label)) |
                ((filtered_df.index.get_level_values('date') == end_date) & (filtered_df.index.get_level_values('Label') <= end_label))
            )
            filtered_df = filtered_df.loc[mask]
            if not filtered_df.index.is_monotonic_increasing:
                filtered_df = filtered_df.sort_index() 
            filtered_df.to_parquet(save_path)
        elif self.if_addition and os.path.exists(save_path) and self.run_mode != 'online':
            existing_df = pd.read_parquet(save_path)
            if not isinstance(existing_df.index, pd.MultiIndex):
                raise ValueError("Existing indicator_df is not a MultiIndex.")
            if existing_df.index.names != ['date', 'Label']:
                existing_df = existing_df.rename_axis(['date', 'Label'])
            if self.run_mode != 'online':
                try:
                    existing_filtered_df = existing_df.loc[idx[self.start_date:self.end_date, :], :]
                except KeyError:
                    raise KeyError("Error in slicing existing_df.")
                existing_mask = (
                    ((existing_filtered_df.index.get_level_values('date') > self.start_date) & (existing_filtered_df.index.get_level_values('date') < self.end_date)) |
                    ((existing_filtered_df.index.get_level_values('date') == self.start_date) & (existing_filtered_df.index.get_level_values('Label') >= self.start_label)) |
                    ((existing_filtered_df.index.get_level_values('date') == self.end_date) & (existing_filtered_df.index.get_level_values('Label') <= self.end_label))
                )
                existing_filtered_df = existing_filtered_df.loc[existing_mask]
                if not existing_filtered_df.index.is_monotonic_increasing:
                    existing_filtered_df = existing_filtered_df.sort_index()
                combined_df = pd.concat([existing_filtered_df, filtered_df]).drop_duplicates()
                combined_df = combined_df.sort_index()
                combined_df.to_parquet(save_path)
        else:
            filtered_df.to_parquet(save_path)

    def gen_delta(self):
        indicator_dict = self.run()
        if isinstance(indicator_dict, dict) and 'indicator' not in indicator_dict:
            raise ValueError("Indicator dictionary is not properly structured.")
        sig = (indicator_dict['indicator'] * self.mask).loc[self.start_date:self.end_date]
        raw_pnl = (sig * self.ret_1lag).sum(1)
        vol = raw_pnl.loc[self.rolling_start:self.rolling_end].std()
        delta = sig / vol
        return delta


    @staticmethod
    def table_exists(cursor, table_name):
        """
        检查表是否存在
        """
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        return cursor.fetchone() is not None

    def report_stats(self):
        FACTOR_VALUES_PATH_NEW = os.path.join(self.factor_values_path, f"{self.name}.parquet")
        logger.info("因子值存储路径:"+FACTOR_VALUES_PATH_NEW)
        if not os.path.exists(FACTOR_VALUES_PATH_NEW):
            indicator_dict = self.run()
            logger.info(f"{self.name} 因子字典未保存.")
        else:
            indicator_dict = pd.read_parquet(FACTOR_VALUES_PATH_NEW)
            logger.info(f"{self.name} 因子字典已保存.")

        if isinstance(indicator_dict, dict) and 'indicator' not in indicator_dict:
            raise ValueError("因子字典结构不正确.")

        # 计算各项指标
        indicator_dict = (indicator_dict.loc[self.start_date:self.end_date] * self.mask.loc[self.start_date:self.end_date]).loc[self.start_date:self.end_date]
        sig = self.normalize_cs(indicator_dict)
    
        raw_pnl = (sig * self.ret_1lag).sum(1)
        vol = raw_pnl.loc[self.rolling_start:self.rolling_end].std()
        if self.factortype2 == 'cs':
            delta = sig
        else:
            delta = sig / vol
        pnl = (delta * self.ret_1lag.loc[self.start_date:self.end_date])
        cpnl = pnl.sum(1)
        cpnl = cpnl.astype(float)
        turnover = delta.diff().abs().sum(1)
        pot = round(pnl.sum(1).sum() / turnover.sum() * 10000, 2)
        freq_hours = self.freq_hours_map.get(self.fre, 1)
        bars_per_day = 24 / freq_hours
        hd = (delta.abs().sum(axis=1).mean() / turnover.mean() * 2) / bars_per_day
        mdd = abs((cpnl.cumsum() - cpnl.cumsum().expanding().max()).min())
        wratio = (cpnl > 0).astype(int).sum() / (len(pnl) * 1.0)
        valid_idx = (self.ret_1lag.std(axis=1) != 0) & (sig.std(axis=1) != 0)
        ic = self.ret_1lag.loc[valid_idx].corrwith(sig.loc[valid_idx], axis=1, drop=True)
        # ic = self.ret_1lag.corrwith(sig, axis=1, drop=True)
        ic_mean = ic.mean()
        icir = ic_mean / ic.std() if ic.std() != 0 else np.nan

        ypnl = cpnl.mean() * bars_per_day * 365
        sharpe = round(cpnl.mean() / cpnl.std() * np.sqrt(bars_per_day * 365), 2)
        benchmark = self.ret_1lag.loc[self.start_date:self.end_date].mean(1)
        gmv = delta.abs().sum(1)
        max_leverage_ratio = gmv.max() / gmv.mean()

        value_start_date = indicator_dict.index[0][0]
        value_end_date = indicator_dict.index[-1][0]
        stats_path = STATS_PATH
        stats_dir = os.path.dirname(stats_path)
        os.makedirs(stats_dir, exist_ok=True)

        current_datetime = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        stats_data = {
            'name': self.name,
            'level': self.level,
            'author': self.author,
            'updatetime': current_datetime,
            'factortype': self.factortype,
            'factortype2': self.factortype2,
            'if_prod': self.if_prod,
            'start_date': value_start_date.strftime('%Y-%m-%d'),
            'end_date': value_end_date.strftime('%Y-%m-%d'),
            'frequency': self.fre,
            'pot': float(pot),
            'hd': float(hd),
            'mdd': float(mdd),
            'wratio': float(wratio),
            'ic' : float(ic_mean),
            'icir': float(icir),
            'ypnl': float(ypnl),
            'sharpe': float(sharpe),
            'max_leverage_ratio': float(max_leverage_ratio),
            'if_crontab': self.if_crontab,
            'out_sample_date': self.out_sample_date,
            'factor_value_path': FACTOR_VALUES_PATH_NEW,
            'factor_code_path': os.path.join(FACTOR_CODE_PATH, f"{self.name}.py"),
            'intermediate_path': os.path.join(INTERMEDIATE_PATH, f"{self.name}.parquet")
        }
        try:
            cnx = mysql.connector.connect(**db_config)
            cursor = cnx.cursor()

            # 确保表存在
            if not self.table_exists(cursor, 'backtest_result'):
                cursor.execute(create_backtest_result_table)
                # logger.info("表 'backtest_result' 已创建。")
                logger.info("表 'backtest_result' 已创建。")
            else:
                # logger.info("表 'backtest_result' 已存在，无需创建。")
                logger.info("表 'backtest_result' 已存在，无需创建。")

            # 检查是否已存在相同的 name, author, frequency 记录
            # cursor.execute("SELECT * FROM backtest_result WHERE name = %s", (stats_data['name'],))

            cursor.execute("SELECT * FROM backtest_result WHERE name = %s AND author = %s AND frequency = %s", 
                        (stats_data['name'], stats_data['author'], stats_data['frequency']))
            existing_record = cursor.fetchone()
            if existing_record:
                # logger.info(f"记录已存在: {existing_record}")
                logger.info(f"记录已存在: {existing_record}")

            # 使用 INSERT ... ON DUPLICATE KEY UPDATE
            insert_update_query = """
            INSERT INTO backtest_result (
                name, frequency, updatetime, factortype, factortype2, level, if_prod, start_date, end_date, 
                pot, hd, mdd, wratio, ic, icir, ypnl, sharpe, max_leverage_ratio,
                if_crontab, out_sample_date, author, factor_value_path, factor_code_path, intermediate_path
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                level = VALUES(level),
                author = VALUES(author),
                updatetime = VALUES(updatetime),
                factortype = VALUES(factortype),
                factortype2 = VALUES(factortype2),
                if_prod = VALUES(if_prod),
                start_date = VALUES(start_date),
                end_date = VALUES(end_date),
                frequency = VALUES(frequency),
                pot = VALUES(pot),
                hd = VALUES(hd),
                mdd = VALUES(mdd),
                wratio = VALUES(wratio),
                ic = VALUES(ic),
                icir = VALUES(icir),
                ypnl = VALUES(ypnl),
                sharpe = VALUES(sharpe),
                max_leverage_ratio = VALUES(max_leverage_ratio),
                if_crontab = VALUES(if_crontab),
                out_sample_date = VALUES(out_sample_date),
                factor_value_path = VALUES(factor_value_path),
                factor_code_path = VALUES(factor_code_path),
                intermediate_path = VALUES(intermediate_path)
            """
            try:
                cursor.execute(insert_update_query, (
                    stats_data['name'],
                    stats_data['frequency'],
                    stats_data['updatetime'],
                    stats_data['factortype'],
                    stats_data['factortype2'],
                    stats_data['level'],
                    stats_data['if_prod'],
                    stats_data['start_date'],
                    stats_data['end_date'],
                    stats_data['pot'],
                    stats_data['hd'],
                    stats_data['mdd'],
                    stats_data['wratio'],
                    stats_data['ic'],
                    stats_data['icir'],
                    stats_data['ypnl'],
                    stats_data['sharpe'],
                    stats_data['max_leverage_ratio'],
                    stats_data['if_crontab'],
                    stats_data['out_sample_date'],
                    stats_data['author'],
                    stats_data['factor_value_path'],
                    stats_data['factor_code_path'],
                    stats_data['intermediate_path']
                ))
                logger.info("SQL 语句执行成功")
            except mysql.connector.Error as err:
                logger.info(f"SQL 语句执行失败: {err}")

            cnx.commit()
            logger.info("事务已提交")

            if cnx.is_connected():
                logger.info("数据库连接仍然有效")
            else:
                logger.info("数据库连接已断开")

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logger.info("错误: 用户名或密码错误。")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                logger.info("错误: 数据库不存在。")
            elif err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                logger.info("错误: 表已存在。")
            else:
                logger.info(f"发生错误: {err}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'cnx' in locals() and cnx.is_connected():
                cnx.close()

        # 继续保存中间数据到Parquet文件
        intermediate_dir = INTERMEDIATE_PATH
        os.makedirs(intermediate_dir, exist_ok=True)
        intermediate_df = pd.concat([cpnl, raw_pnl, gmv, ic, benchmark], axis=1)
        intermediate_df.columns = ['pnl', 'raw_pnl', 'gmv', 'ic', 'benchmark']
        
        if not intermediate_df.index.is_monotonic_increasing:
            intermediate_df = intermediate_df.sort_index()
        
        intermediate_df = intermediate_df.loc[self.start_date:self.end_date]
        
        if not intermediate_df.index.is_monotonic_increasing:
            intermediate_df = intermediate_df.sort_index()
        
        intermediate_path = os.path.join(INTERMEDIATE_PATH, f"{self.name}.parquet")
        intermediate_df.to_parquet(intermediate_path, index=True)

        # 返回统计数据
        stats = {
            'pot': pot,
            'hd': hd,
            'mdd': mdd,
            'wratio': wratio,
            'ic': ic_mean,
            'icir': icir,
            'ypnl': ypnl,
            'sharpe': sharpe,
            'max_leverage_ratio': max_leverage_ratio,
            'raw_pnl': raw_pnl,
            'gmv': gmv,
            'benchmark': benchmark
        }
        logger.info(f"回测表现 {self.name}:\n{stats_data}")
        return stats
    def report_plot(self, stats, author, plot=False, savefig=False, path=IMAGE_PATH, full_title=""):
        if not plot:
            return

        freq_hours = self.freq_hours_map.get(self.fre, 1)
        bars_per_day = 24 / freq_hours
        n_month = int(bars_per_day * 30)

        intermediate_path = os.path.join(INTERMEDIATE_PATH, f"{self.name}.parquet")
        if not os.path.exists(intermediate_path):
            raise FileNotFoundError(f"Intermediate data file not found at {intermediate_path}")
        os.makedirs(path, exist_ok=True)
        intermediate_df = pd.read_parquet(intermediate_path)
        if not intermediate_df.index.is_monotonic_increasing:
            intermediate_df = intermediate_df.sort_index()
        intermediate_df = intermediate_df.loc[self.start_date:self.end_date]
        if not isinstance(intermediate_df.index, pd.MultiIndex):
            raise ValueError("Intermediate DataFrame must have a MultiIndex with 'date' and 'Label'.")
        intermediate_df.index = [f"{x.date().isoformat()}_{y}" for x, y in intermediate_df.index]

        subset_df = intermediate_df.tail(n_month)

        fig1, (ax1, ax1b) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.set_ylabel('Cumulative PnL')
        ax1.plot(intermediate_df.index, intermediate_df['pnl'].cumsum(), label='Cumulative PnL')
        ax1.tick_params(axis='x', rotation=45)
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        ax2 = ax1.twinx()
        ax2.set_ylabel('Cumulative IC')
        ax2.plot(intermediate_df.index, intermediate_df['ic'].cumsum(), color='tab:red', label='Cumulative IC')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
        fig1.suptitle(f"{self.name}_{self.fre}\n"
                      f"pot: {stats['pot']:.2f}, sharpe: {stats['sharpe']:.4f}, mdd: {stats['mdd']:.4f}, wratio: {stats['wratio']:.4f},\n"
                      f"ic: {stats['ic']:.4f}, icir: {stats['icir']:.4f}, ypnl: {stats['ypnl']:.2f}, hd: {stats['hd']:.4f},\n"
                      f"max_leverage_ratio: {stats['max_leverage_ratio']:.4f}", fontsize=14)
        fig1.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(0.1, 0.95), ncol=1)
        ax1b.set_ylabel('Cumulative PnL (Last 30 Days)')
        ax1b.plot(subset_df.index, subset_df['pnl'].cumsum(), label='Cumulative PnL')
        ax1b.tick_params(axis='x', rotation=45)
        ax1b.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        ax1b2 = ax1b.twinx()
        ax1b2.set_ylabel('Cumulative IC (Last 30 Days)')
        ax1b2.plot(subset_df.index, subset_df['ic'].cumsum(), color='tab:red', label='Cumulative IC')
        fig1.tight_layout(rect=[0, 0, 1, 0.95])

        fig2, (ax3, ax3b) = plt.subplots(2, 1, figsize=(12, 10))
        ax3.set_ylabel('GMV')
        ax3.plot(intermediate_df.index, intermediate_df['gmv'], label='GMV')
        ax3.tick_params(axis='x', rotation=45)
        ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        ax4 = ax3.twinx()
        ax4.set_ylabel('Benchmark')
        ax4.plot(intermediate_df.index, intermediate_df['benchmark'].cumsum(), color='tab:orange', label='Benchmark')
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        all_lines2 = lines3 + lines4
        all_labels2 = labels3 + labels4
        fig2.suptitle(f"{self.name}_{self.fre}\n"
                      f"pot: {stats['pot']:.2f}, sharpe: {stats['sharpe']:.4f}, mdd: {stats['mdd']:.4f}, wratio: {stats['wratio']:.4f},\n"
                      f"ic: {stats['ic']:.4f}, icir: {stats['icir']:.4f}, ypnl: {stats['ypnl']:.2f}, hd: {stats['hd']:.4f},\n"
                      f"max_leverage_ratio: {stats['max_leverage_ratio']:.4f}", fontsize=14)
        fig2.legend(all_lines2, all_labels2, loc='upper left', bbox_to_anchor=(0.1, 0.95), ncol=1)
        ax3b.set_ylabel('GMV (Last 30 Days)')
        ax3b.plot(subset_df.index, subset_df['gmv'], label='GMV')
        ax3b.tick_params(axis='x', rotation=45)
        ax3b.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        ax4b = ax3b.twinx()
        ax4b.set_ylabel('Benchmark (Last 30 Days)')
        ax4b.plot(subset_df.index, subset_df['benchmark'].cumsum(), color='tab:orange', label='Benchmark')
        fig2.tight_layout(rect=[0, 0, 1, 0.95])

        if savefig:
            os.makedirs(path, exist_ok=True)
            fig1.savefig(f'{path}/{self.name}_{self.fre}_ic_pnl.png')
            fig2.savefig(f'{path}/{self.name}_{self.fre}_gmv_benchmark.png')
            
        plt.show()