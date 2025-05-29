# calc_alpha.py
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import os
import time
from configs.syspath import (BASE_PATH, DATA_PATH, UNIVERSE_PATH,
                                BACKTEST_PATH, RAWDATA_PATH, IMAGE_PATH, STATS_PATH, FACTOR_CODE_PATH, SHARED_PATH, INTERMEDIATE_PATH, FACTOR_VALUES_PATH)
import importlib
import mysql.connector
from configs.dbconfig import db_config
from configs.tablecreator import create_backtest_result_table
import datetime
from mysql.connector import errorcode
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
        factor_module = importlib.import_module(f"factor_code.{self.name}")
        self.initialize = getattr(factor_module, 'initialize')
        self.preprocess = getattr(factor_module, 'preprocess')
        self.handle_all = getattr(factor_module, 'handle_all')
        self.handle_bar = getattr(factor_module, 'handle_bar')
        self.run_mode = prams['run_mode']
        self.composite_method = prams.get('composite_method', False)
        self.depend_factor_field = prams.get('depend_factor_field', None)
        self.author = prams.get('author', 'Unknown')
        # self.start_date = prams.get('start_date', None)
        # self.end_date = prams.get('end_date', None)
        # define the FACTOR_VALUES_PATH as a global variable
        # global FACTOR_VALUES_PATH
        # FACTOR_VALUES_PATH = os.path.join(SHARED_PATH, self.author, 'factorlib', 'factor_values')
        self.factor_values_path = FACTOR_VALUES_PATH
        os.makedirs(self.factor_values_path, exist_ok=True)
        # global INTERMEDIATE_PATH
        # INTERMEDIATE_PATH = os.path.join(SHARED_PATH, self.author, 'factorlib', 'backtest', 'intermediate')
        self.intermediate_path = INTERMEDIATE_PATH
        os.makedirs(self.intermediate_path, exist_ok=True)
        self.factortype = prams.get('factortype', None)
        self.if_prod = prams.get('if_prod', False)
        self.level = prams.get('level', 1)
        self.if_crontab = prams.get('if_crontab', False)
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
        self.univ = pd.read_parquet(UNIVERSE_PATH).reindex(self.mindex).ffill()
        if prams.get('bar_dict') is None:
            data = {}
            for key in self.bar_fields:
                file_path = f"{self.dpath}{key.lower()}.parquet"
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Data file for {key} not found at {file_path}")
                df = pd.read_parquet(file_path)
                data[key] = df.reindex(self.mindex).ffill()
            bar_rescaled = rescale(data, self.fre, self.bar_fields)
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
        if 'Last_next' not in self.bar_dict:
            raise KeyError("'Last_next' key not found in bar_dict.")
        self.ret_1lag = self.bar_dict['Last_next'].pct_change().shift(-1)
        if self.ret_1lag.empty:
            raise ValueError("ret_1lag is empty. Please check if 'Last_next' data is correct.")
        self.mask = self.univ.loc[self.ret_1lag.index].ffill().replace(False, np.nan)
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
            indicator_dict = self.preprocess(preprocess_bar_dict)
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
        if self.composite_method:
            pass
        if self.run_mode == 'all':
            indicator_dict = self.handle_all(self.bar_dict)
            self.indicator_dict = indicator_dict
            self.save_indicator_dict()
            start_point = indicator_dict['indicator'].index[self.bar_lag+1]
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
                if self.depend_factor_field:
                    handle_dict = {
                        key: self.bar_dict[key]
                        for key in self.depend_factor_field
                        if key in self.bar_dict
                    }
                else:
                    handle_dict = self.bar_dict
                addition_dict = self.get_bar_dict(handle_dict, new_indices)
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
        if not os.path.exists(FACTOR_VALUES_PATH_NEW):
            indicator_dict = self.run()
            print(f"Indicator dictionary for {self.name} is not saved.")
        else:
            indicator_dict = pd.read_parquet(FACTOR_VALUES_PATH_NEW)
            print(f"Indicator dictionary for {self.name} is saved.")
        
        if isinstance(indicator_dict, dict) and 'indicator' not in indicator_dict:
            raise ValueError("Indicator dictionary is not properly structured.")

        # 计算各项指标
        sig = (indicator_dict * self.mask).loc[self.start_date:self.end_date]
        raw_pnl = (sig * self.ret_1lag).sum(1)
        vol = raw_pnl.loc[self.rolling_start:self.rolling_end].std()
        delta = sig / vol
        pnl = (delta * self.mask.loc[self.start_date:self.end_date] * self.ret_1lag.loc[self.start_date:self.end_date]).sum(1)
        pnl = pnl.astype(float)
        turnover = delta.diff().abs().sum(1)
        pot = pnl.sum() / turnover.sum() * 10000
        freq_hours = self.freq_hours_map.get(self.fre, 1)
        hd = delta.abs().sum(axis=1).mean() / turnover.mean() * 2 * freq_hours / 288
        mdd = abs((pnl.cumsum() - pnl.cumsum().expanding().max()).min())
        wratio = (pnl > 0).astype(int).sum() / (len(pnl) * 1.0)
        valid_idx = (self.ret_1lag.std(axis=1) != 0) & (sig.std(axis=1) != 0)
        ic = self.ret_1lag.loc[valid_idx].corrwith(sig.loc[valid_idx], axis=1, drop=True)
        # ic = self.ret_1lag.corrwith(sig, axis=1, drop=True)
        ic_mean = ic.mean()
        ir = ic_mean / ic.std()
        ypnl = pnl.mean() * (288 / freq_hours) * 365
        sharpe = pnl.mean() / pnl.std() * np.sqrt((288 / freq_hours) * 365)
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
            'if_prod': self.if_prod,
            'start_date': value_start_date.strftime('%Y-%m-%d'),
            'end_date': value_end_date.strftime('%Y-%m-%d'),
            'frequency': self.fre,
            'pot': float(pot),
            'hd': float(hd),
            'mdd': float(mdd),
            'wratio': float(wratio),
            'ir': float(ir),
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
                print("表 'backtest_result' 已创建。")
            else:
                print("表 'backtest_result' 已存在，无需创建。")

            # 检查是否已存在相同的 name, author, frequency 记录
            # cursor.execute("SELECT * FROM backtest_result WHERE name = %s", (stats_data['name'],))

            cursor.execute("SELECT * FROM backtest_result WHERE name = %s AND author = %s AND frequency = %s", 
                        (stats_data['name'], stats_data['author'], stats_data['frequency']))
            existing_record = cursor.fetchone()
            if existing_record:
                print(f"记录已存在: {existing_record}")

            # 使用 INSERT ... ON DUPLICATE KEY UPDATE
            insert_update_query = """
            INSERT INTO backtest_result (
                name, frequency, updatetime, factortype, level, if_prod, start_date, end_date, 
                pot, hd, mdd, wratio, ir, ypnl, sharpe, max_leverage_ratio,
                if_crontab, out_sample_date, author, factor_value_path, factor_code_path, intermediate_path
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                level = VALUES(level),
                author = VALUES(author),
                updatetime = VALUES(updatetime),
                factortype = VALUES(factortype),
                if_prod = VALUES(if_prod),
                start_date = VALUES(start_date),
                end_date = VALUES(end_date),
                frequency = VALUES(frequency),
                pot = VALUES(pot),
                hd = VALUES(hd),
                mdd = VALUES(mdd),
                wratio = VALUES(wratio),
                ir = VALUES(ir),
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
                    stats_data['level'],
                    stats_data['if_prod'],
                    stats_data['start_date'],
                    stats_data['end_date'],
                    stats_data['pot'],
                    stats_data['hd'],
                    stats_data['mdd'],
                    stats_data['wratio'],
                    stats_data['ir'],
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
                print("SQL 语句执行成功")
            except mysql.connector.Error as err:
                print(f"SQL 语句执行失败: {err}")

            cnx.commit()
            print("事务已提交")

            if cnx.is_connected():
                print("数据库连接仍然有效")
            else:
                print("数据库连接已断开")

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("错误: 用户名或密码错误。")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("错误: 数据库不存在。")
            elif err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("错误: 表已存在。")
            else:
                print(f"发生错误: {err}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'cnx' in locals() and cnx.is_connected():
                cnx.close()

        # 继续保存中间数据到Parquet文件
        intermediate_dir = INTERMEDIATE_PATH
        os.makedirs(intermediate_dir, exist_ok=True)
        intermediate_df = pd.concat([pnl, raw_pnl, gmv, ic, benchmark], axis=1)
        intermediate_df.columns = ['pnl', 'raw_pnl', 'gmv', 'ic', 'benchmark']
        
        if not intermediate_df.index.is_monotonic_increasing:
            intermediate_df = intermediate_df.sort_index()
        
        intermediate_df = intermediate_df.loc[self.start_date:self.end_date]
        
        if not intermediate_df.index.is_monotonic_increasing:
            intermediate_df = intermediate_df.sort_index()
        
        intermediate_path = os.path.join(INTERMEDIATE_PATH, f"{self.name}.parquet")
        intermediate_df.to_parquet(intermediate_path, index=True)
        print(f"Intermediate data saved to {intermediate_path}")
        # 返回统计数据
        stats = {
            'pot': pot,
            'hd': hd,
            'mdd': mdd,
            'wratio': wratio,
            'ir': ir,
            'ypnl': ypnl,
            'sharpe': sharpe,
            'max_leverage_ratio': max_leverage_ratio,
            'raw_pnl': raw_pnl,
            'ic': ic,
            'gmv': gmv,
            'benchmark': benchmark
        }
        return stats
    
    def report_plot(self, stats, author, plot=False, savefig=False, path=IMAGE_PATH, full_title=""):
        if not plot:
            return



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
        
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        color1 = 'tab:blue'
        color2 = 'tab:green'
        ax1.set_ylabel('Cumulative PnL', color=color1)
        ax1.plot(intermediate_df.index, intermediate_df['pnl'].cumsum(), color=color1, label='Cumulative PnL')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.tick_params(axis='x', rotation=45)

        ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        ax2 = ax1.twinx()
        color3 = 'tab:red'
        ax2.set_ylabel('Cumulative IC', color=color3)
        ax2.plot(intermediate_df.index, intermediate_df['ic'].cumsum(), color=color3, label='Cumulative IC')
        ax2.tick_params(axis='y', labelcolor=color3)
        full_title = f"{self.name}_{self.fre}"

        title_metrics = (
            f"pot: {stats['pot']:.2f}, hd: {stats['hd']:.4f}, mdd: {stats['mdd']:.4f}, wratio: {stats['wratio']:.4f},\n"
            f"ir: {stats['ir']:.4f}, ypnl: {stats['ypnl']:.2f}, sharpe: {stats['sharpe']:.4f},\n"
            f"max_leverage_ratio: {stats['max_leverage_ratio']:.4f}"
        )
        fig1.suptitle(f"{full_title} - IC & PnL\n{title_metrics}", fontsize=14)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
        fig1.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(0.1, 0.95), ncol=1)
        
        fig1.tight_layout(rect=[0, 0, 1, 0.95])
        
        fig2, ax3 = plt.subplots(figsize=(12, 6))
        
        color4 = 'tab:purple'
        color5 = 'tab:orange'
        ax3.set_ylabel('GMV', color=color4)
        ax3.plot(intermediate_df.index, intermediate_df['gmv'], color=color4, label='Cumulative GMV')
        ax3.tick_params(axis='y', labelcolor=color4)
        ax3.tick_params(axis='x', rotation=45)

        ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        ax4 = ax3.twinx()
        color6 = 'tab:brown'
        ax4.set_ylabel('Benchmark', color=color6)
        ax4.plot(intermediate_df.index, intermediate_df['benchmark'].cumsum(), color=color6, label='Cumulative Benchmark')
        ax4.tick_params(axis='y', labelcolor=color6)

        fig2.suptitle(f"{full_title} - GMV & Benchmark\n{title_metrics}", fontsize=14)
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        all_lines2 = lines3 + lines4
        all_labels2 = labels3 + labels4
        fig2.legend(all_lines2, all_labels2, loc='upper left', bbox_to_anchor=(0.1, 0.95), ncol=1)
        
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        path = IMAGE_PATH
        # shared_path = os.path.join(SHARED_PATH, author, 'factorlib','backtest', 'image')
        shared_path = path
        if savefig:
            os.makedirs(path, exist_ok=True) 
            fig1.savefig(f'{path}/{full_title}_ic_pnl.png')
            fig2.savefig(f'{path}/{full_title}_gmv_benchmark.png')

            os.makedirs(shared_path, exist_ok=True)
            fig1.savefig(f'{shared_path}/{full_title}_ic_pnl.png')
            fig2.savefig(f'{shared_path}/{full_title}_gmv_benchmark.png')
        
        plt.show()
