import pandas as pd
import numpy as np

def initialize():
    pass
    
def preprocess(bar_dict):
    indicator_dict = {'indicator':pd.DataFrame()}
    return indicator_dict
 
def handle_all(bar_dict):
    indicator1 = bar_dict['xy']/bar_dict['xy'].rolling(5).mean() -1 
    indicator2 = bar_dict['xy1']/bar_dict['xy1'].rolling(5).mean() -1
    indicator = indicator1 + indicator2
    indicator_dict = {'indicator':indicator}
    return indicator_dict

def handle_bar(bar_dict,indicator_dict=None):
    indicator1 = (bar_dict['xy']/bar_dict['xy'].rolling(5).mean() -1 ).iloc[-1:]
    indicator2 = (bar_dict['xy1']/bar_dict['xy1'].rolling(5).mean() -1).iloc[-1:]
    indicator = indicator1 + indicator2
    indicator_dict = {'indicator':indicator}
    return indicator_dict
