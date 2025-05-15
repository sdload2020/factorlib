import pandas as pd
import numpy as np

def initialize():
    pass
    


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