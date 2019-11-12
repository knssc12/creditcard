import numpy as np
import pandas as pd
from pandas import DataFrame

def prepare_data(csv):
    data = pd.read_csv(csv)
    lim = np.arange(1001,len(data))
    data = data.drop(lim)
    return data

def data_clean():
    data = prepare_data("creditcard.csv")
    data.to_csv("creditcard_small.csv")
