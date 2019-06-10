# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:10:33 2019

@author: 1615055
"""

import os
import pandas as pd

datapath = "C:\\Users\\1615055\\Desktop\\dataset_crawling\\data\\reviews"
filename = "review_{}_to_{}.csv"

def open_dataframe(datapath, filename) :
    df = pd.read_csv(os.path.join(datapath, filename))
    return df

def concat_df(src, new) :
    result = pd.concat([src, new])
    return result

result = open_dataframe(datapath, filename.format(0, 100))

for start, end in zip(range(100, 2901, 100), range(200, 3001, 100)) :
    result = concat_df(result, open_dataframe(datapath, filename.format(start, end)))
    
result.to_csv(os.path.join(datapath, "reviews_whole.csv"), index=False)