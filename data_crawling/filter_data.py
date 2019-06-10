# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:25:43 2019

@author: 1615055
"""

datapath = "C:\\Users\\1615055\\Desktop\\dataset_crawling\\data\\"

import pandas as pd
import os

'''
dataset = pd.read_csv(os.path.join(datapath, "book_info.csv"))

dataset.sort_values("ISBN", inplace=True)
dataset.drop_duplicates(subset="ISBN", keep=False, inplace=True)

dataset.sort_values("book_id", inplace=True)

dataset.to_csv(os.path.join(datapath, "book_info_no_duplicate.csv"), index=False)

'''

dataset = pd.read_csv(os.path.join(datapath, "book_description.csv"))
dataset_ = dataset.loc[pd.notnull(dataset['Description']) | pd.notnull(dataset['Publisher Desc'])]
dataset_.to_csv(os.path.join(datapath, "book_description_drop_na.csv"), index=False)


basic_data = pd.read_csv(os.path.join(datapath, "book_info_no_duplicate.csv"))
book_id = dataset_.book_id.values.tolist()

new_dataset = basic_data.loc[basic_data.book_id.isin(book_id)]

new_dataset.to_csv(os.path.join(datapath, "book_info_drop_dup_no_desc.csv"))