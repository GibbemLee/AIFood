# -*- coding: utf-8 -*-
"""
Make RBM inputs with the lda inputs

Created on Mon May 20 14:15:44 2019

@author: 1615055
"""

datapath = "C:\\Users\\1615055\\Desktop\\2RBM\\data\\"
picklepath = "C:\\Users\\1615055\\Desktop\\2RBM\\pickle\\"





import os
import pickle
import numpy as np

def open_pickle(picklepath, filename) :
    with open(os.path.join(picklepath, filename), "rb") as fp :
        data = pickle.load(fp)
    return data

def pickle_data(picklepath, filename, data) :
    with open(os.path.join(picklepath, filename), "wb") as fp :
        pickle.dump(data, fp)

lda_inputs = open_pickle(picklepath, "lda_inputs")
# open rating_rbm_inputs made from ../RBM_Rating/rating_make_rbm_input.py
rating_rbm_inputs = open_pickle(picklepath, "rating_rbm_inputs")

# just concatenate the two inputs by axis=1
lda_rbm_inputs = np.concatenate((rating_rbm_inputs, lda_inputs), axis=1)
pickle_data(picklepath, "lda_rbm_inputs", lda_rbm_inputs)

