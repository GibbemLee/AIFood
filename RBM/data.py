# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:49:49 2019

@author: 1615055
"""

import numpy as np
import os
import pandas as pd

'''
class Dataset
'''
class Dataset :
    
    def __init__(self, filename, datapath) :
        self.filename = filename
        self.datapath = datapath
        self.open_dataset()
        self.get_num_books_users()
        self.get_only_rating()
        self.get_input_vectors(truncate = True)
        self.get_input_vectors_conditional(truncate = True)

    ### open dataset file (of datapath + filename)
    def open_dataset(self) :
        dataset = pd.read_csv(os.path.join(self.datapath, self.filename))
        self.dataset = dataset
        return dataset

    ### get number of unique books and users in dataset
    def get_num_books_users(self) :
        num = self.dataset.nunique()
        self.num_books = num['book_id']
        self.num_users = num['user_id']
        return self.num_books, self.num_users

    ### get only rating information in the dataset
    def get_only_rating(self) :
        self.dataset_only_rating = self.dataset.loc[(self.dataset['isRead'] == True) & (self.dataset['rating'] >0)]
        return self.dataset_only_rating

    ### create input vectors for training and testing
    ### vector = [5, num_books] where i[5][1] = 1 if book1 is rated 5
    def get_input_vectors(self, truncate = False) :
        if truncate is True :
            dataset_ = self.dataset_only_rating.loc[:10000]
        else :
            dataset_ = self.dataset_only_rating
            
        self.input_vects = []
        
        for i in dataset_['user_id'].unique() :
            temp = np.zeros((5, self.num_books))
            for j in dataset_.loc[dataset_['user_id'] == i].iterrows() :
                temp[int(j[1]['rating'])-1][int(j[1]['book_id'])] = 1
            self.input_vects.append(temp)
        
        return self.input_vects
    
    def get_input_vectors_conditional(self, truncate = False) :
        if truncate is True :
            dataset_ = self.dataset.loc[:10000]
        else :
            dataset_ = self.dataset
            
        self.input_vects_c = []
        
        for i in dataset_['user_id'].unique() :
            temp1 = np.zeros((5, self.num_books))
            temp2 = np.zeros((1, self.num_books))
            for j in dataset_.loc[dataset_['user_id'] == i].iterrows() :
                if j[1]['rating']<1 :
                    temp2[0][int(j[1]['book_id'])] = 1
                else :
                    temp1[int(j[1]['rating'])-1][int(j[1]['book_id'])] = 1
            self.input_vects_c.append([temp1, temp2])
        
        return self.input_vects_c







