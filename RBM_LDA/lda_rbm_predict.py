# -*- coding: utf-8 -*-
"""
Make predictions on the LDA-RBM and explain

Created on Mon May 20 15:25:34 2019

@author: 1615055
"""

modelpath = "C:\\Users\\1615055\\Desktop\\2RBM\\model\\"
picklepath = "C:\\Users\\1615055\\Desktop\\2RBM\\pickle\\"
datapath = "C:\\Users\\1615055\\Desktop\\2RBM\\data\\"





import numpy as np
import pickle
import os
import pandas as pd

def open_pickle(picklepath, filename) :
    with open(os.path.join(picklepath, filename), "rb") as fp :
        data = pickle.load(fp)
    return data


" PREDICTION METHODS "
def sample_v(v, num_topics) :
    lda_inputs =    v[:,-num_topics:]
    rating_inputs = v[:,:-num_topics]
    rating_inputs_sample = np.random.binomial(1, p=rating_inputs)
    lda_inputs_sample = []
    for p in lda_inputs :
        p_sample = np.random.multinomial(10, p)
        lda_inputs_sample.append(np.where(p_sample >= 1, 1, 0))
        
    sample = np.concatenate((rating_inputs_sample, lda_inputs_sample), axis=1)
    return sample

def softmax_on_lda(p_v, num_topics) :
    lda_p =    p_v[:,-num_topics:]
    rating_p = p_v[:,:-num_topics]
    factor = np.sum(lda_p, axis=1)
    lda_p = lda_p / np.expand_dims(factor, axis=1)
    
    return np.concatenate((rating_p, lda_p), axis=1)

def predict_rbm(inputs, W, bv, bh, num_topics) :
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    v0 = sample_v(inputs, num_topics)
    q_h0 = sigmoid(np.matmul(v0, W) + bh)
    prediction = sigmoid(np.matmul(q_h0, np.transpose(W)) + bv)
    prediction = softmax_on_lda(prediction, num_topics)
    
    return prediction

def explain_prediction(inputs, prediction) :
    
    genre_dict = open_pickle(picklepath, "genre_dict")
    genre_dict = {v: k for k, v in genre_dict.items()}
    author_dict = open_pickle(picklepath, "author_dict")
    author_dict = {v: k for k, v in author_dict.items()}
    publisher_dict = open_pickle(picklepath, "publisher_dict")
    publisher_dict = {v: k for k, v in publisher_dict.items()}
    book_id_dict = open_pickle(picklepath, "book_id_conversion")
    book_meta_data = open_pickle(picklepath, "new_meta_data")
    
    len_book = len(book_id_dict)
    len_genre = len(genre_dict)
    len_author = len(author_dict)
    len_publisher = len(publisher_dict)
    
    book_input = inputs[:len_book]
    book_prediction = prediction[:len_book]
    book_input_list = [(i, e) for i, e in enumerate(book_input)]
    book_prediction_list = [(i, e) for i, e in enumerate(book_prediction)]
    
    genre_input = inputs[len_book:len_book+len_genre]
    genre_prediction = prediction[len_book:len_book+len_genre]
    genre_input_list = [(i, e) for i, e in enumerate(genre_input)]
    genre_prediction_list = [(i, e) for i, e in enumerate(genre_prediction)]
    
    author_input = inputs[len_book+len_genre:len_book+len_genre+len_author]
    author_prediction = prediction[len_book+len_genre:len_book+len_genre+len_author]
    author_input_list = [(i, e) for i, e in enumerate(author_input)]
    author_prediction_list = [(i, e) for i, e in enumerate(author_prediction)]
    
    publisher_input = inputs[len_book+len_genre+len_author:len_book+len_genre+len_author+len_publisher]
    publisher_prediction = prediction[len_book+len_genre+len_author:len_book+len_genre+len_author+len_publisher]
    publisher_input_list = [(i, e) for i, e in enumerate(publisher_input)]
    publisher_prediction_list = [(i, e) for i, e in enumerate(publisher_prediction)]
    
    def sortSecond(val): 
        return val[1] 
    
    print("books the user originally liked: ")
    book_input_list.sort(key=sortSecond, reverse=True)
    i = 0
    while True :
        if book_input_list[i][1] <= 0 :
            break
        book_id = book_id_dict[book_input_list[i][0]]
        book_title = book_meta_data.loc[book_meta_data.book_id == book_id].title.values[0]
        print("\t", book_title)
        i = i+1
    
    print("genres the user originally liked: ")
    genre_input_list.sort(key=sortSecond, reverse=True)
    i = 0
    while True :
        if genre_input_list[i][1] <=0 :
            break
        genre_title = genre_dict[genre_input_list[i][0]]
        print("\t", genre_title)
        i = i+1
    
    print("authors the user originally liked: ")
    author_input_list.sort(key=sortSecond, reverse=True)
    i = 0
    while True :
        if author_input_list[i][1] <=0 :
            break
        author_title = author_dict[author_input_list[i][0]]
        print("\t", author_title)
        i = i+1
    
    print("publishers the user originally liked: ")
    publisher_input_list.sort(key=sortSecond, reverse=True)
    i = 0
    while True :
        if publisher_input_list[i][1] <=0 :
            break
        publisher_title = publisher_dict[publisher_input_list[i][0]]
        print("\t", publisher_title)
        i = i+1
    
    print()
    
    print("books recommended: ")
    book_prediction_list.sort(key=sortSecond, reverse=True)
    for i in range(0, 5) :
        book_id = book_id_dict[book_prediction_list[i][0]]
        book_title = book_meta_data.loc[book_meta_data.book_id == book_id].title.values[0]
        print("\t", book_title, "\tscore: ", book_prediction_list[i][1])
    
    print("genres recommended: ")
    genre_prediction_list.sort(key=sortSecond, reverse=True)
    for i in range(0, 5) :
        genre_title = genre_dict[genre_prediction_list[i][0]]
        print("\t", genre_title, "\tscore: ", book_prediction_list[i][1])
    
    print("authors recommended: ")
    author_prediction_list.sort(key=sortSecond, reverse=True)
    for i in range(0, 5) :
        author_title = author_dict[author_prediction_list[i][0]]
        print("\t", author_title, "\tscore: ", author_prediction_list[i][1])
    
    print("publishers recommended: ")
    publisher_prediction_list.sort(key=sortSecond, reverse=True)
    for i in range(0, 5) :
        publisher_title = publisher_dict[publisher_prediction_list[i][0]]
        print("\t", publisher_title, "\tscore: ", publisher_prediction_list[i][1])
        
def explain_lda_prediction(inputs, prediction, index, num_topics) :
    book_meta_data = pd.read_csv(os.path.join(datapath, "meta_dataset.csv"))
    
    vector = prediction[-num_topics:]
    sims = index[vector]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])[:10]
    
    print("books that are similar by LDA: ")
    for sim in sims :
        book_id = sim[0]
        book_title = book_meta_data.title[book_id]
        print("\t", book_title)


" OPEN RBM MODEL PARAMETERS "
rbm_parameters = open_pickle(modelpath, "lda_rbm_parameters")
W = rbm_parameters['W']
bv = rbm_parameters['bv']
bh = rbm_parameters['bh']
num_topics = 50

from gensim.test.utils import datapath as dp
from gensim.similarities import MatrixSimilarity
index_file = dp(os.path.join(modelpath, "lda_similarity.index"))
index = MatrixSimilarity.load(index_file)

user_vecs = open_pickle(picklepath, "lda_rbm_inputs")


