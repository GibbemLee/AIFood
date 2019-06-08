# -*- coding: utf-8 -*-
"""
LDA + Rating RBM

Created on Mon May 20 14:30:08 2019

@author: 1615055
"""

picklepath = "C:\\Users\\1615055\\Desktop\\2RBM\\pickle\\"
modelpath = "C:\\Users\\1615055\\Desktop\\2RBM\\model\\"





import os
import pickle
import numpy as np
import time
import random

def open_pickle(picklepath, filename) :
    with open(os.path.join(picklepath, filename), "rb") as fp :
        data = pickle.load(fp)
    return data

def pickle_data(picklepath, filename, data) :
    with open(os.path.join(picklepath, filename), "wb") as fp :
        pickle.dump(data, fp)


" RBM METHODS "
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
    
def train_rbm(train, valid, num_visible, num_hidden, epochs, batchsize, alpha, k, num_topics) :
    W = np.zeros([num_visible, num_hidden])
    bv = np.zeros(num_visible)
    bh = np.zeros(num_hidden)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def free_energy(p_v0, W, bv, bh, num_topics) :
        v0 = sample_v(p_v0, num_topics)
        q_h0 = sigmoid(np.matmul(v0, W) + bh)
        h0 = np.random.binomial(1, p=q_h0)
        
        temp1 = np.reshape(np.multiply(np.expand_dims(v0, axis=2), np.expand_dims(h0, axis=1)), (-1, num_visible*num_hidden))
        temp2 = np.reshape(W, (num_visible*num_hidden, -1))
        term1 = np.matmul(temp1, temp2)
        term2 = np.matmul(v0, np.expand_dims(bv, axis=1))
        term3 = np.matmul(h0, np.expand_dims(bh, axis=1))

        free_energy = np.mean(-term1-term2-term3)
        return free_energy
        
    def RBM_update(input_vects, batchsize, alpha, W, bv, bh, k, num_topics) :
    
        for i in range(0, len(input_vects), batchsize):
            p_v0 = input_vects[i:i+batchsize]
        
            v0 = sample_v(p_v0, num_topics)
            q_h0 = sigmoid(np.matmul(v0, W) + bh)
            h0 = np.random.binomial(1, p=q_h0)
            hk = h0
            
            for j in range(0, k) :
                p_vk = sigmoid(np.matmul(hk, np.transpose(W)) + bv)
                p_vk = softmax_on_lda(p_vk, num_topics)
                vk = sample_v(p_vk, num_topics)
                q_hk = sigmoid(np.matmul(vk, W) + bh)
                hk = np.random.binomial(1, p=q_hk)

            CD = (np.matmul(np.transpose(v0), h0) - np.matmul(np.transpose(vk), q_hk)) / np.shape(p_v0)[0]
            bh_grad = np.mean((h0 - q_hk), axis=0)
            bv_grad = np.mean((v0 - vk), axis=0)

            W = W + alpha*CD
            bh = bh + alpha*bh_grad
            bv = bv + alpha*bv_grad
    
        return W, bh, bv
    
    
    for epoch in range(0, epochs) :
        start_time = time.time()
        W, bh, bv = RBM_update(train, batchsize, alpha, W, bv, bh, k, num_topics)
        end_time = time.time()

        print("epoch: ", epoch)
        print("time taken: ", end_time-start_time)
        print("free energy on training: ", free_energy(np.asarray(random.sample(train.tolist(), 200)), W, bv, bh, num_topics))
        print("free energy on valid: ", free_energy(valid, W, bv, bh, num_topics))
        print()
        
    return W, bv, bh



" TRAIN RBM "
inputs = open_pickle(picklepath, "lda_rbm_inputs")
random.shuffle(inputs)

train = inputs[:int(len(inputs)*0.8)]
valid = inputs[int(len(inputs)*0.8):]

# hyperparameters
num_visible =   len(train[0])
num_hidden =    100
epochs =        100
batchsize =     20
alpha =         0.1
k =             5
num_topics =    50

W, bv, bh = train_rbm(train, valid, num_visible, num_hidden, epochs, batchsize, alpha, k, num_topics)
pickle_data(modelpath, "lda_rbm_parameters", {'W': W, 'bv': bv, 'bh': bh})