# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:38:47 2019

@author: 1615055
"""

from data import Dataset
import tensorflow as tf
import numpy as np

filename = 'id_converted_dataset.csv'
datapath = 'C:\\Users\\1615055\\Desktop\\rbm1\\dataset\\'

### dataset = Dataset class
# dataset = Whole dataset
# num_books = number of unique books
# num_users = number of unique users
# dataset_only_rating = dataset of only rating information
# input_vects = input_vectors on each users
# input_vects_c = input_vectors on each users [rating_vect, click/read_vect]
dataset = Dataset(filename, datapath)


''''''''''''''''''''''''
''' IMPLEMENTING RBM '''
''''''''''''''''''''''''
##### Hyperparameters & parameters #####
num_hidden = 100
num_visible = dataset.num_books
num_rate = 5
alpha = 0.5
epoch = 1

## placeholders for parameters

# input_vects (batch)
input_vects = tf.placeholder(tf.float32, [None, num_rate, num_visible])
# input_conds (batch)
input_conds = tf.placeholder(tf.float32, [None, num_visible])
# Weight placeholders
## W -- on visible to hidden, W_c -- on conditional to hidden
W = tf.placeholder(tf.float32, [num_rate, num_hidden, num_visible])
W_c = tf.placeholder(tf.float32, [num_hidden, num_visible])
# bias placeholders
## b_v -- bias on visible nodes, b_h -- bias on hidden nodes
b_v = tf.placeholder(tf.float32, [num_rate, num_visible])
b_h = tf.placeholder(tf.float32, [num_hidden])

# create tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
########################################

'''
Samples additional bias for h given the conditional vector

parameters :
    c = conditional vector
    W_c = weight from conditional to hidden
returns :
    h_c = additional hidden layer bias on conditional   shape: [h, None]
'''
def c_to_h(c, W_c) :
    temp = tf.matmul(c, tf.transpose(W_c))
    h_c = tf.transpose(temp)
    ### returns additional conditional bias for the hidden nodes
    # shape: [h, None]
    return h_c

'''
Samples h given v

parameters :
    v = input for the visible layer
    W = weight from visible to hidden
    b_h = hidden bias
    cond = specify whether conditional vector is given
    c = conditional vector
    W_c = weight from conditional to hidden
returns :
    h = sampled h   shape: [h, None]
    h_ = p(h|v)     shape: [h, None]
'''
def v_to_h(v, W, b_h, cond=False, c=0, W_c=0) :
    ### if conditional vector is given
    if cond :
        # sample additional bias h_c for the hidden layer
        # h_c shape: [h, None]
        h_c = c_to_h(c, W_c)
    ### if the conditional vector is not given, the bias would simply be all 0
    else :
        h_c = tf.zeros([num_hidden, 1])

    h_list = []
    for i in range(0, num_rate) :
        temp = tf.matmul(tf.gather_nd(W, [i]), tf.transpose(v[:,i,:]))
        h_list.append(temp)
    h_ = tf.add_n(h_list)
    h_ = tf.add(h_, tf.expand_dims(b_h, 1))
    h_ = tf.add(h_, h_c)
    h_ = tf.nn.sigmoid(h_)
    h = tf.nn.relu(tf.sign(h_ - tf.random_uniform(tf.shape(h_))))
    ### h = sampled h given v, h_ = p(h|v)   shape: [h, None]
    return h, h_

'''
Samples v given h

parameters:
    h = hidden vector
    W = weight from hidden to visible
    b_v = visible layer bias
    mask = mask for the output (mask non rated items)
    use_mask = whether mask should be used (do not use on predictions)
returns :
    v = visible vector given h   shape: [None, r, v]
'''
def h_to_v(h, W, b_v, mask=0, use_mask=False) :
    v_list = []
    for i in range(0, num_rate) :
        temp = tf.add(tf.matmul(tf.transpose(h), tf.gather_nd(W, [i])), tf.gather_nd(b_v, [[i]]))
        v_list.append(temp)
    v = tf.stack(v_list, axis = 1)
    v = tf.nn.softmax(v, axis = 1)
    if use_mask is True :
        v = tf.multiply(v, mask)
    ### v = p(v|h)   shape: [None, r, v]
    return v

'''
Performs Gibbs sampling on the given data

parameters :
    input_vect = input vector for the visible layer
    W = weight matrix b/w visible and hidden
    b_h = hidden layer bias
    b_v = visible layer bias
    k = gibb sampling iteration
    cond = whether conditional data is given
    c = conditional vector
    W_c = weight matrix b\w conditional and hidden
returns :
    v0, vk = input vector & input after k iteration of gibbs sampling   shape: [None, r, v]
    h0, h0_, hk, hk_ = hidden vector & hidden probability   shape: [None, h]
'''
def gibbs_sampling(input_vect, W, b_h, b_v, k=1, cond=False, c=0, W_c=0) :
    mask = tf.reduce_sum(input_vect, axis = 1, keepdims=True)
    v0 = input_vect
    h0, h0_ = v_to_h(v0, W, b_h, cond, c, W_c)
        
    for i in range(0, k) :
        if i == 0 :
            vk = h_to_v(h0, W, b_v, mask, True)
        else :
            vk = h_to_v(hk, W, b_v, mask, True)
        hk, hk_ = v_to_h(vk, W, b_h, cond, c, W_c)
            
    return v0, h0, h0_, vk, hk, hk_

'''
Trains on the vector(s) given

parameters :
    input_vect = input for the visible layer   shape: [None, r, v] or [r, v]
    W = weight matrix b/w visible and hidden
    b_h = hidden layer bias
    b_v = visible layer bias
    batch = whether data is given in batch
    cond = whether conditional data is given
    c = conditional vector
    W_c = weight matrix b/w conditional and hidden
returns :
    gradients list
    w_grad = gradient for weight matrix W
    b_v_grad = gradient for visible bias b_v
    b_h_grad = gradient for hidden bias b_h
    c_grad = gradient for conditional weight matrix W_c
'''
def train_on_vect(input_vect, W, b_h, b_v, cond=False, c=0, W_c=0) :
    v0, h0, h0_, vk, hk, hk_ = gibbs_sampling(input_vect, W, b_h, b_v, 1, cond, c, W_c)
    
    pos_grad_list = []
    ### pos_grad = v0xh0
    for i in range(0, num_rate) :
        temp = tf.multiply(tf.expand_dims(v0[:,i,:], 1), tf.expand_dims(tf.transpose(h0), 2))
        temp = tf.reduce_mean(temp, axis=0)
        pos_grad_list.append(temp)
    pos_grad = tf.stack(pos_grad_list) ## [h, v]
        
    neg_grad_list = []
    ### neg_grad = vkxhk
    for i in range(0, num_rate) :
        temp = tf.multiply(tf.expand_dims(vk[:,i,:], 1), tf.expand_dims(tf.transpose(hk), 2))
        temp = tf.reduce_mean(temp, axis=0)
        neg_grad_list.append(temp)
    neg_grad = tf.stack(neg_grad_list) ## [h, v]
        
    h0_ = tf.reduce_mean(h0_, axis=1)
    hk_ = tf.reduce_mean(hk_, axis=1)
    v0 = tf.reduce_mean(v0, axis=0)
    vk = tf.reduce_mean(vk, axis=0)

    b_h_grad = h0_ - hk_
    b_v_grad = v0 - vk
    w_grad = pos_grad - neg_grad
    
    ### if conditional data is given, compute gradient for W_c as well
    if cond :
        c_grad = tf.multiply(tf.expand_dims((h0-hk), 1), tf.expand_dims(tf.transpose(c), 0))
        c_grad = tf.reduce_mean(c_grad, axis=2)  ## [h, v]
        return [w_grad, b_v_grad, b_h_grad, c_grad]
    else :
        return [w_grad, b_v_grad, b_h_grad]
    
def predict(input_vect, W, b_h, b_v, cond=False, c=0, W_c=0) :
    h, h_ = v_to_h(input_vect, W, b_h, cond, c, W_c)   # [h, None]
    v = h_to_v(h_, W, b_v)   # [None, r, v]

    prediction = tf.argmax(v, axis = 1)   #[None, v]
        
    return prediction

'''
TRAIN RBM
'''
### initialize parameters
cur_w = np.zeros([num_rate, num_hidden, num_visible], np.float32)
cur_b_v = np.zeros([num_rate, num_visible], np.float32)
cur_b_h = np.zeros([num_hidden], np.float32)
cur_w_c = np.zeros([num_hidden, num_visible], np.float32)

grads = train_on_vect(input_vects, W, b_h, b_v, True, input_conds, W_c)

### TRAIN LOOP
print("starting training")
# per epoch
for i in range(0, epoch) :
    print("epoch: ", i)
    # per batch of size 100 in input vectors
    for j in range(0, len(dataset.input_vects_c), 100) :
        vects = dataset.input_vects_c[j:j+100]
        in_vects = [i[0] for i in vects]
        cond_vects = [i[1][0] for i in vects]
        print(j, ' to ', j+100)
        # obtain gradients for each vector and update
        gradients = sess.run(grads, feed_dict={
                input_vects: in_vects, W: cur_w, b_h: cur_b_h, b_v: cur_b_v, input_conds: cond_vects, W_c: cur_w_c})
    
        cur_w = cur_w + alpha*gradients[0]
        cur_b_v = cur_b_v + alpha*gradients[1]
        cur_b_h = cur_b_h + alpha*gradients[2]
        cur_w_c = cur_w_c + alpha*gradients[3]
        
        
### For making prediction
pred = predict(input_vects, W, b_h, b_v, True, input_conds, W_c)

pred_idx = 95
pred_vects = dataset.input_vects_c[pred_idx]
pred_in_vect = pred_vects[0]
pred_cond_vect = pred_vects[1]

prediction = sess.run(pred, feed_dict = {
        input_vects: [pred_in_vect], W: cur_w, b_h: cur_b_h, b_v: cur_b_v, input_conds: pred_cond_vect, W_c: cur_w_c})


