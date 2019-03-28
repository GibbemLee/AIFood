# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:31:08 2019

@author: 이은진
"""

import pandas as pd
import numpy as np
import random

ratings = pd.read_csv('new_reviews.csv')


#### filter out users/recipes with less than 5 ratings
min_user_ratings = 5
filter_users = ratings['UserID'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

min_recipe_ratings = 5
filter_recipes = ratings['RecipeID'].value_counts() > min_recipe_ratings
filter_recipes = filter_recipes[filter_recipes].index.tolist()

ratings = ratings[(ratings['UserID'].isin(filter_users) & ratings['RecipeID'].isin(filter_recipes))]

### create Dataframe remembering the original user/recipe id to the updated user/recipe id
u = ratings['UserID'].unique()
user_id_conversion = pd.DataFrame({'UserID': u})
user_id_conversion['NewIndex'] = user_id_conversion.index

r = ratings['RecipeID'].unique()
recipe_id_conversion = pd.DataFrame({'RecipeID': r})
recipe_id_conversion['NewIndex'] = recipe_id_conversion.index

### change the ratings dataset user/recipe id to the updated ids
pd.options.mode.chained_assignment = None

for x in range(0, ratings.shape[0]) :
    temp = ratings.iloc[x,1]
    ratings.iloc[x,1] = user_id_conversion[user_id_conversion['UserID'] == temp]['NewIndex'].values[0]
    
    temp = ratings.iloc[x,0]
    ratings.iloc[x,0] = recipe_id_conversion[recipe_id_conversion['RecipeID'] == temp]['NewIndex'].values[0]

### pivot the dataset and make arrays of each user where the i-th entry is the user's rating of the recipe i
user_group = ratings.groupby("UserID")

num_recipes = len(ratings['RecipeID'].unique())

total = []
for userID, curReader in user_group:
    temp = np.zeros(num_recipes)

    for num, recipe in curReader.iterrows():
        temp[int(recipe['RecipeID'])] = recipe['Rating'] / 5.0

    total.append(temp)

### shuffle the data and divide to test and train sets
random.shuffle(total)
train = total[:1500]
valid = total[1500:]

import tensorflow as tf

hiddenUnits = 64
visibleUnits = num_recipes

# Number of unique movies
vb = tf.placeholder(tf.float32, [visibleUnits])

# Number of features that we are going to learn
hb = tf.placeholder(tf.float32, [hiddenUnits])
W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])  # Weight Matrix

v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
# Gibb's Sampling
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))

# Hidden layer activation
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

# Learning rate
alpha = 0.6

# Creating the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)

# Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

# Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

# Set the error function, here we use Mean Absolute Error Function
err = v0 - v1
err_sum = tf.reduce_mean(err * err)

# Current weight
cur_w = np.random.normal(loc=0, scale=0.01, size=[visibleUnits, hiddenUnits])

# Current visible unit biases
cur_vb = np.zeros([visibleUnits], np.float32)

# Current hidden unit biases
cur_hb = np.zeros([hiddenUnits], np.float32)

# Previous weight
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

# Previous visible unit biases
prv_vb = np.zeros([visibleUnits], np.float32)

# Previous hidden unit biases
prv_hb = np.zeros([hiddenUnits], np.float32)

# Running the session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

def free_energy(v_sample, W, vb, hb):
    ''' Function to compute the free energy '''
    wx_b = np.dot(v_sample, W) + hb
    vbias_term = np.dot(v_sample, vb)
    hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis = 1)
    return -hidden_term - vbias_term

epochs = 50
batchsize = 100
errors = []
energy_train = []
energy_valid = []
for i in range(epochs):
    for start, end in zip(range(0, len(train), batchsize), range(batchsize, len(train), batchsize)):
        batch = train[start:end]
        cur_w = sess.run(update_w, feed_dict={
                         v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={
                          v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={
                          v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb

    energy_train.append(np.mean(free_energy(train, cur_w, cur_vb, cur_hb)))
    energy_valid.append(np.mean(free_energy(valid, cur_w, cur_vb, cur_hb)))

    errors.append(sess.run(err_sum, feed_dict={
                  v0: train, W: cur_w, vb: cur_vb, hb: cur_hb}))
    if i % 10 == 0:
        print("Error in epoch {0} is: {1}".format(i, errors[i]))
        
        
### testing on a particular user
user = 22
# reconstruct the user's array of ratings
u = ratings[ratings['UserID'] == user]
temp = np.zeros(num_recipes)
for x in u.iterrows() :
    temp[x[1]['RecipeID']] = x[1]['Rating']
inputUser = [temp]


# Feeding in the User and Reconstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})


recipes = recipe_id_conversion[['NewIndex']]
recipes['ratings'] = rec[0]


# Find all books the mock user has read before
cooked = ratings[ratings['UserID'] == user]['RecipeID'].values.tolist()

recipes_cooked = recipes.loc[recipes['NewIndex'].isin(cooked)]

recipes_cooked = recipes_cooked.sort_values(by = 'ratings', ascending = False)

recipes_not_cooked = recipes.loc[~recipes['NewIndex'].isin(cooked)]

recipes_not_cooked = recipes_not_cooked.sort_values(by = 'ratings', ascending = False)
