# -*- coding: utf-8 -*-

'''
Created on Tue Mar 19 20:46:28 2019

@author: 이은진

https://nipunbatra.github.io/blog/2017/neural-collaborative-filtering.html
위에 올라온 코드를 바탕으로 [사용자/레시피/리뷰] 데이터를 이용해
NeuMF 모델을 설계해 testset에 모델이 얼마나 잘 작동하는지 테스트하는 코드
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("new_reviews.csv", encoding = "utf-8")

#### filter out users/recipes with less than 5 ratings
min_user_ratings = 5
filter_users = dataset['UserID'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

min_recipe_ratings = 5
filter_recipes = dataset['RecipeID'].value_counts() > min_recipe_ratings
filter_recipes = filter_recipes[filter_recipes].index.tolist()

dataset = dataset[(dataset['UserID'].isin(filter_users) & dataset['RecipeID'].isin(filter_recipes))]

#### renumber the RecipeID and UserID with the smallest range
dataset.RecipeID = dataset.RecipeID.astype('category').cat.codes.values
dataset.UserID = dataset.UserID.astype('category').cat.codes.values

#### split test and training data
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.2)

y_true = test.Rating

#### set neural network with keras
import keras

# number of inputs and latent dimension
n_latent_factors_user = 5
n_latent_factors_recipe = 8
n_latent_factors_mf = 5
n_users, n_recipes = len(dataset.UserID.unique()), len(dataset.RecipeID.unique())

# recipe input layer + embedding for MLP
recipe_input = keras.layers.Input(shape=[1],name='Item')
recipe_embedding_mlp = keras.layers.Embedding(n_recipes + 1, n_latent_factors_recipe, name='Recipe-Embedding-MLP')(recipe_input)
recipe_vec_mlp = keras.layers.Flatten(name='FlattenRecipes-MLP')(recipe_embedding_mlp)
recipe_vec_mlp = keras.layers.Dropout(0.2)(recipe_vec_mlp)
# recipe input layer + embedding for MF
recipe_embedding_mf = keras.layers.Embedding(n_recipes + 1, n_latent_factors_mf, name='Recipe-Embedding-MF')(recipe_input)
recipe_vec_mf = keras.layers.Flatten(name='FlattenRecipes-MF')(recipe_embedding_mf)
recipe_vec_mf = keras.layers.Dropout(0.2)(recipe_vec_mf)

# user input layer + embedding for MLP
user_input = keras.layers.Input(shape=[1],name='User')
user_vec_mlp = keras.layers.Flatten(name='FlattenUsers-MLP')(keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding-MLP')(user_input))
user_vec_mlp = keras.layers.Dropout(0.2)(user_vec_mlp)
# user input layer + embedding for MLP
user_vec_mf = keras.layers.Flatten(name='FlattenUsers-MF')(keras.layers.Embedding(n_users + 1, n_latent_factors_mf,name='User-Embedding-MF')(user_input))
user_vec_mf = keras.layers.Dropout(0.2)(user_vec_mf)

#### MLP
# concatenate user and recipe vectors
concat = keras.layers.concatenate([recipe_vec_mlp, user_vec_mlp], axis = 1,name='Concat')
concat_dropout = keras.layers.Dropout(0.2)(concat)

# add more neural layers to describe the interaction b/w the user and recipe vectors
dense = keras.layers.Dense(200,name='FullyConnected')(concat_dropout)
dense_batch = keras.layers.BatchNormalization(name='Batch')(dense)
dropout_1 = keras.layers.Dropout(0.2,name='Dropout-1')(dense_batch)
dense_2 = keras.layers.Dense(100,name='FullyConnected-1')(dropout_1)
dense_batch_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)
dropout_2 = keras.layers.Dropout(0.2,name='Dropout-2')(dense_batch_2)
dense_3 = keras.layers.Dense(50,name='FullyConnected-2')(dropout_2)
dense_4 = keras.layers.Dense(20,name='FullyConnected-3', activation='relu')(dense_3)

pred_mlp = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)

#### GMF
# layer combining the embeddings of user and recipe with dot product
pred_mf = keras.layers.dot([recipe_vec_mf, user_vec_mf], axes = 1,name='Dot')

#### Combine GMF and MLP
# concatenate the two resulting vectors
combine_mlp_mf = keras.layers.concatenate([pred_mf, pred_mlp], axis = 1,name='Concat-MF-MLP')
result_combine = keras.layers.Dense(100,name='Combine-MF-MLP')(combine_mlp_mf)
deep_combine = keras.layers.Dense(100,name='FullyConnected-4')(result_combine)

result = keras.layers.Dense(1,name='Prediction')(deep_combine)

# set model and compile
model = keras.Model([user_input, recipe_input], result)
opt = keras.optimizers.Adam(lr =0.01)
model.compile(optimizer='adam',loss= 'mean_absolute_error')

# train model
history = model.fit([train.UserID, train.RecipeID], train.Rating, epochs=100, verbose=0, validation_split=0.1)

# get prediction values on the test set
from sklearn.metrics import mean_absolute_error

y_hat_2 = np.round(model.predict([test.UserID, test.RecipeID]),0)

print(mean_absolute_error(y_true, y_hat_2))

print(mean_absolute_error(y_true, model.predict([test.UserID, test.RecipeID])))