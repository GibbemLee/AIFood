# -*- coding: utf-8 -*-
"""
Make LDA inputs for the RBM

Created on Mon May 20 12:37:21 2019

@author: 1615055
"""

datapath = "C:\\Users\\1615055\\Desktop\\2RBM\\data\\"
picklepath = "C:\\Users\\1615055\\Desktop\\2RBM\\pickle\\"
modelpath = "C:\\Users\\1615055\\Desktop\\2RBM\\model\\"





import os
import pickle
import pandas as pd
import numpy as np
import re

def pickle_data(picklepath, filename, data) :
    with open(os.path.join(picklepath, filename), "wb") as fp :
        pickle.dump(data, fp)


" STEM AND FILTER VERBS AND UNNECESSARY WORDS "
from konlpy.tag import Okt
okt = Okt()
def process_input(ins) :
    def filter_unnecessary_tags(token_list) :
        used_tags = ['Adjective', 'Adverb', 'Alpha', 'Foreign', 'Hashtag', 'Noun', 'Number']
        res = [j[0] for j in token_list if j[1] in used_tags]
        res = ' '.join(res)
        return res
    temp = okt.pos(ins, norm=True, stem=True)
    temp = filter_unnecessary_tags(temp)
    return temp

info_data = pd.read_csv(os.path.join(datapath, "info_data.csv"))
res = []
for i in range(0, info_data.shape[0]) :
    if i%100 == 0:
        print(i)
    temp = process_input(info_data.iloc[i]['info'])
    res.append(temp)
info_data['info_filtered_no_verbs'] = res
info_data.to_csv(os.path.join(datapath, "info_data_no_verbs.csv"), index=False)


" PRODUCE LDA AND TF_IDF MODELS "
documents = info_data.info_filtered_no_verbs.values.tolist()
documents = [re.sub('[,\.!?]', '', x) for x in documents]

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
texts = [tokenizer.tokenize(raw) for raw in documents]

from gensim import corpora, models
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# use TF_IDF
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# make LDA model
num_topics = 50
ldamodel = models.ldamodel.LdaModel(corpus_tfidf, num_topics=num_topics, id2word = dictionary, passes=20)

# import tfidf and ldamodel
from gensim.test.utils import datapath as dp
lda_file = dp(os.path.join(modelpath, "lda_model"))
tfidf_file = dp(os.path.join(modelpath, "tfidf_model"))
ldamodel.save(lda_file)
tfidf.save(tfidf_file)


" MAKE LDA TOPIC VECTOR FOR EACH USER FOR RBM INPUTS "
review_data = pd.read_csv(os.path.join(datapath, "review_dataset.csv"))
review_data = review_data.dropna()

def get_user_doc(review_data, info_data) :
    user_list = review_data.user_id.unique()
    user_docs = []
    for user in user_list :
        user_doc = []
        user_book = review_data.loc[review_data.user_id == user]['book_id'].values.tolist()
        for book in user_book :
            user_doc.append(info_data.loc[info_data.book_id == book]['info_filtered'].values[0])
        user_docs.append(' '.join(user_doc))
    return user_docs

def get_user_vec(user_doc) :
    user_vec = []
    for doc in user_doc :
        tokens = tokenizer.tokenize(doc)
        bow = dictionary.doc2bow(tokens)
        bow_tfidf = tfidf[bow]
        vec = ldamodel.get_document_topics(bow_tfidf)
        temp = np.zeros(num_topics)
        for topic in vec :
            temp[topic[0]] = topic[1]
        user_vec.append(temp)
    return user_vec

user_doc = get_user_doc(review_data, info_data)
user_vec = get_user_vec(user_doc)
pickle_data(picklepath, "lda_inputs", user_vec)


" MAKE SIMILARITY MATRIX "
from gensim.similarities import MatrixSimilarity
index = MatrixSimilarity(ldamodel[corpus_tfidf])
index_file = dp(os.path.join(modelpath, "lda_similarity.index"))
index.save(index_file)

'''
vector = prediction[:,-num_topics:][0]
sims = index[vector]
sims = sorted(enumerate(sims), key=lambda item: -item[1])[:10]
'''















