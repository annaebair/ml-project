# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:11:18 2017

@author: Amir
"""
import numpy as np
from read_data import *
from scipy.spatial.distance import cosine
from scipy import stats

np.set_printoptions(threshold=10000)

def cos_dist(X1 , X2 ):
    return cosine(X1,X2)

def classify( X , X_train, Y_train):
    index = 0
    min_dist = 1
    for i in range(len(X_train)):
        dist = cos_dist(X_train[i],X)
        if dist < min_dist:
            index = i 
            min_dist = dist
            
    return Y_train[index]

def multi_classify(X, X_train, Y_train):
    indices = []
    min_dist = 1
    for i in range(len(X_train)):
        dist = cos_dist(X_train[i], X)
        indices.append((i, dist))

    indices = sorted(indices, key = lambda x: x[1])
    k = 5
    top_k = indices[:5]
    ys = []
    for i in range(len(top_k)):
        index, dist = top_k[i]
        ys.append(Y_train[index])
    return stats.mode(ys)[0][0]


def test(X_train, Y_train, X_test, Y_test):
    error= 0
    for i in range(len(X_test)):
        if multi_classify(X_test[i] , X_train, Y_train) != Y_test[i]:
            error +=1
            
    return 1 - error*1.0 / len(X_test)
    


X_train, Y_train, X_val, Y_val, X_test, Y_test, headers = get_split_data()
# print(Y_train)
neg = 0
pos = 0
for i in Y_train:
    if i == 1:
        pos += 1
    else:
        neg += 1
print("neg: ", neg)
print("pos: ", pos)
print("VAL: ")
for i in Y_val:
    if i == 1:
        pos += 1
    else:
        neg += 1
print("neg: ", neg)
print("pos: ", pos)
print("TEST: ")
for i in Y_test:
    if i == 1:
        pos += 1
    else:
        neg += 1
print("neg: ", neg)
print("pos: ", pos)
print( test(X_train, Y_train, X_test, Y_test) )




