# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:11:18 2017

@author: Amir
"""
import numpy as np
from read_data import *
from scipy.spatial.distance import cosine

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

def test(X_train, Y_train, X_test, Y_test):
    error= 0
    for i in range(len(X_test)):
        if classify(X_test[i] , X_train, Y_train) != Y_test[i]:
            error +=1
            
    return 1 - error*1.0 / len(X_test)
    


X_train, Y_train, X_val, Y_val, X_test, Y_test, headers = get_split_data()
print( test(X_train, Y_train, X_test, Y_test) )



