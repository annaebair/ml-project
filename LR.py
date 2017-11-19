from sklearn import linear_model 
from sklearn.model_selection import GridSearchCV
import numpy as np
from read_data import *
from math import *
from plotBoundary import *

X_train, y_train = get_traindata() 
X_val, y_val = get_valdata()

X_train_and_val = np.concatenate((X_train, X_val))
y_train_and_val = np.concatenate((y_train, y_val))

print(X_train.shape)
print(y_train.shape)

LR = linear_model.LogisticRegression()

LR.fit(X_train,y_train)

print("LR Score: ", LR.score(X_val, y_val))


params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
          'penalty': ['l1', 'l2']
         }
clf = GridSearchCV(LR, params)
clf.fit(X_train_and_val, y_train_and_val)

print(clf.best_estimator_)
print("Best score: ", clf.best_score_)
print("Best C: ", clf.best_estimator_.C)
print("Best penalty: ", clf.best_estimator_.penalty)



'''
This takes a while to run; here are the results I got:
LR score: 0.694444444444
Output of GridSearchCV:

Best score:  0.826570048309
Best C:  0.1
Best penalty:  l1
'''