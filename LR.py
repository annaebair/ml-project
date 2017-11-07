from sklearn import linear_model as lm
from sklearn import grid_search as gs
import numpy as np
from read_data import *
from math import *
from plotBoundary import *

#X_train, Y_train = get_traindata() 
print X_train.shape
print Y_train.shape
p = 'l2'
LR=lm.LogisticRegression(penalty=p, C=1.0, max_iter=100, intercept_scaling=100, random_state=1)
#param = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
#clf = gs.GridSearchCV(LR, param)
LR.fit(X_train,Y_train.ravel())

print len(LR.coef_.ravel())

def predictLR(x):
    print x
    return LR.coef_.ravel() * mat(x).T + LR.intercept_

plotDecisionBoundary(X_train, Y_train, predictLR, [0], title = 'LR Train')
pl.show()


