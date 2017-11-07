import numpy as np
from sklearn import linear_model
import read_data




# load training and val data
X_train, y_train = read_data.get_traindata()
X_val, y_val = read_data.get_valdata()

# define and fit a basic SVM model
svm = LinearSVC(random_state=0)
svm.fit(X_train, y_train)

# print accuracy on validation data of above SVM 
print svm.score(X_val, y_val)




# L1 regularized SVM 
svm_lasso = LinearSVC(random_state=0, 'penalty': "l1")
svm_lasso.fit(X_train, y_train)

# print accuracy on validation data of above SVM 
print svm_lasso.score(X_val, y_val)




# tuning hyperparameters

param_grid = [
  {'C': [1, 10, 100, 1000],},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
 ] 

svm = LinearSVC(random_state=0)
# tune using GridSearch and cross validation set
svm_tune= GridSearchCV(svm, param_grid, 'score': svm_tune.score(X_val, y_val))

# print best found results
print svm_tune.score(X_val, y_val)














