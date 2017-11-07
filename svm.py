import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import read_data


X_train, y_train = read_data.get_traindata()
X_val, y_val = read_data.get_valdata()
X_train_and_val = np.concatenate((X_train, X_val))
y_train_and_val = np.concatenate((y_train, y_val))


#basic SVM model

basic_svm = LinearSVC()
basic_svm.fit(X_train, y_train)
print("Basic SVM Score: ", basic_svm.score(X_val, y_val))


# L1 regularized SVM 

svm_lasso = LinearSVC(penalty= "l1", dual=False)
svm_lasso.fit(X_train, y_train)
print("SVM with L1 regularization Score: ", svm_lasso.score(X_val, y_val))


# tuning hyperparameters

param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000], 
              'penalty': ['l1', 'l2'],
              'loss': ['hinge', 'squared_hinge'],
              'dual': [True, False]
             }

basic_svm_2 = LinearSVC()
svm_tune = GridSearchCV(basic_svm_2, param_grid, error_score=0.0, verbose=False)
svm_tune.fit(X_train_and_val, y_train_and_val)
print(svm_tune.best_estimator_)
print("Best score: ", svm_tune.best_score_)
print("Best C: ", svm_tune.best_estimator_.C)
print("Best penalty: ", svm_tune.best_estimator_.penalty)
print("Best loss: ", svm_tune.best_estimator_.loss)
print("Best dual: ", svm_tune.best_estimator_.dual)














