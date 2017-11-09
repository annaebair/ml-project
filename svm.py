import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import read_data





def train_basicSVM():

	#basic SVM model
	print ("------training basic SVM L2 model----------")

	basic_svm = LinearSVC()
	basic_svm.fit(X_train, y_train)
	print("Basic SVM Score: ", basic_svm.score(X_val, y_val))


def train_L1SVM():
	# L1 regularized SVM 
	print ("------training SVM L1 model----------")
	svm_lasso = LinearSVC(penalty= "l1", dual=False)
	svm_lasso.fit(X_train, y_train)
	print("SVM with L1 regularization Score: ", svm_lasso.score(X_val, y_val))



def tune_SVM():
	# tuning hyperparameters
	print ("-------- tuning hyperparameters---------")

	param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000], 
	              'penalty': ['l1', 'l2'],
	              'loss': ['hinge', 'squared_hinge'],
	             'dual':[False, True]}

	basic_svm_2 = LinearSVC()
	svm_tune = GridSearchCV(basic_svm_2, param_grid, error_score = 0.0)
	svm_tune.fit(X_train_and_val, y_train_and_val)
	print(svm_tune.best_estimator_)
	print("Best score: ", svm_tune.best_score_)
	print("Best C: ", svm_tune.best_estimator_.C)
	print("Best penalty: ", svm_tune.best_estimator_.penalty)
	print("Best loss: ", svm_tune.best_estimator_.loss)




if __name__ == "__main__":

    print ("--------------- LOADING DATA -------------------")
    X_train, y_train = read_data.get_traindata()
    X_val, y_val = read_data.get_valdata()
    X_train_and_val = np.concatenate((X_train, X_val))
    y_train_and_val = np.concatenate((y_train, y_val))

    print ("--------------- DATA IS LOADED -------------------")

    train_basicSVM()
    train_L1SVM()
    tune_SVM()













