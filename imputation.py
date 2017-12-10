from read_data import get_split_data
import pandas as pd
import numpy as np
from statsmodels.imputation import mice
import statsmodels.api as sm
from sklearn.preprocessing import Imputer

np.set_printoptions(threshold=10000)

X_train, Y_train, X_val, Y_val, X_test, Y_test, headers = get_split_data()
np.place(X_train, X_train==99.99, [np.nan])

#### MICE ####


# data = pd.DataFrame(clean_X)
# data.columns = headers
# print("columns: ", data.columns)
# imp = mice.MICEData(data)


# print(imp)
# model=mice.MICE(model_formula=None, model_class=sm.OLS, data=imp)
# print(imp.data)
# results=model.fit()
# print(results.summary())


#### Sklearn Imputer ####

imp = Imputer(strategy='mean', axis=0)
X_train_new = imp.fit_transform(X_train)
X_val_new = imp.fit_transform(X_val)
X_test_new = imp.fit_transform(X_test)


def get_imputed_traindata():
	return X_train_new, Y_train

def get_imputed_valdata():
	return X_val_new, Y_val

def get_imputed_testdata():
	return X_test_new, Y_test
