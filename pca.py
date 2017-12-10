import numpy as np
from sklearn.decomposition import PCA
from read_data import *
from imputation import *


def apply_pca(X, num_components):

	pca = PCA(n_components=num_components)
	X_transform = pca.fit_transform(X)

	return X_transform


X_train, Y_train = get_imputed_traindata() 
# X,Y = clean_data(.95,X_train,Y_train)

print( apply_pca(X_train, 10).shape )
