import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from read_data import *


def apply_pca(X, num_components):
	'''
	return the transformed matrix
	''' 

	
	pca = PCA(n_components=num_components)
	X_transform = pca.fit_transform(X)

	return X_transform

def apply_SparsePCA(X, num_components, alpha=1 ,ridge_alpha=0.1):
    
    sparse_pca = SparsePCA(n_components=num_components , alpha = alpha, ridge_alpha = ridge_alpha)
    X_transform = sparse_pca.fit_transform(X)
    
    return X_transform


X_train, Y_train = get_traindata() 
X,Y = clean_data(.95,X_train,Y_train)

print( apply_SparsePCA(X,10).shape )


	



	



