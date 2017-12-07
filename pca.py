import numpy as np
from sklearn.decomposition import PCA












def apply_pca(X, num_components):
	'''
	return the transformed matrix
	''' 

	
	pca = PCA(n_components=num_components)
	X_transform = pca.transform(X)

	return X_transform




	



	



