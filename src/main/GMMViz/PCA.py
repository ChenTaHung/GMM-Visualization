import numpy as np
import pandas as pd

class PCA:
    def __init__(self,n_components = 3):
        self.n_components = n_components
        
        # PCA components
        self.components = None
        self.mean = None
        
        # SVD components
        self.SVD_components = None

    def _PCAsvd(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Calculate the covariance matrix and ensure it's symmetric
        cov_matrix = np.cov(X_centered.T)
        cov_matrix = (cov_matrix + cov_matrix.T) / 2 + np.eye(cov_matrix.shape[0]) * 1e-10  # Regularization to avoid numerical instability

        # Use SVD to calculate the eigenvectors
        U, S, V = np.linalg.svd(cov_matrix)
        self.SVD_components = {"U" : U, 
                               "S" : S, 
                               "V" : V}
        # Select the top n_components eigenvectors (SVD returns them sorted)
        self.components = U[:, :self.n_components]

    def _PCAtransform(self, X) -> np.ndarray:
        # Center the data
        X_centered = X - self.mean

        # Project the data onto the principal components
        return np.dot(X_centered, self.components)
    
    def fit(self, X):
        if X is not None:
            if isinstance(X, np.ndarray):
                self.data = X
            elif isinstance(X, pd.DataFrame):
                self.data = X.values
            else:
                raise ValueError("The dataset should be either a numpy array or a pandas dataframe.")
            
        self._PCAsvd(self.data)
        X_transformed = self._PCAtransform(self.data)
        
        return X_transformed