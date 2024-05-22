import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from .KmeansPlus import Kmeans_Plus
from scipy.special import logsumexp
from .PCA import PCA

class GMM:
    """
    1. Initialization:
    - Choose the number of components (clusters) K.
    - Initialize the means, covariances, and mixing coefficients randomly.

    2. Expectation Step:
    - Compute the responsibilities using the current parameters.
    - The responsibility of a component k for a data point x is the probability of the component given the data point.

    3. Maximization Step:
    - Update the means, covariances, and mixing coefficients using the responsibilities.
    - The new mean of component k is the weighted average of all data points, where the weights are the responsibilities.
    - The new covariance of component k is the weighted covariance of all data points, where the weights are the responsibilities.                                                                                                              
    - The new mixing coefficient of component k is the fraction of responsibilities assigned to component k.

    4. Convergence:
    - Repeat steps 2 and 3 until the log-likelihood converges or the maximum number of iterations is reached.

    """
    def __init__(self, n_clusters, random_state = None, max_iter = 100, tol = 1e-4):
        
        self.data = None # assign in the fit method
        self.n_clusters = n_clusters
        self.random_state = random_state
        ### Parameters
        self.mean = None # Mean Array
        self.Sigma = None # Covariance Matrix
        self.pi = None # Mixing Coefficients (Latent Variable)
        
        # Estimation
        self.LogL = -np.inf # Log Likelihood for convergence check
        
        # Store the estimated parameters
        self.estimands_mean = []
        self.estimands_Sigma = []
        self.estimands_logLikelihood = []
        
        self.estimands = None
        
        ### Convergence
        self.max_iter = max_iter
        self.tol = tol
        # set the minimum value for the log-likelihood to avoid -inf
        self.min_val = np.finfo(float).tiny
    
    def _record_estimands(self):
        """
        Update the estimated parameters of the Gaussian Mixture Model.
        """
        mean = self.mean.copy()
        Sigma = self.Sigma.copy()
        self.estimands_mean.append(mean)
        self.estimands_Sigma.append(Sigma)
        
        self.estimands_logLikelihood.append(self.LogL)
        
    def _regularize_Sigma(self, eps = 1e-6) -> None:
        """ 
        To stablize the covariance matrix and make sure it is invertible, add a small value to the diagonal
        """
        for k in range(self.n_clusters):
            self.Sigma[k] += eps * np.eye(self.data.shape[1])
        
    def _init_params(self, random_state=None) -> None:
        """
        Initialize the parameters of the Gaussian Mixture Model.

        Parameters:
            random_state (int): Random seed for reproducibility. Defaults to None.

        Returns:
            None
        """
        if random_state:
            np.random.seed(random_state)

        # n: number of samples, p: number of features
        n, p = self.data.shape

        # get initialized centroids by Kmeans++
        centroids, cen_index = Kmeans_Plus(self.data, self.n_clusters).getCenter(random_state=random_state)
        # centroids = self.data[np.random.choice(n, self.n_clusters, replace=False)] 
        
        # use centroids as mean
        self.mean = centroids

        # initialize covariance matrix as identity matrix
        variances = np.var(self.data, axis=0)
        self.Sigma = np.array([np.diag(variances) for _ in range(self.n_clusters)])
        self._regularize_Sigma() # Regularize the covariance matrix
        
        # equal weights for each cluster
        self.pi = np.ones(self.n_clusters) / self.n_clusters
        
        # store the initial parameters
        self._record_estimands()
   
        
    def _Estep(self, X) -> np.ndarray:
        """
        In E-step, we introduce the latent variable z, which represents the cluster assignment of each data point.
        And the Z is corresponding to the responsibilities, which is the probability of the data point belonging to each cluster.
        """
        n, _ = X.shape
        
        # initialize responsibilities: n by k matrix
        log_resp = np.zeros((n, self.n_clusters))
        
        for k in range(self.n_clusters):
            # calculate the responsibilities in log-space
            pi_k = np.maximum(self.pi[k], self.min_val) # avoid log(0) = -inf
            log_resp[:, k] = np.log(pi_k) + multivariate_normal.logpdf(X, self.mean[k].A1, self.Sigma[k],) # A1: return a flattened array
            
        log_resp = log_resp - logsumexp(log_resp, axis = 1, keepdims = True) # normalize the responsibilities
        
        resp = np.exp(log_resp)
        
        return resp
    
    def _Mstep(self, resp: np.ndarray) -> None:
        """
        Performs the M-step of the Gaussian Mixture Model algorithm.

        Parameters:
        - resp: numpy.ndarray
            The responsibilities matrix of shape (n_samples, n_clusters).

        Returns:
        None
        """

        n, p = self.data.shape
        # sum of responsibilities
        nK = np.sum(resp, axis=0)

        # update mixing coefficients (latent variable)
        self.pi = nK / n

        for k in range(self.n_clusters):
            # update mean
            weighted_sum = np.sum(resp[:, k, np.newaxis] * self.data, axis=0)
            self.mean[k] = weighted_sum / nK[k]

            # update covariance matrix
            diff = self.data - self.mean[k]  # deviation of data from current mean
            # for i in range(n):
            #     vec = diff[i, :, np.newaxis]
            #     self.Sigma[k] += resp[i, k] * np.dot(vec, vec.T)
                
            # self.Sigma[k] /= nK[k]  # normalize by the sum of responsibilities

            outer_products = np.einsum('bi,bj->bij', diff, diff)  # shape(n, p, p)
            weighted_sum = np.einsum('b,bij->ij', resp[:, k], outer_products)  # shape(p, p)
            
            self.Sigma[k] = weighted_sum / nK[k]
            self._regularize_Sigma()
            
    def _cal_log_likelihood(self, X, resp):
        """
        Calculate the log likelihood of the Gaussian Mixture Model.
        Q function: Sum of each data, and each cluster, r_ij (log pi_k + log N(x_i|mu_k, Sigma_k))
        
        Parameters:
        - X (numpy.ndarray): The input data points.
        - resp (numpy.ndarray): The responsibilities matrix of shape (n_samples, n_clusters).

        Returns:
        - log_likelihood (float): The log likelihood of the Gaussian Mixture Model.
        """
        Q = 0 # Q function 
        for k in range(self.n_clusters):
            pi_k = np.maximum(self.pi[k], self.min_val)
            log_resp = np.log(pi_k) + multivariate_normal.logpdf(X, self.mean[k].A1, self.Sigma[k], allow_singular=True)
            Q +=  np.sum(resp[:, k] * log_resp)
            
        return Q
    
    def fit(self, X):
        """
        Fits the Gaussian Mixture Model to the data.
        
        This method initializes the parameters, performs the Expectation-Maximization (EM) algorithm,
        and checks for convergence based on the change in log likelihood.

        Returns:
            None
        """
        if isinstance(X, np.ndarray):
            self.data = X
        elif isinstance(X, pd.DataFrame):
            self.data = X.values
        else:
            raise TypeError("The dataset should be either a numpy array or a pandas dataframe.")
        
        self._init_params(random_state=self.random_state)
        final_iter = None
        for i in range(self.max_iter):
            resp = self._Estep(X=self.data)
        
            self._Mstep(resp)

            # calculate log likelihood
            log_likelihood = self._cal_log_likelihood(X=self.data, resp = resp)
            
            # check convergence
            if np.abs(log_likelihood - self.LogL) < self.tol:
                print('Converged at iteration', i + 1)
                self.LogL = log_likelihood # update the log likelihood
                final_iter = i + 1
                break
            
            self.LogL = log_likelihood
            # store and append the parameters in each iteration
            self._record_estimands()
            
        
        if final_iter is None: # no early stop
            final_iter = self.max_iter
        
        # store the number of iterations
        self.n_iter_ = final_iter
    
    def PCA_fit(self, X, n_components = 3):
        """
        Choosing the n components by PCA and fit the Gaussian Mixture Model to the data.
        
        Parameters:
        X (numpy.ndarray or pandas.DataFrame): The dataset to fit the model on.
        n_components (int): The number of components to choose by PCA.
        
        Raises:
        TypeError: If the dataset is not a numpy array or a pandas dataframe.
        """
        if isinstance(X, np.ndarray):
            self.data = X
        elif isinstance(X, pd.DataFrame):
            self.data = X.values
        else:
            raise TypeError("The dataset should be either a numpy array or a pandas dataframe.")
        
        pca = PCA(n_components = n_components)
        pca_transformed_df = pca.fit(self.data)
        
        self.fit(pca_transformed_df)
        
            
    def predict(self, X, return_prob=False):
        """
        Predicts the cluster labels for the given data points.

        Parameters:
        - X (array-like): The input data points to be predicted.
        - return_prob (bool): If True, returns the probability of each data point belonging to each cluster.
                                                    If False, returns the predicted cluster labels.

        Returns:
        - If return_prob is True, returns an array of shape (n_samples, n_clusters) containing the probability of each
            data point belonging to each cluster.
        - If return_prob is False, returns an array of shape (n_samples,) containing the predicted cluster labels.
        """
        if isinstance(X, np.ndarray):
            X = X
        elif isinstance(X, pd.DataFrame):
            X = X.values
        else:
            raise ValueError("The dataset should be either a numpy array or a pandas dataframe.")
        
        resp = self._Estep(X)
        
        if return_prob:
            return resp
        else:
            return np.argmax(resp, axis=1)

    def getEstimands(self, parm = None):
        
        if parm is None:
            # store the estimated parameters
            estimands = {
                "mean": self.estimands_mean,
                "Sigma": self.estimands_Sigma,
                "log_likelihood": self.estimands_logLikelihood
            }    
            return estimands
        
        elif parm == 'mean':
            return self.estimands_mean
        elif parm == 'Sigma':
            return self.estimands_Sigma
        elif parm == 'log_likelihood':
            return self.estimands_logLikelihood
        else:
            raise ValueError("The parameter should be either 'mean', 'Sigma', or 'log_likelihood'.")
