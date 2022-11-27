import random

import numpy as np
from scipy import spatial
from scipy import stats
from scipy import mean




class KNN:
    """
    Implementation of the k-nearest neighbors algorithm for classification
    and regression problems.
    """
    def __init__(self, k, aggregation_function):
        """
        Takes two parameters.  k is the number of nearest neighbors to use
        to predict the output variable's value for a query point. The
        aggregation_function is either "mode" for classification or
        "average" for regression.
        
        Parameters
        ----------
        k : int
           Number of neighbors
        
        aggregation_function : {"mode", "average"}
           "mode" : for classification
           "average" : for regression.
        """
        self.k = k
        self.aggregation_function = aggregation_function
        self.X = None
        self.y = None
        
    def fit(self, X, y):
        """
        Stores the reference points (X) and their known output values (y).
        
        Parameters
        ----------
        X : 2D-array of shape (n_samples, n_features) 
            Training/Reference data.
        y : 1D-array of shape (n_samples,) 
            Target values.
        """
        self.X = X
        self.y = y
        
    def predict(self, X):
        """
        Predicts the output variable's values for the query points X.
        
        Parameters
        ----------
        X : 2D-array of shape (n_queries, n_features)
            Test samples.
            
        Returns
        -------
        y : 1D-array of shape (n_queries,) 
            Class labels for each query.
        """
        # Calculates the distances between each X and each point in self.X, 
        # returning a matrix of size len(X), len(self.X)
        distances = spatial.distance.cdist(X, self.X) 
        
        # Sorts the lowest computed distances by traversing accross distance matrix from step before
        # returning an array the size len(X), self.k
        sorted_indexes = np.argsort(distances, axis=1).transpose()[:self.k].transpose() 
        
        # Returns the associated plant for the indexes that were sorted above 
        # returning an array of size len(X), self.k
        targets = self.y[sorted_indexes] 
        

        # Returns a 1-D array of predictions 
        # Returns array length X with the predictions for each row of X
        if self.aggregation_function == 'mode':
            return stats.mode(targets, axis=1, keepdims=False)[0]
        else:
            print(f'Mean: {np.mean(targets, axis=1)}')
            return np.mean(targets, axis=1)



