import numpy as np
from sklearn.linear_model import ridge_regression
from sklearn.base import RegressorMixin, BaseEstimator
import warnings
from sklearn.exceptions import ConvergenceWarning


class STLSQ(BaseEstimator, RegressorMixin):
    def __init__(self, threshold=1, max_iter=100):
        self.threshold = threshold
        self.max_iter = max_iter
        
    def fit(self, data, target):
        if len(target.shape) == 1:
            target = target[:, np.newaxis]
        n_targets = target.shape[1]
        coef = np.zeros((n_targets, data.shape[1]))
        ind = np.ones((n_targets, data.shape[1]), dtype=bool)
        
        for _ in range(self.max_iter):
            end = True
            for i in range(n_targets):
                if ind[i].sum() == 0:
                    coef = np.zeros((n_targets, data.shape[1]))
                    warnings.warn('All coefficients were thresholded, try lower threshold parameter', ConvergenceWarning)
                    break
                coef_i = ridge_regression(data[:, ind[i]], target[:, i], alpha=1)

                thresholded = np.abs(coef_i) < self.threshold
                if np.sum(thresholded) == 0:
                    continue
                else:
                    end = False

                coef_i[thresholded] = 0
                ind_i = np.logical_not(thresholded)
                
                coef[i, ind[i]] = coef_i
                ind[i, ind[i]] = ind_i
            if end:
                break
        
        if n_targets == 1:
            coef = coef[0]
        
        self.coef_ = coef
        self.inf_ = ind
        return self
    
    def predict(self, data):
        return data @ self.coef_.T
