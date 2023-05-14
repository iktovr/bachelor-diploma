import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class FiniteDifference(BaseEstimator, TransformerMixin):
    def transform(self, x, t):
        assert(len(x) > 2)
        h = t[1] - t[0]
        d = np.zeros_like(x)
        d[0] = (- 3 * x[0] + 4 * x[1] - x[2]) / (2 * h)
        d[-1] = (x[-3] - 4 * x[-2] + 3 * x[-1]) / (2 * h)
        for i in range(1, len(x)-1):
            d[i] = (x[i+1] - x[i-1]) / (2 * h)
        return d

    def fit(self, x, t=None):
        return self

    def fit_transform(self, x, t):
        return self.transform(x, t)

    def get_feature_names_out(self, input_features):
        return ['d' + i for i in input_features]
