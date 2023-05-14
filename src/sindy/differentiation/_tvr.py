import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class TVR(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.01, max_iter=2):
        self.alpha = alpha
        self.max_iter = max_iter

    def diff_matr(self, n, h):
        D = np.zeros((n, n+1))
        for i in range(n):
            D[i, i] = -1
            D[i, i+1] = 1
        D /= h
        return D

    def integr_matr(self, n, h):
        A = np.zeros((n, n+1))
        for i in range(n):
            for j in range(i+1):
                A[i, j] = 1
            A[i, 0] = 0.5
            A[i, i+1] = 0.5
        A *= h
        return A

    def transform(self, x, t):
        unsqueeze = False
        if len(x.shape) == 1:
            unsqueeze = True
            x = x[:, np.newaxis]

        n = len(x)
        h = t[1] - t[0]
        D = self.diff_matr(n, h)
        A = self.integr_matr(n, h)

        res = np.zeros_like(x)
        for i in range(x.shape[1]):
            u = np.full(n+1, 0.0)
            for _ in range(self.max_iter):
                E = np.diag(1 / np.sqrt((D @ u) ** 2 + 1e-6))

                L = h * D.T @ E @ D
                H = A.T @ A + self.alpha * L
                g = A.T @ A @ u - A.T @ (x[:, i] - x[0, i]) + self.alpha * L @ u
                s = np.linalg.solve(H, -g)

                u += s
            res[:, i] = u[1:]

        if unsqueeze:
            res = res.squeeze()
        return res

    def fit(self, x, t=None):
        return self

    def fit_transform(self, x, t):
        return self.transform(x, t)

    def get_feature_names_out(self, input_features):
        return ['d' + i for i in input_features]


class FastTVR(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def diff_matr(self, n, h):
        D = np.zeros((n, n+1))
        for i in range(n):
            D[i, i] = -1
            D[i, i+1] = 1
        D /= h
        return D

    def integr_matr(self, n, h):
        A = np.zeros((n, n+1))
        for i in range(n):
            for j in range(i+1):
                A[i, j] = 1
            A[i, 0] = 0.5
            A[i, i+1] = 0.5
        A *= h
        return A

    def transform(self, x, t):
        unsqueeze = False
        if len(x.shape) == 1:
            unsqueeze = True
            x = x[:, np.newaxis]

        n = len(x)
        h = t[1] - t[0]
        D = self.diff_matr(n, h)
        A = self.integr_matr(n, h)

        res = np.zeros_like(x)
        for i in range(x.shape[1]):
            u = x[:, i] - x[0, i]
            B = A.T @ A + self.alpha * D.T @ D
            b = A.T @ u

            s = np.linalg.solve(B, b)
            res[:, i] = s[1:]

        if unsqueeze:
            res = res.squeeze()
        return res

    def fit(self, x, t=None):
        return self

    def fit_transform(self, x, t):
        return self.transform(x, t)

    def get_feature_names_out(self, input_features):
        return ['d' + i for i in input_features]
