import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .utils import Equation


class SINDy(BaseEstimator):
    def __init__(self, optimizer, differentiator, feature_library):
        self.optimizer = optimizer
        self.differentiator = differentiator
        self.feature_library = feature_library

    def fit(self, data, t, *, target=None, feature_names='x'):
        assert(target is None or len(target) == len(data))

        if isinstance(feature_names, str):
            self.feature_names_ = [f"{feature_names}{i+1}" for i in range(data.shape[1])]
        else:
            assert(len(feature_names) == data.shape[1])
            self.feature_names_ = feature_names

        features = self.feature_library.fit_transform(data)
        if target is None:
            target = self.differentiator.fit_transform(data, t)
        self.optimizer.fit(features, target)

        self.rhs_names_ = self.feature_library.get_feature_names_out(self.feature_names_)
        self.lhs_names_ = self.differentiator.get_feature_names_out(self.feature_names_) if target is None else \
            ['d' + i for i in self.feature_names_]
        self.coef_ = self.optimizer.coef_

        return self

    def predict(self, data):
        check_is_fitted(self)
        features = self.feature_library.fit_transform(data)
        return self.optimizer.predict(features)

    def get_equation(self):
        check_is_fitted(self)
        return Equation(self.feature_library, self.optimizer.coef_)

    def print_equation(self, *, fmt='.2g', threshold=1e-8):
        check_is_fitted(self)
        res = ''
        fmt_string = '{' + f':{fmt}' + '}'
        for i, l in enumerate(self.lhs_names_):
            res += f"{l} = {' + '.join([f'{fmt_string.format(self.coef_[i][j])} * {self.rhs_names_[j]}' for j, a in enumerate(np.abs(self.coef_[i]) > threshold) if a])}\n"
        return res

    def set_params(self, **params):
        optim_params = dict()
        dif_params = dict()
        lib_params = dict()
        
        for k, v in params.items():
            if k.startswith('optim__'):
                optim_params[k[7:]] = v
            elif k.startswith('dif__'):
                dif_params[k[5:]] = v
            elif k.startswith('lib__'):
                lib_params[k[5:]] = v
            else:
                raise ValueError(f"Invalid parameter {k!r} for estimator {self}.")

        self.optimizer.set_params(**optim_params)
        self.differentiator.set_params(**dif_params)
        self.feature_library.set_params(**lib_params)
        return self

    def get_params(self, deep=False):
        params = dict()
        for k, v in self.optimizer.get_params().items():
            params['optim__' + k] = v

        for k, v in self.differentiator.get_params().items():
            params['dif__' + k] = v

        for k, v in self.feature_library.get_params().items():
            params['lib__' + k] = v

        return params
