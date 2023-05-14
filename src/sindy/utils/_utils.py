import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from scipy.integrate import solve_ivp


class GaussianNoise(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale
        self.rng_ = np.random.default_rng()
        
    def transform(self, x, y=None):
        return x + self.rng_.normal(loc=self.loc, scale=self.scale, size=x.shape)
    
    def fit(self, x, t=None):
        return self

    def fit_transform(self, x, t):
        return self.transform(x, t)


class Equation:
    def __init__(self, feature_library, coef):
        self.feature_library = feature_library
        self.coef = coef

    def __call__(self, t, u):
        data = self.feature_library.fit_transform(np.array(u).reshape(1, -1))
        return (data @ self.coef.T)[0]
    
    # TODO
    def __str__(self):
        pass


def get_true_coef(terms, feauture_library, input_feature_names='x'):
    if isinstance(input_feature_names, str):
        input_feature_names = [f"{input_feature_names}{i+1}" for i in range(len(terms))]
    
    input_features = np.zeros((1, len(terms)))
    features = feauture_library.fit_transform(input_features)[0]
    feature_names = feauture_library.get_feature_names_out(input_feature_names)
    feature_names = {name: i for i, name in enumerate(feature_names)}
    
    coef = np.zeros((len(terms), len(features)), dtype=int)
    for i, eq in enumerate(terms):
        for term in eq:
            coef[i, feature_names[term]] = 1
    return coef


def gen_data(f, time, samples=100, u0=None, args=[], recurrent=False, return_derivative=False):
    if recurrent:  # recurrent
        assert(u0 is not None and isinstance(time, int))
        t = np.arange(time)
        data = [u0]
        for _ in range(time-1):
            data.append(f(data[-1], *args))
        data = np.array(data)
    elif u0 is not None:  # ode
        if (isinstance(time, np.ndarray)):
            t = time
        else:
            t = np.linspace(*time, int(samples * (time[1] - time[0])))
        res = solve_ivp(f, (t[0], t[-1]), u0, t_eval=t, args=args)
        data = np.stack(res.y, axis=-1)
        if return_derivative:
            der = np.array([f(i, j) for i, j in zip(res.t, data)])
    else:  # regular function
        if (isinstance(time, np.ndarray)):
            t = time
        else:
            t = np.linspace(*time, int(samples * (time[1] - time[0])))
        data = np.array([f(i) for i in t])
        
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    
    if not recurrent and u0 is not None and return_derivative:
        return data, t, der
    
    return data, t
