from sklearn import linear_model
from functools import partial


Lasso = partial(linear_model.Lasso, fit_intercept=False)
