from sklearn import linear_model


class Lasso(linear_model.Lasso):
    def __init__(self, alpha=1.0, **kwargs):
        kwargs['fit_intercept'] = False
        super(linear_model.Lasso, self).__init__(alpha, **kwargs)

    def set_params(self, **kwargs):
        super(linear_model.Lasso, self).set_params(**kwargs)

    def get_params(self, deep=False):
        return super(linear_model.Lasso, self).get_params(deep)
