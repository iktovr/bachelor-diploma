import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, r2_score
from IPython.display import clear_output
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from ._utils import Equation, GaussianNoise, get_true_coef, gen_data
from .metrics import iou_score, me_score
from ..differentiation import FiniteDifference


# common
def plot_data(data, t, *, ax=None, xlim=None, ylim=None, zlim=None, azim=None, figsize=None, ax_kwargs=dict(), **plot_kwargs):
    args = dict(linestyle='-', linewidth=0.2, marker='.', markersize=1)
    args.update(plot_kwargs)

    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=figsize)
        if len(data.shape) != 1 and data.shape[1] > 2:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.gca()
    ax.set(**ax_kwargs)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    
    if len(data.shape) == 1:
        ax.plot(t, data, **args)
    elif data.shape[1] == 1:
        ax.plot(t, data[:, 0], **args)
    elif data.shape[1] == 2:
        ax.plot(data[:, 0], data[:, 1], **args)
    else:
        ax.view_init(azim=azim)
        ax.plot(data[:, 0], data[:, 1], data[:, 2], **args)

    if standalone:
        plt.show()


# sindy
def plot_error_over_time_noise(data, t, u0, coefs, feature_library, noise_values, *, metric=mean_squared_error, ax=None):
    standalone = ax is None
    if standalone:
        fig = plt.figure()
        ax = fig.gca()
        
    ax.set(yscale='log', xlabel='t', ylabel='error')
    
    for coef, noise in zip(coefs, noise_values):        
        eq = Equation(feature_library, coef)
        res = solve_ivp(eq, (t[0], t[-1]), u0, t_eval=t)
        pred = np.stack(res.y, axis=-1)
        
        error = [metric(gt, p) for (gt, p) in zip(data, pred)]
        
        ax.plot(t, error, label=f'{noise}')
    
    ax.legend(title='noise', loc='lower right')
    
    if standalone:
        plt.show()


def plot_iou_over_noise(coefs, feature_library, terms, noise_values, *, input_feature_names='x', threshold=1e-8, ax=None):
    gt = get_true_coef(terms, feature_library, input_feature_names).flatten()
    
    standalone = ax is None
    if standalone:
        fig = plt.figure()
        ax = fig.gca()
        
    ax.set(xlabel='noise', ylabel='iou', xticks=noise_values)
    ax.set_xscale('symlog', linthresh=noise_values[1])
    
    iou = []
    for coef in coefs:
        iou.append(iou_score(gt, coef, threshold))
        
    ax.plot(noise_values, iou, marker='.')
    
    if standalone:
        plt.show()


def plot_missing_extra_terms_over_noise(coefs, feature_library, terms, noise_values, *, input_feature_names='x', threshold=1e-8, ax=None):
    gt = get_true_coef(terms, feature_library, input_feature_names).flatten()
    
    standalone = ax is None
    if standalone:
        fig = plt.figure()
        ax = fig.gca()
        
    ax.set(xlabel='noise', xticks=noise_values)
    ax.set_xscale('symlog', linthresh=noise_values[1])
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    
    missing = np.zeros(len(coefs))
    extra = np.zeros(len(coefs))
    for i, coef in enumerate(coefs):
        missing[i], extra[i] = me_score(gt, coef, threshold)
        
    ax.plot(noise_values, missing, label='missing terms', marker='.')
    trans_offset = matplotlib.transforms.offset_copy(ax.transData, y=5, units='dots')
    ax.plot(noise_values, extra, label='extra terms', transform=trans_offset, marker='.')
    ax.set_ylim(-0.25, max(np.max(missing), np.max(extra))+0.25)
    ax.legend(loc='upper left')
    
    if standalone:
        plt.show()


def plot_results(data, t, u0, terms, models, noise_values, arguments, *, size=(3.5, 2.5), return_coefs=False, verbose=False, **plot_data_kwargs):
    assert(len(models) == len(arguments))
    for args in arguments:
        assert(len(args) == 0 or len(args) == len(noise_values))
        
    true_coef = get_true_coef(terms, models[next(iter(models))].feature_library)
    
    w = len(noise_values) + 1
    h = len(models) + 1
    fig = plt.figure(figsize=(w * size[0], h * size[1]))
    gs = GridSpec(h, w, figure=fig, width_ratios=[1e-6] + [1 / w] * (w-1))
    
    subplot_args = dict()
    if data.shape[1] == 3:
        subplot_args['projection'] = '3d'

    if return_coefs:
        coefs = [[] for _ in range(len(models))]
    
    rng = np.random.default_rng()
    for i, noise in enumerate(noise_values):
        noise_data = data + rng.normal(scale=noise, size=data.shape)
        ax = fig.add_subplot(gs[0, i+1], **subplot_args)
        ax.set_title(f'Noise: {noise:.2g}')
        plot_data(noise_data, t, ax=ax, **plot_data_kwargs)
        
        for j, ((name, model), args) in enumerate(zip(models.items(), arguments), 1):
            if i == 0:
                ax = fig.add_subplot(gs[j, 0])
                ax.set(xticks=[], yticks=[], frame_on=False)
                ax.text(1, 0.5, name, fontsize=12, rotation=90, ha='center', va='center')
                
            if len(args) != 0:
                model.set_params(**args[i])
            model.fit(noise_data, t)
            coef = model.optimizer.coef_
            iou = iou_score(true_coef, coef)

            if return_coefs:
                coefs[j-1].append(coef)

            equation = model.get_equation()
            pred, _ = gen_data(equation, t, u0=u0)
            
            ax = fig.add_subplot(gs[j, i+1], **subplot_args)
            ax.set_title(f'IoU: {iou:.2g}')
            plot_data(pred, t, ax=ax, **plot_data_kwargs)
        
        if verbose:
            print(f'Noise value {i+1}/{len(noise_values)} ({noise}): Done')
    if verbose:
        clear_output()
        
    fig.tight_layout()
    plt.show()

    if return_coefs:
        return coefs


def plot_sindy_scores(terms, coefs, feature_library, noise_values, *, input_feature_names='x', threshold=1e-8):
    fig = plt.figure(figsize=(14, 4))
    axes = fig.subplots(1, 2)

    plot_missing_extra_terms_over_noise(coefs, feature_library, terms, noise_values, input_feature_names=input_feature_names, threshold=threshold, ax=axes[0])
    plot_iou_over_noise(coefs, feature_library, terms, noise_values, input_feature_names=input_feature_names, threshold=threshold, ax=axes[1])
    
    fig.subplots_adjust(wspace=0.15)
    plt.show()


def plot_sindy_scores_all(data, t, u0, terms, coefs, feature_library, noise_values, *, input_feature_names='x', metric=mean_squared_error, threshold=1e-8):
    fig = plt.figure(figsize=(11, 6))
    
    plot_error_over_time_noise(data, t, u0, coefs, feature_library, noise_values, metric=metric, ax=fig.add_subplot(2, 2, (1, 2)))
    plot_missing_extra_terms_over_noise(coefs, feature_library, terms, noise_values, input_feature_names=input_feature_names, threshold=threshold, ax=fig.add_subplot(2, 2, 3))
    plot_iou_over_noise(coefs, feature_library, terms, noise_values, input_feature_names=input_feature_names, threshold=threshold, ax=fig.add_subplot(2, 2, 4))
    
    fig.tight_layout()
    plt.show()


# regression
def plot_regression_scores(model, data, target, *, ax=None, figsize=(5, 5)):
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, aspect='equal')
    
    pred = model.predict(data)

    scores = f"""\
MAE: {mean_absolute_error(target, pred):.4e}
MSE: {mean_squared_error(target, pred):.4e}
R^2: {r2_score(target, pred):.4f}"""

    if len(target.shape) == 1:
        scores = f'Max error: {max_error(target, pred):.4e}\n' + scores
    
    ax.set(xlabel='Actual values', ylabel='Predicted values')
    ax.axline((0, 0), (1, 1), ls='--', c='grey')
    ax.scatter(target, pred, marker='o')
    ax.text(np.min(target), np.max(pred), scores, ha='left', va='top')
    
    if standalone:
        plt.show()


def plot_coef(model, *, shape=(10, 10), ax=None, figsize=(5, 5), cbar_ax=None):
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, aspect='equal')
        
    if isinstance(model, np.ndarray):
        coef = model
    else:
        coef = model.coef_
    
    sns.heatmap(coef.reshape(shape), cmap='seismic', center=0, square=True, linewidth=0.5, linecolor='lightgray', 
                xticklabels=0, yticklabels=0, cbar_ax=cbar_ax, ax=ax)
    # ax.set_title('Coefficients')
    if standalone:
        plt.show()

    
def plot_iou_over_regression_parameter(data, t, terms, model, name, values, noise_value=0):
    noise_data = GaussianNoise(scale=noise_value).transform(data)
    deriv = model.differentiator.transform(noise_data, t)
    true_coef = get_true_coef(terms, model.feature_library)
    iou = []
    for value in values:
        model.optimizer.set_params(**{name: value})
        with ignore_warnings(category=ConvergenceWarning):
            model.fit(noise_data, t, target=deriv)
        iou.append(iou_score(true_coef, model.coef_))
    ax = plt.gca()
    ax.set(ylabel='IoU', xlabel=name, title=f'noise: {noise_value}')
    ax.plot(values, iou)
    plt.show()

def plot_iou_over_two_regression_parameters(data, t, terms, model, name1, values1, name2, values2, noise_value=0, **heatmap_kwargs):
    noise_data = GaussianNoise(scale=noise_value).transform(data)
    deriv = model.differentiator.transform(noise_data, t)
    true_coef = get_true_coef(terms, model.feature_library)
    iou = np.zeros((len(values2), len(values1)))
    for i, value1 in enumerate(values1):
        for j, value2 in enumerate(values2):
            model.optimizer.set_params(**{name1: value1, name2: value2})
            with ignore_warnings(category=ConvergenceWarning):
                model.fit(noise_data, t, target=deriv)
            iou[i, j] = iou_score(true_coef, model.coef_)
    ax = plt.gca()
    xticklabels=[values1[0]] + ['' for i in range(len(values1)-2)] + [values1[-1]]
    yticklabels=[values2[0]] + ['' for i in range(len(values2)-2)] + [values2[-1]]
    ax = sns.heatmap(iou, annot=False, xticklabels=xticklabels, yticklabels=yticklabels, cmap='Reds', linewidth=0, ax=ax, **heatmap_kwargs)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    ax.set(xlabel=name1, ylabel=name2, title=f'noise: {noise_value}')
    plt.show()
    return iou

def plot_iou_over_noise_arg(data, t, terms, model, name, values, noise_values, args, *, size=None, ax=None, verbose=False, **heatmax_kwargs):
    true_coef = get_true_coef(terms, model.feature_library)
    iou = np.zeros((len(noise_values), len(values)))
    for i, (noise, params) in zip(range(len(noise_values)-1, -1, -1), zip(noise_values, args)):
        model.set_params(**params)
        noise_data = GaussianNoise(scale=noise).transform(data)
        deriv = model.differentiator.transform(noise_data, t)
        for j, value in enumerate(values):
            model.optimizer.set_params(**{name: value})
            with ignore_warnings(category=ConvergenceWarning):
                model.fit(noise_data, t, target=deriv)
            iou[i, j] = iou_score(true_coef, model.coef_)
        
        if verbose:
            print(f'Noise value {len(noise_values)-i}/{len(noise_values)} ({noise}): Done')
    if verbose:
        clear_output()
    
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

    xticklabels=[values[0]] + ['' for i in range(len(values)-2)] + [values[-1]]
    ax = sns.heatmap(iou, annot=False, xticklabels=xticklabels, yticklabels=noise_values[::-1], cmap='Reds', ax=ax, **heatmax_kwargs)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    ax.set(xlabel=name, ylabel='noise')

    lowest = iou.argmax(1)
    highest = np.zeros_like(lowest)
    for i in range(len(iou)):
        for j in range(lowest[i]+1, len(iou[i])):
            if np.abs(iou[i][j] - iou[i][lowest[i]]) > 1e-5:
                highest[i] = j - 1
                break
        else:
            highest[i] = len(iou[i]) - 1

    ax.plot(lowest + 0.5, np.arange(len(noise_values)) + 0.5, linewidth=1.5, color='lightgray', linestyle='--')
    ax.plot(highest + 0.5, np.arange(len(noise_values)) + 0.5, linewidth=1.5, color='lightgray', linestyle='--')
    print(list(zip(values[lowest][::-1], values[highest][::-1])))

    if standalone:
        plt.show()


# differentiation
def plot_dif_test(f, time, samples, algs, *, noise=0, size=(4.5, 3.5), u0=None):
    """Тестирование различных методов на конкретной функции с заданным уровнем шума"""
    if isinstance(f, np.ndarray):
        t = np.linspace(*time, int(samples * (time[1] - time[0])))
        assert(len(f) == len(t))
        data = np.copy(f)
    else:
        data, t = gen_data(f, time, samples, u0)
    real_deriv = FiniteDifference().fit_transform(data, t)
    
    if noise > 0:
        data += np.random.default_rng().normal(scale=noise, size=data.shape)
    
    w, h = max(len(algs), 2), 2
    fig = plt.figure(figsize=(w * size[0], h * size[1]))
    
    args = dict()
    if data.shape[1] == 3:
        args['projection'] = '3d'
    
    ax = fig.add_subplot(h, w, 1, **args)
    ax.set_title(f'Data. Noise: {noise}')
    plot_data(data, t, ax=ax)
    
    ax = fig.add_subplot(h, w, 2, **args)
    ax.set_title('Derivative')
    plot_data(real_deriv, t, ax=ax)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim() if data.shape[1] == 3 else None
    
    for i, alg in enumerate(algs, w+1):
        ax = fig.add_subplot(h, w, i, **args)
        deriv = alg.fit_transform(data, t)
        error = mean_absolute_error(real_deriv, deriv)
        ax.set_title(f"{type(alg).__name__}. MAE: {error:.2g}")
        plot_data(deriv, t, xlim=xlim, ylim=ylim, zlim=zlim, ax=ax)
    
    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.show()


def plot_dif_test_arg(f, time, samples, alg, name, values, *, u0=None, noise=0, xscale='log', metric=mean_absolute_error, verbose=False, ax=None, rotate=False, **plot_kwargs):
    """Зависимость между ошибкой и параметром метода"""
    if isinstance(f, np.ndarray):
        t = np.linspace(*time, int(samples * (time[1] - time[0])))
        assert(len(f) == len(t))
        data = np.copy(f)
    else:
        data, t = gen_data(f, time, samples, u0)
    real_deriv = FiniteDifference().fit_transform(data, t)
    
    if noise > 0:
        data += np.random.default_rng().normal(scale=noise, size=data.shape)
        
    error = []
    for i, value in enumerate(values):
        alg.set_params(**{name: value})
        deriv = alg.fit_transform(data, t)
        error.append(metric(real_deriv, deriv))

        if verbose:
            print(f'Arg value {i+1}/{len(values)} ({value}): Done')
    if verbose:
        clear_output()
    
    standalone = ax is None
    if standalone:
        fig = plt.figure()
        ax = fig.gca()

    if xscale == 'symlog':
        ax.set_xscale(xscale, linthresh=np.partition(values, 1)[1])
    else:
        ax.set_xscale(xscale)
    ax.set(xlabel=name, ylabel='error', xticks=values)
    ax.get_xaxis().set_major_formatter('{x:n}')
    ax.plot(values, error, marker='.', **plot_kwargs)

    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if standalone:
        plt.show()


def plot_dif_test_samples(f, time, n_values, alg, *, u0=None, noise=0, metric=mean_absolute_error, ax=None, verbose=False, **plot_kwargs):
    """Зависимость между ошибкой и параметрами разбиения"""
    error = []
    for i, n in enumerate(n_values):
        data, t = gen_data(f, time, n, u0)
        real_deriv = FiniteDifference().fit_transform(data, t)

        if noise > 0:
            data += np.random.default_rng().normal(scale=noise, size=data.shape)
            
        deriv = alg.fit_transform(data, t)
        error.append(metric(real_deriv, deriv))

        if verbose:
            print(f'Samples value {i+1}/{len(n_values)} ({n}): Done')
    if verbose:
        clear_output()

    standalone = ax is None
    if standalone:
        fig = plt.figure()
        ax = fig.gca()

    ax.set(xlabel='samples per second', ylabel='error', xticks=n_values)
    ax.plot(n_values, error, marker='.', **plot_kwargs)

    if standalone:
        plt.show()


def plot_dif_test_noise(f, time, samples, alg, noise_values, *, u0=None, repeat=5, metric=mean_absolute_error, xscale='symlog', verbose=False, ax=None, rotate=False, **plot_kwargs):
    """Зависимость между ошибкой и шумом"""
    if isinstance(f, np.ndarray):
        t = np.linspace(*time, int(samples * (time[1] - time[0])))
        assert(len(f) == len(t))
        data = np.copy(f)
    else:
        data, t = gen_data(f, time, samples, u0)
    real_deriv = FiniteDifference().fit_transform(data, t)
    
    error = []
    for i, noise in enumerate(noise_values):
        cur_error = 0
        for _ in range(repeat):
            noise_data = data + np.random.default_rng().normal(scale=noise, size=data.shape)
            deriv = alg.fit_transform(noise_data, t)
            cur_error += metric(real_deriv, deriv)
        error.append(cur_error / repeat)

        if verbose:
            print(f'Noise value {i+1}/{len(noise_values)} ({noise}): Done')
    if verbose:
        clear_output()
    
    standalone = ax is None
    if standalone:
        fig = plt.figure()
        ax = fig.gca()

    if xscale == 'symlog':
        ax.set_xscale(xscale, linthresh=np.partition(noise_values, 1)[1])
    else:
        ax.set_xscale(xscale)
    ax.set(xlabel='noise', ylabel='error', xticks=noise_values)
    ax.get_xaxis().set_major_formatter('{x:n}')
    ax.plot(noise_values, error, marker='.', **plot_kwargs)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if standalone:
        plt.show()


def plot_dif_test_arg_noise(f, time, samples, alg, name, values, noise_values, *, u0=None, repeat=5, metric=mean_absolute_error, size=None, vmax=None, ax=None, verbose=False, return_error=False, **heatmap_kwargs):
    """Зависимость между ошибкой, параметром и шумом"""
    if isinstance(f, np.ndarray):
        t = np.linspace(*time, int(samples * (time[1] - time[0])))
        assert(len(f) == len(t))
        data = np.copy(f)
    else:
        data, t = gen_data(f, time, samples, u0)
    real_deriv = FiniteDifference().fit_transform(data, t)
    
    error = np.zeros((len(noise_values), len(values)))
    for i, noise in zip(range(len(noise_values)-1, -1, -1), noise_values):
        for _ in range(repeat):
            noise_data = data + np.random.default_rng().normal(scale=noise, size=data.shape)
            for j, value in enumerate(values):
                alg.set_params(**{name: value})
                deriv = alg.fit_transform(noise_data, t)
                error[i, j] += metric(real_deriv, deriv) / repeat
        
        if verbose:
            print(f'Noise ({name}) value {len(noise_values)-i}/{len(noise_values)} ({noise}): Done')
    if verbose:
        clear_output()
    
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

    if vmax is not None:
        heatmap_kwargs['vmax'] = vmax

    ax = sns.heatmap(error, annot=True, xticklabels=values, yticklabels=noise_values[::-1], vmin=0, cmap='Reds', linewidth=.1, ax=ax, **heatmap_kwargs)
    ax.set(ylabel='noise', xlabel=name)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    ax.plot(error.argmin(1) + 0.5, np.arange(len(noise_values)) + 0.5, linewidth=2.5, color='lightgray', linestyle='--')

    if standalone:
        plt.show()

    if return_error:
        return error


def plot_dif_results(f, time, samples, models, noise_values, arguments, u0=None, size=(3.5, 2.5), verbose=False):
    assert(len(models) == len(arguments))
    for args in arguments:
        assert(len(args) == 0 or len(args) == len(noise_values))
        
    if isinstance(f, np.ndarray):
        t = np.linspace(*time, int(samples * (time[1] - time[0])))
        assert(len(f) == len(t))
        data = np.copy(f)
    else:
        data, t = gen_data(f, time, samples, u0)
    real_deriv = FiniteDifference().fit_transform(data, t)
    
    w = len(noise_values) + 2
    h = len(models) + 1
    fig = plt.figure(figsize=(w * size[0], h * size[1]))
    
    width_ratios = [1 / w] * w
    width_ratios[1] = 0.001
    gs = GridSpec(h, w, figure=fig, width_ratios=width_ratios)
    
    subplot_args = dict()
    if data.shape[1] == 3:
        subplot_args['projection'] = '3d'
    
    ax = fig.add_subplot(gs[0, 0], **subplot_args)
    ax.set_title('Data')
    plot_data(data, t, ax=ax)
    
    ax = fig.add_subplot(gs[1, 0], **subplot_args)
    ax.set_title('Derivative')
    plot_data(real_deriv, t, ax=ax)

    rng = np.random.default_rng()
    for i, noise in enumerate(noise_values):
        noise_data = data + rng.normal(scale=noise, size=data.shape)
        ax = fig.add_subplot(gs[0, i+2], **subplot_args)
        ax.set_title(f'Noise: {noise:.2g}')
        plot_data(noise_data, t, ax=ax)
        
        for j, (model, args) in enumerate(zip(models, arguments), 1):
            if i == 0:
                ax = fig.add_subplot(gs[j, 1])
                ax.set(xticks=[], yticks=[], frame_on=False)
                ax.text(1, 0.5, type(model).__name__, fontsize=12, rotation=90, ha='center', va='center')
            
            if len(args) != 0:
                model.set_params(**args[i])
            deriv = model.transform(noise_data, t)
            error = mean_absolute_error(real_deriv, deriv)
            
            ax = fig.add_subplot(gs[j, i+2], **subplot_args)
            ax.set_title(f'MAE: {error:.2g}')
            plot_data(deriv, t, ax=ax)
            
        if verbose:
            print(f'Noise value {i+1}/{len(noise_values)} ({noise}): Done')
    if verbose:
        clear_output()
        
    fig.tight_layout()
    plt.show()
