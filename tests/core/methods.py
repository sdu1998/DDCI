import os
import numpy as np
import pandas as pd
import copy
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from tqdm import tqdm
import pdb
import warnings
from scipy.special import expit
from core import standard_weighted_quantile

"""
    BASELINES
"""

def trailing_window(
    scores,
    alpha,
    lr, # Dummy argument
    weight_length,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if min(weight_length, t_pred) < np.ceil(1/alpha):
            qs[t] = scores[t_pred]
        else:
            qs[t] = np.quantile(scores[max(t_pred-weight_length,0):t_pred], 1-alpha, method='higher')
    results = {"method": "Trail", "q" : qs}
    return results


def trailing_window_Nex(
    scores,
    alpha,
    lr,  # dummy
    weight_length,
    ahead,
    decay=0.99,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))

    for t in range(T_test):
        t_pred = t - ahead + 1
        if t_pred <= 0:
            continue

        start = max(0, t_pred - weight_length)
        window = scores[start:t_pred]
        n = len(window)
        if n == 0:
            continue

        # 去掉 NaN，防止 .max() 出问题
        window = window[np.isfinite(window)]
        if len(window) == 0:
            continue

        # 指数衰减权重，最近样本权重最大
        exponents = np.arange(len(window) - 1, -1, -1)
        w = (decay ** exponents).astype(float)
        w /= w.sum()

        # 直接调用标准加权分位函数，maxscore 改成 window 的最大值
        qs[t] = standard_weighted_quantile(
            window,
            w,
            1 - alpha,
            maxscore=window.max(),
            allow_illegal_weights=False
        )

    return {"method": "Trail_Nex", "q": qs}


def aci_clipped(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        clip_value = scores[max(t_pred-window_length,0):t_pred].max() if t_pred > 0 else np.infty
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t_pred+1):
                qs[t] = np.infty
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.infty
        if qs[t] == np.infty:
            qs[t] = clip_value
    results = { "method": "ACI (clipped)", "q" : qs, "alpha" : alphas}
    return results


def aci(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t_pred+1):
                qs[t] = np.infty
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.infty
    results = { "method": "ACI", "q" : qs, "alpha" : alphas}

    return results

"""
    New methods
"""

def quantile(
    scores,
    alpha,
    lr,
    ahead,
    proportional_lr=True,
    *args,
    **kwargs
):
    T_burnin = kwargs['T_burnin']
    results = quantile_integrator_log(scores, alpha, lr, 1.0, 0, ahead, T_burnin, proportional_lr=proportional_lr)
    results['method'] = 'Quantile'
    return results

def mytan(x):
    if x >= np.pi/2:
        return np.infty
    elif x <= -np.pi/2:
        return -np.infty
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI):
    if KI == 0:
        return 0
    tan_out = mytan(x * np.log(t+1)/(Csat * (t+1)))
    out = KI * tan_out
    return  out

def saturation_fn_sqrt(x, t, Csat, KI):
    return KI * mytan((x * np.sqrt(t+1))/((Csat * (t+1))))

def quantile_integrator_log(
    scores,
    alpha,
    lr,
    Csat,
    KI,
    ahead,
    T_burnin,
    proportional_lr=True,
    *args,
    **kwargs
):
    data = kwargs['data'] if 'data' in kwargs.keys() else None
    results = quantile_integrator_log_scorecaster(scores, alpha, lr, data, T_burnin, Csat, KI, True, ahead, proportional_lr=proportional_lr, scorecast=False)
    results['method'] = "Quantile+Integrator (log)"
    return results


def OGD(
    scores,
    alpha,
    lr,
    ahead,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    covereds = np.zeros((T_test,))

    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue

        covereds[t] = qs[t] >= scores[t]

        grad = alpha if covereds[t_pred] else -(1-alpha)

        if t < T_test - 1:
            qs[t+1] = qs[t] - lr * grad
    results = {"method": "OGD", "q" : qs}
  
    return results


def SF_OGD(
    scores,
    alpha,
    lr,
    ahead,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    covereds = np.zeros((T_test,))

    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue

        covereds[t] = qs[t] >= scores[t]

        grad = alpha if covereds[t_pred] else -(1-alpha)
        grad_all_square = (alpha - (covereds - 1)) ** 2

        lr_norm = lr / np.sqrt(np.sum(grad_all_square[:t+1]))

        if t < T_test - 1:
            qs[t+1] = qs[t] - lr_norm * grad
    results = {"method": "SF_OGD", "q" : qs}
  
    return results


def decay_OGD(
    scores,
    alpha,
    lr,
    ahead,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    eps = 0.1

    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue
        lr_t = lr * t**(-0.5-eps) if t else lr

        covereds[t] = qs[t] >= scores[t]

        grad = alpha if covereds[t_pred] else -(1-alpha)

        if t < T_test - 1:
            qs[t+1] = qs[t] - lr_t * grad
    results = {"method": "decay_OGD", "q" : qs}
  
    return results


"""
    This is the master method for the quantile, integrator, and scorecaster methods.
"""
def quantile_integrator_log_scorecaster(
    scores,
    alpha,
    lr,
    data,
    T_burnin,
    Csat,
    KI,
    upper,
    ahead,
    integrate=True,
    proportional_lr=True,
    scorecast=True,
#    onesided_integrator=False,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    scorecasts = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    seasonal_period = kwargs.get('seasonal_period')
    if seasonal_period is None:
        seasonal_period = 1
    # Load the scorecaster
    try:
        # If the data contains a scorecasts column, then use it!
        if 'scorecasts' in data.columns:
            scorecasts = np.array([s[int(upper)] for s in data['scorecasts'] ])
            train_model = False
        else:
            scorecasts = np.load('./.cache/scorecaster/' + kwargs.get('config_name') + '_' + str(upper) + '.npy')
            train_model = False
    except:
        train_model = True
    # Run the main loop
    # At time t, we observe y_t and make a prediction for y_{t+ahead}
    # We also update the quantile at the next time-step, q[t+1], based on information up to and including t_pred = t - ahead + 1.
    #lr_t = lr * (scores[:T_burnin].max() - scores[:T_burnin].min()) if proportional_lr and T_burnin > 0 else lr
    for t in tqdm(range(T_test)):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived
        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update and saturation function
        grad = alpha if covereds[t_pred] else -(1-alpha)
        #integrator = saturation_fn_log((1-covereds)[T_burnin:t_pred].sum() - (t_pred-T_burnin)*alpha, (t_pred-T_burnin), Csat, KI) if t_pred > T_burnin else 0
        integrator_arg = (1-covereds)[:t_pred].sum() - (t_pred)*alpha
        #if onesided_integrator:
        #    integrator_arg = np.clip(integrator_arg, 0, np.infty)
        integrator = saturation_fn_log(integrator_arg, t_pred, Csat, KI)
        # Train and scorecast if necessary
        if scorecast and train_model and t_pred > T_burnin and t+ahead < T_test:
            curr_scores = np.nan_to_num(scores[:t_pred])
            model = ThetaModel(
                    curr_scores.astype(float),
                    period=seasonal_period,
                    ).fit()
            scorecasts[t+ahead] = model.forecast(ahead)
        # Update the next quantile
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = integrator if integrate else 0
            qs[t+1] = qts[t+1] + integrators[t+1]
            if scorecast:
                qs[t+1] += scorecasts[t+1]
    results = {"method": "Quantile+Integrator (log)+Scorecaster", "q" : qs}
    if train_model and scorecast:
        os.makedirs('./.cache/', exist_ok=True)
        os.makedirs('./.cache/scorecaster/', exist_ok=True)
        np.save('./.cache/scorecaster/' + kwargs.get('config_name') + '_' + str(upper) + '.npy', scorecasts)
    
    return results


"""
    This is the our method for the conditional quantile methods with length optimization. (CCI?)
"""
def ECI(
    scores,
    alpha,
    lr,
    T_burnin,
    ahead,
    proportional_lr=True,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    c = 1 

    # Run the main loop
    for t in tqdm(range(T_test)):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived

        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]

        # Next, calculate the quantile update
        grad = alpha if covereds[t_pred] else -(1-alpha)

        # Then, caluate the error quantification term
        eq = (scores[t_pred]-qs[t_pred])*c*np.exp(-c*(scores[t_pred]-qs[t_pred]))/((1+np.exp(-c*(scores[t_pred]-qs[t_pred]))) ** 2)  # (s-q)*f'
        integrator = np.mean(eq)

        # Update the next quantile
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = lr_t * integrator
            qs[t+1] = qts[t+1] + integrators[t+1]

    results = {"method": "ECI", "q" : qs}

    return results


def ECI_cutoff(
    scores,
    alpha,
    lr,
    T_burnin,
    ahead,
    proportional_lr=True,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    c = 1

    # Run the main loop
    for t in tqdm(range(T_test)):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived

        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]

        # Next, calculate the quantile update
        grad = alpha if covereds[t_pred] else -(1-alpha)

        if t_pred == 0:
            continue

        # Then, caluate the error quantification term
        eq = (scores[t_pred]-qs[t_pred])*c*np.exp(-c*(scores[t_pred]-qs[t_pred]))/((1+np.exp(-c*(scores[t_pred]-qs[t_pred]))) ** 2)  # (s-q)*f'

        if abs(scores[t_pred]-qs[t_pred]) < 1 * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()):  # cutoff
            eq = 0
        integrator = np.sum(eq)
        
        # Update the next quantile
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = lr_t * integrator
            qs[t+1] = qts[t+1] + integrators[t+1]
    results = {"method": "ECI_cutoff", "q" : qs}

    return results


def ECI_integral(
    scores,
    alpha,
    lr,
    T_burnin,
    ahead,
    proportional_lr=True,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    c = 1

    # Run the main loop
    for t in tqdm(range(T_test)):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived

        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]

        # Next, calculate the quantile update and saturation function
        grad = alpha if covereds[t_pred] else -(1-alpha)

        # Then, caluate the error quantification term
        eq = (scores[:t_pred]-qs[:t_pred])*c*np.exp(-c*(scores[:t_pred]-qs[:t_pred]))/((1+np.exp(-c*(scores[:t_pred]-qs[:t_pred]))) ** 2)  # weighted s-q
        # weight = np.linspace(0.1, 1, num=t_pred)  # linear weight
        weight = 0.95 ** (t_pred - np.arange(1, t_pred+1))  # exp_decay weight
        weight = weight / np.sum(weight)
        eq = weight * eq

        integrator = np.sum(eq)

        # Update the next quantile
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = lr_t * integrator
            qs[t+1] = qts[t+1] + integrators[t+1]
    results = {"method": "ECI_integral", "q" : qs}

    return results


def DDCI(
    scores,
    alpha,
    lr,
    T_burnin,
    ahead,
    epsilon,
    proportional_lr=True,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    integrators_estimate = np.zeros((T_test,))
    covereds_estimate = np.zeros((T_test,))
    epsilon  = epsilon

    deltas = np.zeros((T_test,))            # 保存 delta
    deltas_estimate = np.zeros((T_test,))
    eqs = np.zeros((T_test,))               # 保存 eq
    eqs_estimate = np.zeros((T_test,))

    # Run the main loop
    for t in tqdm(range(T_test)):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived

        # if t >= T_burnin and scores[t_pred] < np.quantile(scores[t_lr_min:t_lr], 1-alpha, method='higher'):
        if t >= T_burnin:
            
            qs_estimate = np.quantile(scores[t_lr_min:t_lr], 1-alpha, method='higher')
            covereds_estimate[t] = qs_estimate >= scores[t]
            grad_estimate = 0
            scale_estimate = np.abs(scores[t_pred] - qs_estimate) / (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min())
            delta_estimate = scores[t_pred] - qs_estimate
            # eq_estimate = sigmoid_max_estimate * scale_estimate * np.tanh(0.5*delta_estimate)  # (s-q)*f'  # Using a tanh function for smoother behavior # tanh function is used to avoid large gradients
            eq_estimate = 1 * (1-scale_estimate) * np.tanh(0.5*delta_estimate)
        else:
            qs_estimate = 0
            covereds_estimate[t] = 0
            grad_estimate = 0
            delta_estimate = 0
            eq_estimate = 0

        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update
        grad = alpha if covereds[t_pred] else -(1-alpha)

        # coverage_rate = covereds[t_pred-T_burnin:t_pred].mean() if t_pred - T_burnin >= T_burnin else covereds[T_burnin:t_pred].mean()
        # c = c / coverage_rate

        # Then, caluate the error quantification term

        delta = scores[t_pred]-qs[t_pred]

        scale = np.abs(delta) / (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if t >= T_burnin else 0
        # max_score = scores[:t_lr].max() if t >= T_burnin else 0
        # sigmoid(max_score)
        # sigmoid_max = 1 / (1 + np.exp(-0.1*max_score))
        eq = 1 * scale * np.tanh(0.5*delta)  # (s-q)*f'
        # eq = (scores[t_pred]-qs[t_pred])*c*np.exp(-c*(scores[t_pred]-qs[t_pred]))/((1+np.exp(-c*(scores[t_pred]-qs[t_pred]))) ** 2)

        deltas[t] = delta
        deltas_estimate[t] = delta_estimate

        eqs[t] = eq
        eqs_estimate[t] = eq_estimate
        
        # c_estimate = c / empirical_coverage_rate

        integrator = np.mean(eq)
        integrator_estimate = np.mean(eq_estimate)

        if integrator * integrator_estimate > 0:
            integrator_estimate = -integrator_estimate

        # Update the next quantile
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad - lr_t * epsilon * grad_estimate
            integrators[t+1] = lr_t * integrator
            integrators_estimate[t+1] = lr_t * integrator_estimate
            qs[t+1] = qts[t+1] + integrators[t+1] + epsilon * (integrators_estimate[t+1])

    results = {
        "method": "DDCI",
        "q": qs,
        "delta": deltas,
        "delta_estimate": deltas_estimate,
        "eq": eqs,
        "eq_estimate": eqs_estimate,
    }

    return results


def DDCI_Nex(
    scores,
    alpha,
    lr,
    T_burnin,
    ahead,
    epsilon,
    proportional_lr=True,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    integrators_estimate = np.zeros((T_test,))
    covereds_estimate = np.zeros((T_test,))
    epsilon = epsilon

    deltas = np.zeros((T_test,))            # 保存 delta
    deltas_estimate = np.zeros((T_test,))
    eqs = np.zeros((T_test,))               # 保存 eq
    eqs_estimate = np.zeros((T_test,))

    # Run the main loop
    for t in tqdm(range(T_test)):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived

        # if t >= T_burnin and scores[t_pred] < np.quantile(scores[t_lr_min:t_lr], 1-alpha, method='higher'):
        # if t >= T_burnin:

        #     weight = 0.99 ** (t_pred - np.arange(1, t+1))  # exp_decay weight
        #     weight = weight / np.sum(weight)

        #     weighted_scores = scores[:t_lr] * weight
            
        #     qs_estimate = np.quantile(weighted_scores, 1-alpha, method='higher')
        #     covereds_estimate[t] = qs_estimate >= scores[t]
        #     scale_estimate = np.abs(scores[t_pred] - qs_estimate) / (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min())
        #     grad_estimate = 0
        #     delta_estimate = scores[t_pred] - qs_estimate
        #     max_score = scores[:t_lr].max()
        #     sigmoid_max_estimate = 1 / (1 + np.exp(-0.1*max_score))
        #     eq_estimate = sigmoid_max_estimate * (1-scale_estimate) * np.tanh(0.1*delta_estimate)  # (s-q)*f'  # Using a tanh function for smoother behavior # tanh function is used to avoid large gradients

        # else:
        #     qs_estimate = 0
        #     covereds_estimate[t] = 0
        #     grad_estimate = 0
        #     delta_estimate = 0
        #     eq_estimate = 0

        if t >= T_burnin:
            start = max(0, t_pred - T_burnin)
            window = scores[start:t_pred]
            window = window[np.isfinite(window)]
            n = len(window)

            if n == 0:
                qs_estimate = 0.0
                covereds_estimate[t] = 0
                grad_estimate = 0.0
                delta_estimate = 0.0
                eq_estimate = 0.0
            else:
                exponents = np.arange(n - 1, -1, -1)
                w = (0.99 ** exponents).astype(float)
                w /= w.sum()

                qs_estimate = standard_weighted_quantile(
                    window, w, 1 - alpha,
                    maxscore=window.max(),
                    allow_illegal_weights=False
                )

                covereds_estimate[t] = qs_estimate >= scores[t]
                denom = scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()
                denom = denom if np.isfinite(denom) and denom > 0 else 1.0
                scale_estimate = np.abs(scores[t_pred] - qs_estimate) / denom

                grad_estimate = 0.0
                delta_estimate = scores[t_pred] - qs_estimate
                max_score = scores[:t_lr].max() if t_lr > 0 else 0.0
                sigmoid_max_estimate = 1.0 / (1.0 + np.exp(-0.1 * max_score))
                eq_estimate = sigmoid_max_estimate * (1 - scale_estimate) * np.tanh(0.1 * delta_estimate)
        else:
            qs_estimate = 0.0
            covereds_estimate[t] = 0
            grad_estimate = 0.0
            delta_estimate = 0.0
            eq_estimate = 0.0

        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update
        grad = alpha if covereds[t_pred] else -(1-alpha)

        # coverage_rate = covereds[t_pred-T_burnin:t_pred].mean() if t_pred - T_burnin >= T_burnin else covereds[T_burnin:t_pred].mean()
        # c = c / coverage_rate

        # Then, caluate the error quantification term
        delta = scores[t_pred]-qs[t_pred]
        scale = np.abs(delta) / (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if t >= T_burnin else 0
        max_score = scores[:t_lr].max() if t >= T_burnin else 0
        # sigmoid(max_score)
        sigmoid_max = 1 / (1 + np.exp(-0.1*max_score))
        eq = sigmoid_max * scale * np.tanh(0.5*delta)
        # eq = (scores[t_pred]-qs[t_pred])*c*np.exp(-c*(scores[t_pred]-qs[t_pred]))/((1+np.exp(-c*(scores[t_pred]-qs[t_pred]))) ** 2)

        deltas[t] = delta
        deltas_estimate[t] = delta_estimate
        

        # if delta * delta_estimate > 0:
        #     eq_estimate = -eq_estimate

        eqs[t] = eq
        eqs_estimate[t] = eq_estimate
        
        # c_estimate = c / empirical_coverage_rate

        integrator = np.mean(eq)
        integrator_estimate = np.mean(eq_estimate)

        if integrator * integrator_estimate > 0:
            integrator_estimate = -integrator_estimate

        # Update the next quantile
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad - lr_t * epsilon * grad_estimate
            integrators[t+1] = lr_t * integrator
            integrators_estimate[t+1] = lr_t * integrator_estimate
            qs[t+1] = qts[t+1] + integrators[t+1] + epsilon * (integrators_estimate[t+1])

    results = {
        "method": "DDCI_Nex",
        "q": qs,
        "delta": deltas,
        "delta_estimate": deltas_estimate,
        "eq": eqs,
        "eq_estimate": eqs_estimate,
    }

    return results
