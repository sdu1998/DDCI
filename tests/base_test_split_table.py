import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import pandas as pd
from core import standard_weighted_quantile, trailing_window, aci, aci_clipped, quantile, quantile_integrator_log, quantile_integrator_log_scorecaster, ECI, OGD, SF_OGD, ECI_cutoff, ECI_integral, decay_OGD, DDCI, DDCI_Nex
from core.synthetic_scores import generate_scores
from core.model_scores import generate_forecasts
from datasets import load_dataset
from darts import TimeSeries
import yaml
import pickle
import pdb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    json_name = sys.argv[1]
    if len(sys.argv) > 2:
        overwrite = sys.argv[2].split(",")
    else:
        overwrite = []
    args = yaml.safe_load(open(json_name))
    # Set up folder and filename
    foldername = './results/'
    config_name = json_name.split('.')[-2].split('/')[-1]
    filename = foldername + config_name + ".pkl"
    os.makedirs(foldername, exist_ok=True)
    real_data = args['real']
    quantiles_given = args['quantiles_given']
    multiple_series = args['multiple_series']
    score_function_name = args['score_function_name'] if real_data else "synthetic"
    model_names = args['sequences'][0]['model_names'] if real_data else ["synthetic"]
    ahead = args['ahead'] if real_data and 'ahead' in args.keys() else 1
    minsize = args['minsize'] if real_data and 'minsize' in args.keys() else 0
    asymmetric = False

    # Try reading in results
    try:
        with open(filename, 'rb') as handle:
            all_results = pickle.load(handle)
    except:
        all_results = {}

    metrics_rows = []

    for model_name in model_names:
        try:
            results = all_results[model_name]
        except:
            results = {}
        # Initialize the score function
        if real_data:
            score_function_name = args['score_function_name']
            if score_function_name == "absolute-residual":
                def score_function(y, forecast):
                    return np.abs(y - forecast)
                def set_function(forecast, q):
                    return np.array([forecast - q, forecast + q])
            elif score_function_name == "signed-residual":
                def score_function(y, forecast):
                    return np.array([forecast - y, y - forecast])
                def set_function(forecast, q):
                    return np.array([forecast - q[0], forecast + q[1]])
                asymmetric = True
            elif score_function_name == "cqr-symmetric":
                def score_function(y, forecasts):
                    return np.maximum(forecasts[0] - y, y - forecasts[-1])
                def set_function(forecast, q):
                    return np.array([forecast[0] - q, forecast[-1] + q])
            elif score_function_name == "cqr-asymmetric":
                def score_function(y, forecasts):
                    return np.array([forecasts[0] - y, y - forecasts[-1]])
                def set_function(forecast, q):
                    return np.array([forecast[0] - q[0], forecast[-1] + q[1]])
                asymmetric = True
            else:
                raise ValueError("Invalid score function name")

        # Get dataframe and add forecasts and scores to it
        if real_data:
            data = load_dataset(args['sequences'][0]['dataset'])
            # Get the forecasts
            if 'forecasts' not in data.columns:
                os.makedirs('./datasets/proc/', exist_ok=True)
                os.makedirs('./datasets/proc/' + config_name, exist_ok=True)
                args['sequences'][0]['savename'] = './datasets/proc/' + config_name +  '/' + model_name + '.npz'
                args['sequences'][0]['T_burnin'] = args['T_burnin']
                args['sequences'][0]['ahead'] = ahead
                args['sequences'][0]['model_name'] = model_name
                data['forecasts'] = generate_forecasts(data, **args['sequences'][0])
            # Compute scores
            if 'scores' not in data.columns:
                data['scores'] = [ score_function(y, forecast) for y, forecast in zip(data['y'], data['forecasts']) ]
        else:
            scores_list = []
            for key in args['sequences'].keys():
                scores_list += [generate_scores(**args['sequences'][key])]
            scores = np.concatenate(scores_list).astype(float)
            # Make a pandas dataframe with a datetime index and the scores in their own column called `scores'.
            data = pd.DataFrame({'scores': scores}, index=pd.date_range(start='1/1/2018', periods=len(scores), freq='D'))

        # Loop through each method and learning rate, and compute the results
        for method in args['methods'].keys():
            if (method in results.keys()) and (method not in overwrite):
                continue
            fn = None
            if method == "Trail":
                fn = trailing_window
                args['methods'][method]['lrs'] = [None]
            elif method == "ACI":
                fn = aci
            elif method == "ACI (clipped)":
                fn = aci_clipped
            elif method == "OGD":
                fn = OGD
            elif method == "SF_OGD":
                fn = SF_OGD
            elif method == "decay_OGD":
                fn = decay_OGD
            elif method == "Quantile":
                fn = quantile
            elif method == "Quantile+Integrator (log)":
                fn = quantile_integrator_log
            elif method == "Quantile+Integrator (log)+Scorecaster":
                fn = quantile_integrator_log_scorecaster
            elif method == "ECI":
                fn = ECI
            elif method == "ECI_cutoff":
                fn = ECI_cutoff
            elif method == "ECI_integral":
                fn = ECI_integral
            elif method == 'DDCI':
                fn = DDCI
            elif method == 'DDCI_Nex':
                fn = DDCI_Nex
            else:
                raise Exception(f"Method {method} not implemented")
            lrs = args['methods'][method]['lrs']
            kwargs_base = args['methods'][method]
            kwargs_base["T_burnin"] = args["T_burnin"]
            kwargs_base["data"] = data if real_data else None
            kwargs_base["seasonal_period"] = args.get("seasonal_period")
            kwargs_base["config_name"] = config_name
            kwargs_base["ahead"] = ahead


            uses_eps = method in ("DDCI", "DDCI_Nex")

            raw_eps = args['methods'][method].get('epsilons',
                    args['methods'][method].get('epsilon', 0.0))
            if isinstance(raw_eps, (list, tuple, np.ndarray)):
                eps_list = list(raw_eps)
            else:
                eps_list = [raw_eps]

            if not uses_eps:
                eps_list = [None]


            results[method] = {}
            # pdb.set_trace()
            for lr in lrs:
                for epsilon in eps_list:
                    kwargs = dict(kwargs_base)
                    if uses_eps:
                        kwargs['epsilon'] = float(epsilon)
                    else:
                        kwargs.pop('epsilon', None)

                    if asymmetric:
                        stacked_scores = np.stack(data['scores'].to_list())
                        kwargs['upper'] = False
                        res0 = fn(stacked_scores[:,0], args['alpha']/2, lr, **kwargs)
                        kwargs['upper'] = True
                        res1 = fn(stacked_scores[:,1], args['alpha']/2, lr, **kwargs)
                        q = [ np.array([res0['q'][i], res1['q'][i]]) for i in range(len(res0['q'])) ]
                        res = {"q": q}
                    else:
                        kwargs['upper'] = True
                        res = fn(data['scores'].to_numpy(), args['alpha'], lr, **kwargs)
                        q = res['q']

                    if real_data:
                        sets = [ set_function(data['forecasts'].interpolate().to_numpy()[i], q[i]) for i in range(len(q)) ]
                        sets = [ np.array([np.minimum(sets[j][0], sets[j][1]-minsize),
                                            np.maximum(sets[j][1], sets[j][0]+minsize)]) for j in range(len(sets)) ]
                    else:
                        sets = None

                    key = (lr, epsilon)
                    results[method][key] = {
                        "q": q,
                        "sets": sets,
                        "delta": res.get("delta"),
                        "delta_estimate": res.get("delta_estimate"),
                        "eq": res.get("eq"),
                        "eq_estimate": res.get("eq_estimate"),
                    }

                    if sets is not None:
                        intervals = [y - x for x, y in sets]
                        average_interval = np.mean(intervals)
                        median_interval  = np.median(intervals)
                    else:
                        average_interval = np.nan
                        median_interval  = np.nan

                    metrics_rows.append({
                        "model": model_name,
                        "method": method,
                        "lr": lr,
                        "epsilon": epsilon,
                        "average_interval": average_interval,
                        "median_interval": median_interval
                    })

                    print(f"{model_name}: {method} lr={lr} eps={epsilon}: "
                        f"avg={average_interval:.6f}, med={median_interval:.6f}")

        # Save some metadata

        results["scores"] = data['scores']
        results["alpha"] = args['alpha']
        results["T_burnin"] = args['T_burnin']
        results["quantiles_given"] = quantiles_given
        results["multiple_series"] = multiple_series
        results["real_data"] = real_data
        results["score_function_name"] = score_function_name
        results["asymmetric"] = asymmetric

        if real_data:
            results["forecasts"] = data['forecasts']
            results["data"] = data
        all_results[model_name] = results

    # Save results
    with open(filename, 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_metrics = pd.DataFrame(metrics_rows).sort_values(
        by=["model", "method", "lr", "epsilon"], kind="stable"
    )

    eps_methods = {"DDCI", "DDCI_Nex"}
    df_eps = df_metrics[df_metrics["method"].isin(eps_methods)].copy()
    df_others = df_metrics[~df_metrics["method"].isin(eps_methods)].copy()

    excel_eps = f"./results/{config_name}_intervals_eps.xlsx"
    with pd.ExcelWriter(excel_eps, engine="openpyxl") as writer:
        df_eps.to_excel(writer, sheet_name="long", index=False)

        avg_pivot = df_eps.pivot_table(
            index=["model", "method", "epsilon"],
            columns="lr",
            values="average_interval",
            aggfunc="mean"
        ).sort_index()
        avg_pivot.to_excel(writer, sheet_name="average_pivot")

        med_pivot = df_eps.pivot_table(
            index=["model", "method", "epsilon"],
            columns="lr",
            values="median_interval",
            aggfunc="mean"
        ).sort_index()
        med_pivot.to_excel(writer, sheet_name="median_pivot")

        avg_pivot_no_eps = df_eps.pivot_table(
            index=["model", "method"],
            columns="lr",
            values="average_interval",
            aggfunc="mean"
        ).sort_index()
        avg_pivot_no_eps.to_excel(writer, sheet_name="average_pivot_no_eps")

        med_pivot_no_eps = df_eps.pivot_table(
            index=["model", "method"],
            columns="lr",
            values="median_interval",
            aggfunc="mean"
        ).sort_index()
        med_pivot_no_eps.to_excel(writer, sheet_name="median_pivot_no_eps")

    print(f"[OK] Exported EPS methods to {excel_eps}")

    excel_others = f"./results/{config_name}_intervals_others.xlsx"
    with pd.ExcelWriter(excel_others, engine="openpyxl") as writer:
        df_others.to_excel(writer, sheet_name="long", index=False)

        avg_pivot_others = df_others.pivot_table(
            index=["model", "method"],
            columns="lr",
            values="average_interval",
            aggfunc="mean"
        ).sort_index()
        avg_pivot_others.to_excel(writer, sheet_name="average_pivot")

        med_pivot_others = df_others.pivot_table(
            index=["model", "method"],
            columns="lr",
            values="median_interval",
            aggfunc="mean"
        ).sort_index()
        med_pivot_others.to_excel(writer, sheet_name="median_pivot")

    print(f"[OK] Exported OTHER methods to {excel_others}")