import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pickle as pkl

def _coverage_from_sets(sets, y, T_burnin):
    sets_arr = np.stack(sets)
    lower = sets_arr[T_burnin+1:, 0]
    upper = sets_arr[T_burnin+1:, 1]
    y_use = y[T_burnin+1:]
    covered = (lower <= y_use) & (y_use <= upper)
    return float(np.mean(covered))

def _gather_coverage_long_from_pkl(pkl_path):

    with open(pkl_path, "rb") as f:
        all_results = pkl.load(f)

    rows = []
    for model_name, res in all_results.items():
        if not isinstance(res, dict) or "T_burnin" not in res or "real_data" not in res:
            continue

        T_burnin = res["T_burnin"]
        real_data = res["real_data"]

        if real_data:
            y_series = res["data"]["y"]
            y_np = y_series.to_numpy().astype(float)
        else:
            continue

        for method in ("DDCI", "DDCI_Nex"):
            if method not in res:
                continue
            method_dict = res[method]
            for key, node in method_dict.items():
                if not isinstance(key, tuple) or "sets" not in node:
                    continue
                lr_k, eps_k = key
                sets = node["sets"]
                if sets is None:
                    continue

                coverage = _coverage_from_sets(sets, y_np, T_burnin)
                rows.append({
                    "model": model_name,
                    "method": method,                  
                    "method_plot": "DDCI-Nex" if method == "DDCI_Nex" else method, 
                    "lr": float(lr_k) if lr_k is not None else None,
                    "epsilon": float(eps_k) if eps_k is not None else None,
                    "coverage": coverage,
                })

    return pd.DataFrame(rows)


def plot_coverage_from_pkl(
    pkl_path,
    dataset_name,
    lr_fix=0.5,
    model="all",
    methods_to_plot=("DDCI", "DDCI-Nex"),
    percent_yaxis=True,
    lrs_by_model: str = None, 
):

    root = os.path.join("plots", "sa", dataset_name)
    os.makedirs(root, exist_ok=True)

    sns.set_theme(context="notebook", style="whitegrid")
    plt.rcParams.update({
        "font.size": 32,
        "axes.titlesize": 40,
        "axes.labelsize": 36,
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "legend.fontsize": 32,
    })

    y_min, y_max = 0.85, 0.95
    ref_line = 0.9

    df = _gather_coverage_long_from_pkl(pkl_path)
    if df.empty:
        raise ValueError("No DDCI/DDCI_Nex coverage data found in PKL.")

    df["method_plot"] = df["method_plot"].astype(str)

    methods_plot_set = set(methods_to_plot)

    if model != "all":
        sub = df[(df["model"] == model) & (np.isclose(df["lr"], float(lr_fix)))]
        if sub.empty:
            raise ValueError(f"{model}: lr={lr_fix} no data in PKL")

        sub = sub[sub["method_plot"].isin(methods_plot_set)]
        if sub.empty:
            raise ValueError(f"{model}: lr={lr_fix} has no DDCI/DDCI-Nex entries")

        plt.figure(figsize=(14, 10))
        ax = sns.lineplot(
            data=sub.sort_values(["method_plot", "epsilon"]),
            x="epsilon", y="coverage", hue="method_plot", marker="o",
            linewidth=4, markersize=16
        )
        plt.title(f"{model} — Coverage Rate vs Epsilon (lr={lr_fix})")
        plt.xlabel("Epsilon")
        plt.ylabel("Coverage Rate")

        ax.set_ylim(y_min, y_max)
        if percent_yaxis:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.axhline(ref_line, color="red", linestyle="--", linewidth=3)

        plt.grid(True, alpha=0.35, linestyle="--")
        plt.legend(title="method", frameon=False)
        plt.tight_layout()

        base_name = f"{dataset_name}_{model}_lr{lr_fix}_eps_coverage_fromPKL"
        for ext in ["png", "pdf"]:
            out_path = os.path.join(root, f"{base_name}.{ext}")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[OK] saved: {out_path}")
        plt.close()

    else:
        models = ["ar", "prophet", "theta", "transformer"]

        if not lrs_by_model:
            raise ValueError("model='all' requires --lrs-by-model, e.g. 'ar:0.05;prophet:0.5;theta:0.1;transformer:0.5'")
        model_lr_dict = {}
        for pair in lrs_by_model.split(";"):
            m, lr = pair.split(":")
            model_lr_dict[m.strip()] = float(lr)

        fig, axes = plt.subplots(1, 4, figsize=(8 * 4, 10), sharex=True, sharey=False)
        axes = axes.flatten()

        for i, m in enumerate(models):
            ax = axes[i]
            if m not in model_lr_dict:
                print(f"[WARN] skip {m}, no lr specified in --lrs-by-model")
                continue

            lr_m = model_lr_dict[m]
            sub = df[(df["model"] == m) & (np.isclose(df["lr"], lr_m))]
            sub = sub[sub["method_plot"].isin(methods_plot_set)]

            if not sub.empty:
                sns.lineplot(
                    data=sub.sort_values(["method_plot", "epsilon"]),
                    x="epsilon", y="coverage", hue="method_plot", marker="o",
                    linewidth=4, markersize=16, ax=ax
                )
            ax.set_title(f"{m}")
            ax.set_xlabel("Epsilon")
            ax.set_ylabel("Coverage Rate")

            ax.set_ylim(y_min, y_max)
            if percent_yaxis:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
            ax.axhline(ref_line, color="red", linestyle="--", linewidth=3)

            ax.grid(True, alpha=0.35, linestyle="--")
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(methods_plot_set), frameon=False)

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        base_name = f"{dataset_name}_allmodels_customlr_eps_coverage_fromPKL"
        for ext in ["pdf", "png"]:
            out_path = os.path.join(root, f"{base_name}.{ext}")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[OK] saved: {out_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot coverage vs epsilon from PKL (super large fonts, clean % labels)")
    parser.add_argument("--pkl", type=str, required=True,
                        help="Path to results <config>.pkl")
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset/pkl name, e.g., GOOGL_test")
    parser.add_argument("--model", type=str, default="all",
                        help="One of ['ar','prophet','theta','transformer','all']")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Fixed learning rate for single-model mode")
    parser.add_argument("--lrs-by-model", type=str, default=None,
                        help="For model='all', e.g. 'ar:0.05;prophet:0.5;theta:0.1;transformer:0.5'")
    args = parser.parse_args()

    plot_coverage_from_pkl(
        pkl_path=args.pkl,
        dataset_name=args.dataset,
        lr_fix=args.lr,
        model=args.model,
        methods_to_plot=("DDCI", "DDCI-Nex"),
        percent_yaxis=True,
        lrs_by_model=args.lrs_by_model
    )