import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 28,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
})

def _gather_rows_from_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        all_results = pkl.load(f)

    rows = []
    for model_name, res in all_results.items():
        if not isinstance(res, dict) or "T_burnin" not in res:
            continue

        for method in ("DDCI", "DDCI_Nex"):
            if method not in res:
                continue
            method_dict = res[method]

            # method_dict 的键是 (lr, epsilon)
            for key, node in method_dict.items():
                if not isinstance(key, tuple) or "sets" not in node:
                    continue
                lr_k, eps_k = key

                sets = node["sets"]
                if sets is None:
                    continue
                arr = np.stack(sets)  # shape (T, 2)
                avg_interval = float(np.mean(arr[:, 1] - arr[:, 0]))

                rows.append({
                    "model": model_name,
                    "method": method,
                    "lr": float(lr_k) if lr_k is not None else None,
                    "epsilon": float(eps_k) if eps_k is not None else None,
                    "average_interval": avg_interval,
                })

    df = pd.DataFrame(rows)
    df["method_plot"] = df["method"].replace({"DDCI_Nex": "DDCI-Nex"})
    return df


def plot_avg_interval_from_pkl(
    pkl_path, dataset_name, lrs_by_model
):

    df = _gather_rows_from_pkl(pkl_path)

    model_lr_dict = {}
    for pair in lrs_by_model.split(";"):
        m, lr = pair.split(":")
        model_lr_dict[m.strip()] = float(lr)

    save_dir = os.path.join("plots", "sa", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    sns.set_theme(context="notebook", style="whitegrid")

    models = ["ar", "prophet", "theta", "transformer"]
    rows, cols = 1, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), sharex=True, sharey=False)
    axes = axes.flatten()

    handles_all, labels_all = [], []

    for i, m in enumerate(models):
        ax = axes[i]
        if m not in model_lr_dict:
            print(f"[WARN] {m} has no specified lr; skipped.")
            continue

        lr_fix = model_lr_dict[m]
        subm = df[(df["model"] == m) & (np.isclose(df["lr"], lr_fix))]

        if subm.empty:
            print(f"[WARN] {m}, lr={lr_fix} has no data in PKL; skipped.")
            continue

        subm = subm[subm["method"].isin(["DDCI", "DDCI_Nex"])]

        subm_sorted = subm.sort_values(["method_plot", "epsilon"])

        sns.lineplot(
            data=subm_sorted,
            x="epsilon", y="average_interval", hue="method_plot", marker="o", ax=ax
        )

        ax.set_title(f"{m}", fontsize=16)
        ax.set_xlabel("Epsilon", fontsize=14)
        ax.set_ylabel("Average Size", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, alpha=0.35, linestyle="--")

        h, l = ax.get_legend_handles_labels()
        handles_all.extend(h)
        labels_all.extend(l)
        ax.get_legend().remove()

    seen, handles_dedup, labels_dedup = set(), [], []
    for h, l in zip(handles_all, labels_all):
        if l not in seen and l in ["DDCI", "DDCI-Nex"]:
            seen.add(l); handles_dedup.append(h); labels_dedup.append(l)

    fig.legend(handles_dedup, labels_dedup,
               loc="lower center", ncol=len(labels_dedup),
               fontsize=12, frameon=False)

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    out_path = os.path.join(save_dir, f"{dataset_name}_allmodels_customlr_eps_avginterval_fromPKL.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot average interval vs epsilon (custom lr per model) from PKL")
    parser.add_argument("--pkl", type=str, required=True, help="Path to results <config>.pkl")
    parser.add_argument("--dataset", type=str, required=True, help="dataset/pkl name, e.g., GOOGL_test")
    parser.add_argument("--lrs-by-model", type=str, required=True,
                        help="e.g. 'ar:0.05;prophet:0.5;theta:0.1;transformer:0.5'")
    args = parser.parse_args()

    plot_avg_interval_from_pkl(args.pkl, dataset_name=args.dataset, lrs_by_model=args.lrs_by_model)