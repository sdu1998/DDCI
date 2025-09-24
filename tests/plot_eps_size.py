import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 28,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
})

def plot_avg_interval_from_excel(
    excel_path, dataset_name, lrs_by_model
):

    df = pd.read_excel(excel_path, sheet_name="long")
    df["lr_num"] = pd.to_numeric(df["lr"], errors="coerce")
    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    df["average_interval"] = pd.to_numeric(df["average_interval"], errors="coerce")
    df["method"] = df["method"].replace({"DDCI_Nex": "DDCI-Nex"})

    save_dir = os.path.join("plots", "ablation_study", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    sns.set_theme(context="notebook", style="whitegrid")

    model_lr_dict = {}
    for pair in lrs_by_model.split(";"):
        m, lr = pair.split(":")
        model_lr_dict[m.strip()] = float(lr)

    models = ["ar", "prophet", "theta", "transformer"]

    rows, cols = 1, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), sharex=True, sharey=False)
    axes = axes.flatten()

    handles_all, labels_all = [], []

    for i, m in enumerate(models):
        ax = axes[i]
        if m not in model_lr_dict:
            print(f"[WARN] {m} no clear  lr, skip")
            continue
        lr_fix = model_lr_dict[m]

        subm = df[(df["model"] == m) & (np.isclose(df["lr_num"], lr_fix))]
        if subm.empty:
            print(f"[WARN] {m}, lr={lr_fix} no data")
            continue

        sns.lineplot(
            data=subm.sort_values(["method", "epsilon"]),
            x="epsilon", y="average_interval", hue="method", marker="o", ax=ax
        )
        ax.set_title(f"{m}", fontsize=16)
        ax.set_xlabel("Epsilon", fontsize=14)
        ax.set_ylabel("Average Size", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, alpha=0.35, linestyle="--")

        handles, labels = ax.get_legend_handles_labels()
        handles_all.extend(handles)
        labels_all.extend(labels)
        ax.get_legend().remove()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.legend(
        handles_all[:len(set(labels_all))],
        labels_all[:len(set(labels_all))],
        loc="lower center", ncol=len(set(labels_all)),
        fontsize=12, frameon=False
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    fname = f"{dataset_name}_allmodels_customlr_eps_avginterval.pdf"
    path = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[OK] saved: {path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot average interval vs epsilon (custom lr per model)")
    parser.add_argument("--excel", type=str, required=True, help="Path to *_intervals_eps.xlsx")
    parser.add_argument("--dataset", type=str, required=True, help="dataset/pkl name, e.g., GOOGL_test")
    parser.add_argument("--lrs-by-model", type=str, required=True,
                        help="e.g. 'ar:0.05;prophet:0.5;theta:0.1;transformer:0.5'")
    args = parser.parse_args()

    plot_avg_interval_from_excel(args.excel, dataset_name=args.dataset, lrs_by_model=args.lrs_by_model)