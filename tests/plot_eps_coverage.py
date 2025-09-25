import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


def plot_coverage_from_excel(
    dataset_name,
    lr_fix=0.5,
    model="all",
    methods_to_plot=("DDCI", "DDCI-Nex"),
    percent_yaxis=True
):
    root = os.path.join("plots", "ablation_study", dataset_name)
    os.makedirs(root, exist_ok=True)

    model_names = ["ar", "prophet", "theta", "transformer"]

    sns.set_theme(context="notebook", style="whitegrid")
    plt.rcParams.update({
        "font.size": 32,
        "axes.titlesize": 40,
        "axes.labelsize": 36,
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "legend.fontsize": 32,
    })

    y_min, y_max = 0.88, 0.92
    ref_line = 0.9

    if model != "all":
        excel_path = os.path.join("plots", dataset_name, model, "coverage_rates.xlsx")
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"cannot find: {excel_path}")

        df = pd.read_excel(excel_path, sheet_name="coverage_long")
        df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
        df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
        df["lr_num"] = pd.to_numeric(df["lr"], errors="coerce")

        df["method"] = df["method"].replace({"DDCI_Nex": "DDCI-Nex"})

        sub = df[
            (np.isclose(df["lr_num"], float(lr_fix)))
            & (df["method"].isin(methods_to_plot))
        ]

        if sub.empty:
            raise ValueError(f"{model}: lr={lr_fix} no data")

        plt.figure(figsize=(14, 10))
        ax = sns.lineplot(
            data=sub.sort_values(["method", "epsilon"]),
            x="epsilon", y="coverage", hue="method", marker="o",
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

        base_name = f"{dataset_name}_{model}_lr{lr_fix}_eps_coverage"
        for ext in ["png", "pdf"]:
            out_path = os.path.join(root, f"{base_name}.{ext}")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[OK] saved: {out_path}")
        plt.close()

    else:
        fig, axes = plt.subplots(1, 4, figsize=(8 * 4, 10), sharex=True, sharey=False)
        axes = axes.flatten()

        for i, m in enumerate(model_names):
            excel_path = os.path.join("plots", dataset_name, m, "coverage_rates.xlsx")
            if not os.path.exists(excel_path):
                print(f"[WARN] skip {m}, {excel_path} not found")
                continue

            df = pd.read_excel(excel_path, sheet_name="coverage_long")
            df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
            df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
            df["lr_num"] = pd.to_numeric(df["lr"], errors="coerce")

            df["method"] = df["method"].replace({"DDCI_Nex": "DDCI-Nex"})

            sub = df[
                (np.isclose(df["lr_num"], float(lr_fix)))
                & (df["method"].isin(methods_to_plot))
            ]

            ax = axes[i]
            if not sub.empty:
                sns.lineplot(
                    data=sub.sort_values(["method", "epsilon"]),
                    x="epsilon", y="coverage", hue="method", marker="o",
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
        fig.legend(handles, labels, loc="lower center", ncol=len(methods_to_plot),
                   frameon=False)

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        base_name = f"{dataset_name}_allmodels_lr{lr_fix}_eps_coverage"
        for ext in ["pdf", "png"]:
            out_path = os.path.join(root, f"{base_name}.{ext}")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[OK] saved: {out_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot coverage vs epsilon from Excel (super large fonts, clean % labels)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset/pkl name, e.g., GOOGL_test")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Fixed learning rate to filter")
    parser.add_argument("--model", type=str, default="all",
                        help="One of ['ar','prophet','theta','transformer','all']")
    args = parser.parse_args()

    plot_coverage_from_excel(
        dataset_name=args.dataset,
        lr_fix=args.lr,
        model=args.model
    )