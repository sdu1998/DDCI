import os
import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def moving_average(x, window=50):
    x = np.array(x)
    ma = np.zeros(x.shape)
    for i in range(x.shape[0]):
        ma[i] = np.mean(x[max(0, i - window + 1):i + 1])
    return ma


def plot_coverage(filename, method1, lr1, method2, lr2, window, burnin, save_root="plots"):
    pkl_name = os.path.splitext(os.path.basename(filename))[0]

    with open(filename, "rb") as f:
        all_data = pkl.load(f)

    for model_name in all_data.keys():
        data = all_data[model_name]
        T_burnin = data["T_burnin"]

        sets1 = [np.stack(data[method1][lr1]["sets"])[T_burnin + 1:, 0],
                 np.stack(data[method1][lr1]["sets"])[T_burnin + 1:, 1]]
        sets2 = [np.stack(data[method2][lr2]["sets"])[T_burnin + 1:, 0],
                 np.stack(data[method2][lr2]["sets"])[T_burnin + 1:, 1]]

        y = data["data"]["y"].to_numpy(dtype=float)[T_burnin + 1:]
        alpha = data["alpha"]

        covered1 = ((y >= sets1[0]) & (y <= sets1[1])).astype(float)
        covered2 = ((y >= sets2[0]) & (y <= sets2[1])).astype(float)

        coverage1 = moving_average(covered1, window=window)[burnin:]
        coverage2 = moving_average(covered2, window=window)[burnin:]

        datetimes = data["data"].index[T_burnin + 1:][burnin:]

        fig, ax = plt.subplots(figsize=(10, 4), sharex=True, sharey=False)

        ax.plot(datetimes, coverage1, label=f"{method1} (lr={lr1})", color="blue")
        ax.plot(datetimes, coverage2, label=f"{method2} (lr={lr2})", color="orange")
        ax.axhline(1 - alpha, color="red", linestyle="--", label="Target coverage")

        ax.set_ylabel("Coverage", fontsize=15)
        ax.set_xlabel("Time", fontsize=15)
        ax.set_title(f"{model_name}", fontsize=15)

        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax])
        ax.set_yticks([0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.yaxis.set_tick_params(labelsize=13)

        fig.autofmt_xdate()
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_dir = os.path.join(save_root, pkl_name, "coverage_plot")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name}_{method1}_vs_{method2}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot coverage comparison")
    parser.add_argument("--filename", type=str, required=True, help="Path to pickle file")
    parser.add_argument("--method1", type=str, default="ECI")
    parser.add_argument("--method2", type=str, default="DDCI")
    parser.add_argument("--lr1", type=float, default=0.5)
    parser.add_argument("--lr2", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=50, help="Moving average window length")
    parser.add_argument("--burnin", type=int, default=0, help="Coverage burn-in")
    parser.add_argument("--save-root", type=str, default="plots", help="Root directory to save figures")

    args = parser.parse_args()
    plot_coverage(args.filename, args.method1, args.lr1, args.method2, args.lr2,
                  args.window, args.burnin, args.save_root)