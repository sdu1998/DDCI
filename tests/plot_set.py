import os
import os.path as osp
import pickle as pkl
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ===== 全局字体和样式（与您提供的一致）=====
plt.rcParams.update({
    "font.size": 18,        # 默认字体
    "axes.titlesize": 24,   # 图标题
    "axes.labelsize": 20,   # x、y 轴标题
    "xtick.labelsize": 18,  # x 轴刻度
    "ytick.labelsize": 18,  # y 轴刻度
    "legend.fontsize": 18,  # 图例
})

def _get_sets_block(data, method, lr, eps=None, burnin_global=0):
    """
    读取某 method 在指定 (lr, eps) 下的 sets，并在最外层做一次 T_burnin+1 的一致裁剪。
    兼容两种 key 结构：
      1) 普通：results[method][lr]
      2) 带 epsilon：results[method][(lr, eps)]
    返回 (lower, upper) 两个 ndarray（已裁剪至 burnin_global 之后）。
    """
    method_dict = data[method]

    node = None
    if eps is not None:
        key_tuple = (lr, eps)
        if key_tuple in method_dict:
            node = method_dict[key_tuple]

    if node is None and lr in method_dict:
        node = method_dict[lr]

    # 进一步容错：float 精度导致 miss 的情况
    if node is None:
        for k in method_dict.keys():
            if isinstance(k, tuple) and len(k) == 2:
                lr_k, eps_k = k
                if np.isclose(lr_k, lr, atol=1e-12) and (eps is None or np.isclose(eps_k, eps, atol=1e-12)):
                    node = method_dict[k]
                    break
        if node is None:
            for k in method_dict.keys():
                if not isinstance(k, tuple) and np.isclose(k, lr, atol=1e-12):
                    node = method_dict[k]
                    break

    if node is None:
        raise KeyError(f"[{method}] cannot find results for lr={lr}, eps={eps}. "
                       f"Available keys: {list(method_dict.keys())[:8]} ...")

    sets = np.stack(node["sets"])  # shape: (T, 2)
    T_burnin = data["T_burnin"]

    # 与常见逻辑一致：T_burnin+1 之后的数据用于画图
    lower = sets[T_burnin+1:, 0]
    upper = sets[T_burnin+1:, 1]

    # 再应用全局 burnin（命令行 --burnin），与 y 对齐
    if burnin_global > 0:
        lower = lower[burnin_global:]
        upper = upper[burnin_global:]

    return lower, upper


def plot_overview_with_inset_onlymiss_in_inset(
    datetimes, y_true, lower1, upper1, lower2, upper2,
    ecifail_idx=None, ddcifail_idx=None,
    method1="ECI", method2="DDCI", model_name="Model",
    inset_start=None, inset_end=None,
    figsize=(16,8), alpha1=0.45, alpha2=0.45,
    miss_size=20, true_lw=2.0, true_ls='-',
    inset_box=(0.1, 0.5, 0.45, 0.35),
    pad_top_ratio=0.25, save_pdf_path=None
):
    # ---- 清理与对齐 ----
    mask_all = (np.isfinite(y_true) & np.isfinite(lower1) & np.isfinite(upper1)
                & np.isfinite(lower2) & np.isfinite(upper2))
    x = np.asarray(datetimes)[mask_all]
    y = np.asarray(y_true)[mask_all]
    l1, u1 = np.asarray(lower1)[mask_all], np.asarray(upper1)[mask_all]
    l2, u2 = np.asarray(lower2)[mask_all], np.asarray(upper2)[mask_all]

    def _map_fail_idx(idx):
        if idx is None: return np.array([], dtype=int)
        kept = np.where(mask_all)[0]
        pos = np.searchsorted(kept, idx)
        ok  = (pos < len(kept)) & (kept[pos] == idx)
        return pos[ok]

    eci_miss  = _map_fail_idx(ecifail_idx)
    ddci_miss = _map_fail_idx(ddcifail_idx)

    # ---- 颜色 ----
    c1_fill, c2_fill = "#3497d8", "#d6d626"
    c1_miss, c2_miss = "#3855c7", "#d73b3b"

    # ---- 主图 ----
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(x, l1, u1, color=c1_fill, alpha=alpha1, label=f"{method1} interval", zorder=1)
    ax.fill_between(x, l2, u2, color=c2_fill, alpha=alpha2, label=f"{method2} interval", zorder=1)
    ax.plot(x, y, color="green", linewidth=true_lw, linestyle=true_ls, label="True values", zorder=4)

    ax.set_title("Prediction Intervals")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + pad_top_ratio * abs(ymax - ymin))

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # 注册 miss 图例
    ax.scatter([], [], color=c1_miss, marker="x", s=miss_size*0.8, label=f"{method1} miss")
    ax.scatter([], [], color=c2_miss, marker="o", s=miss_size*0.8, label=f"{method2} miss")

    # ---- inset ----
    if inset_start is not None and inset_end is not None:
        mask_in = (x >= np.datetime64(inset_start)) & (x <= np.datetime64(inset_end))
        if mask_in.any():
            x0, y0, w, h = inset_box
            axins = ax.inset_axes([x0, y0, w, h])
            axins.set_facecolor("whitesmoke")

            axins.fill_between(x[mask_in], l1[mask_in], u1[mask_in], color=c1_fill, alpha=alpha1, zorder=1)
            axins.fill_between(x[mask_in], l2[mask_in], u2[mask_in], color=c2_fill, alpha=alpha2, zorder=1)
            axins.plot(x[mask_in], y[mask_in], color="green", linewidth=true_lw, linestyle=true_ls, zorder=4)

            # miss 点只在 inset 内显示
            if len(eci_miss) > 0:
                eci_miss_in  = eci_miss[(x[eci_miss]  >= x[mask_in][0]) & (x[eci_miss]  <= x[mask_in][-1])]
                axins.scatter(x[eci_miss_in],  y[eci_miss_in],  color=c1_miss, marker="x", s=miss_size, zorder=6)
            if len(ddci_miss) > 0:
                ddci_miss_in = ddci_miss[(x[ddci_miss] >= x[mask_in][0]) & (x[ddci_miss] <= x[mask_in][-1])]
                axins.scatter(x[ddci_miss_in], y[ddci_miss_in], color=c2_miss, marker="o", s=miss_size*0.9, zorder=7)

            axins.set_xticks([]); axins.set_yticks([])
            axins.grid(False)

            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#c0c0c0", lw=1.5)

    ax.legend(loc="best", frameon=True, ncol=2)
    plt.tight_layout()

    # ---- 仅保存 PDF ----
    if save_pdf_path is None:
        raise ValueError("save_pdf_path must be provided.")
    os.makedirs(osp.dirname(save_pdf_path), exist_ok=True)
    if not save_pdf_path.lower().endswith(".pdf"):
        save_pdf_path = save_pdf_path + ".pdf"
    plt.savefig(save_pdf_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {save_pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot intervals with inset zoom (ε shown in legend, PDF only)")
    parser.add_argument("--filename", type=str, required=True, help="Path to pickle file")
    parser.add_argument("--method1", type=str, default="ECI")
    parser.add_argument("--method2", type=str, default="DDCI")
    parser.add_argument("--lr1", type=float, default=0.5)
    parser.add_argument("--lr2", type=float, default=0.5)
    parser.add_argument("--eps1", type=float, default=None, help="epsilon for method1 (if applicable)")
    parser.add_argument("--eps2", type=float, default=None, help="epsilon for method2 (if applicable)")
    parser.add_argument("--burnin", type=int, default=0, help="Extra coverage burn-in applied after T_burnin+1")
    parser.add_argument("--inset-start", type=str, default=None)
    parser.add_argument("--inset-end", type=str, default=None)
    parser.add_argument("--save-root", type=str, default="plots", help="Root directory to save figures")
    args = parser.parse_args()

    # 提取 pkl 文件名（不含扩展名）
    pkl_name = osp.splitext(osp.basename(args.filename))[0]

    with open(args.filename, "rb") as f:
        all_data = pkl.load(f)

    # legend 标签（带 epsilon）
    m1_label = args.method1 if args.eps1 is None else f"{args.method1} (ε={args.eps1})"
    m2_label = args.method2 if args.eps2 is None else f"{args.method2} (ε={args.eps2})"

    for model_name, data in all_data.items():
        if not isinstance(data, dict) or "data" not in data:
            continue  # 跳过全局元数据等

        T_burnin = data["T_burnin"]
        # 真实值与时间轴（与 sets 的 T_burnin+1 对齐）
        y_full = data["data"]["y"].to_numpy(float)[T_burnin+1:]
        datetimes_full = data["data"].index[T_burnin+1:]

        # 再应用命令行 --burnin
        y_plot = y_full[args.burnin:]
        datetimes = datetimes_full[args.burnin:]

        # 读取两组 sets（自动处理是否带 epsilon 的键）
        lower1, upper1 = _get_sets_block(data, args.method1, args.lr1, eps=args.eps1, burnin_global=args.burnin)
        lower2, upper2 = _get_sets_block(data, args.method2, args.lr2, eps=args.eps2, burnin_global=args.burnin)

        # 对齐并剔除 NaN/inf
        mask = (np.isfinite(y_plot) & np.isfinite(lower1) & np.isfinite(upper1)
                & np.isfinite(lower2) & np.isfinite(upper2))
        y_plot, lower1, upper1, lower2, upper2, datetimes = (
            y_plot[mask], lower1[mask], upper1[mask],
            lower2[mask], upper2[mask], datetimes[mask]
        )

        # 计算 miss 索引（相对于已裁剪后的序列）
        miss1_idx = np.where((y_plot < lower1) | (y_plot > upper1))[0]
        miss2_idx = np.where((y_plot < lower2) | (y_plot > upper2))[0]

        # 仅 PDF 保存路径
        save_dir = osp.join(args.save_root, pkl_name, "set_plot")
        fname = f"{model_name}_{args.method1}{'' if args.eps1 is None else f'_eps{args.eps1}'}_vs_" \
                f"{args.method2}{'' if args.eps2 is None else f'_eps{args.eps2}'}"
        save_pdf_path = osp.join(save_dir, fname + ".pdf")

        plot_overview_with_inset_onlymiss_in_inset(
            datetimes, y_plot, lower1, upper1, lower2, upper2,
            ecifail_idx=miss1_idx, ddcifail_idx=miss2_idx,
            method1=m1_label, method2=m2_label, model_name=model_name,
            inset_start=args.inset_start, inset_end=args.inset_end,
            figsize=(14, 8),
            save_pdf_path=save_pdf_path
        )