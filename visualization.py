import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


def plot_matrix(mean, std=None, save_path=None):
    if std is None:
        std = np.zeros(mean.shape) + np.nan

    def data_to_label(data, text):
        label = []
        for data, text in zip(data.flatten(), text.flatten()):
            if pd.isna(text):
                label.append("{0:.2f}".format(data))
            else:
                label.append("{0:.2f}\n".format(data) + "\u00B1" + "{0:.2f}".format(text))
        return np.asarray(label).reshape(mean.shape)

    sns.set(font_scale=1.5)
    labels = data_to_label(np.array(mean), np.array(std))
    print(labels.shape, mean.shape)
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean, annot=labels, fmt="", cmap="YlGnBu")
    plt.xticks(np.arange(7) + 0.5, np.arange(4, 32, 4))
    plt.yticks(np.arange(7) + 0.5, np.arange(4, 32, 4))
    plt.xlabel("Duration user")
    plt.ylabel("Duration pool")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def feature_comparison_table(path_to_acc):
    feature_comp_dict = {}
    for feats in ["in_degree", "out_degree", "shortest_path", "centrality", "combined"]:
        mean = pd.read_csv(os.path.join(path_to_acc, f"mean_kldiv_{feats}.csv"), index_col="p_duration")
        mean_28 = np.mean(mean["28"].values)
        mean_all = np.nanmean(mean.values)
        feature_comp_dict[feats] = {
            "Mean reciprocal rank (p_duration=28)": mean_28,
            "Mean reciprocal rank (overall)": mean_all,
            "Max. reciprocal rank": np.nanmax(mean.values),
        }
    print(pd.DataFrame(feature_comp_dict).to_latex(float_format="%.2f"))


def plot_intra_inter(rank_df, save_path=None):
    # intra user
    std_rank_per_user = (
        rank_df.groupby(["p_duration", "u_user_id"])
        .agg({"rank": "std", "u_user_id": "count"})
        .rename(columns={"rank": "rank_std", "u_user_id": "nr_samples"})
    )
    intra_user = std_rank_per_user.reset_index().groupby("p_duration").agg({"rank_std": ["mean", "std"]})

    # inter user
    # v1
    # mean_rank_per_user = rank_df.groupby(["p_duration", "u_user_id"]).agg({"rank": "mean"})
    # inter_user = (
    #     mean_rank_per_user.reset_index().groupby("p_duration").agg({"rank": "std"}).rename(columns={"rank": "rank_std"})
    # )
    std_rank_per_bin = (
        rank_df.groupby(["p_duration", "u_duration", "u_filename", "p_filename"])
        .agg({"rank": "std"})
        .rename(columns={"rank": "rank_std"})
    )
    inter_user = std_rank_per_bin.reset_index().groupby("p_duration").agg({"rank_std": ["mean", "std"]})

    offset = 0.25
    plt.errorbar(
        inter_user.index - offset,
        inter_user[("rank_std", "mean")],
        yerr=inter_user[("rank_std", "std")],
        fmt="o",
        ecolor="r",
        color="r",
        markersize=10,
    )
    plt.errorbar(
        intra_user.index + offset,
        intra_user[("rank_std", "mean")],
        yerr=intra_user[("rank_std", "std")],
        fmt="o",
        ecolor="b",
        color="b",
        markersize=10,
    )
    plt.xticks(inter_user.index, inter_user.index)
    plt.xlabel("Tracking period of pool")
    plt.ylabel("Standard deviation of rank")

    legend_elements = [
        Line2D([0], [0], marker="o", color="b", lw=2, label="Intra-user std", markerfacecolor="b", markersize=10),
        Line2D([0], [0], marker="o", color="r", lw=2, label="Inter-user std", markerfacecolor="r", markersize=10),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # matrix
    save_path = "../topology_privacy/1paper/figures"
    mean = pd.read_csv(
        "../topology_privacy/outputs/v1_wrong_std/acc_k10/mean_kldiv_combined.csv", index_col="p_duration"
    )
    std = pd.read_csv("../topology_privacy/outputs/v1_wrong_std/acc_k10/std_kldiv_combined.csv", index_col="p_duration")
    out_name = "acc_kldiv.pdf"
    plot_matrix(mean, std, os.path.join(save_path, out_name))

    # Table for feature comparison
    path_to_acc = "../topology_privacy/outputs/acc_k0/"
    feature_comparison_table(path_to_acc)
