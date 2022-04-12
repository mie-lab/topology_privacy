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


def feature_comparison_table(in_path, use_acc=10):
    feature_comp_dict = {}
    for metric in ["kldiv", "mse", "wasserstein", "all"]:
        for feats in ["in_degree", "out_degree", "shortest_path", "centrality", "combined"]:
            if metric == "all" and feats != "combined":
                continue
            feature_comp_dict[(metric, feats)] = {}
            for study in ["gc1", "gc2"]:
                mean = pd.read_csv(
                    os.path.join(in_path, study, "acc_k0", f"mean_{metric}_{feats}.csv"), index_col="p_duration"
                )
                mean_acc = pd.read_csv(
                    os.path.join(in_path, study, "acc_k" + str(use_acc), f"mean_{metric}_{feats}.csv"),
                    index_col="p_duration",
                )
                feature_comp_dict[(metric, feats)].update(
                    {
                        (study, "Mean RR"): np.nanmean(mean.values),
                        (study, "Max RR"): np.nanmax(mean.values),
                        (study, f"Mean {use_acc}-Acc"): np.nanmean(mean_acc.values),
                        (study, f"Max {use_acc}-Acc"): np.nanmax(mean_acc.values),
                    }
                )
    df_raw = pd.DataFrame(feature_comp_dict)
    df = df_raw.swapaxes(1, 0)
    print(df)
    print(df.to_latex(float_format="%.2f"))


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
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def privacy_loss_plot(gc1_ranks, gc2_ranks):

    # implementation of privacy loss according to paper
    def prob_informed(rank_df):
        rank_df["recip_rank"] = rank_df["rank"].apply(lambda x: 1 / x)
        sum_recip_rank = rank_df.groupby("u_user_id").agg({"recip_rank": "sum"})
        rank_df = rank_df.merge(sum_recip_rank, how="left", left_on="u_user_id", right_index=True)
        rank_df["prob_informed"] = rank_df["recip_rank_x"] / rank_df["recip_rank_y"]
        return rank_df

    def privacy_loss(rank_df, nr_ranks):
        rank_df_same_user = rank_df[rank_df["p_user_id"] == rank_df["u_user_id"]]
        rank_df_same_user["privacy_loss"] = rank_df_same_user["prob_informed"] / (1 / nr_ranks) - 1
        return rank_df_same_user

    prob_informed_df = prob_informed(gc1_ranks.copy())
    out_gc1 = privacy_loss(prob_informed_df, len(prob_informed_df["p_user_id"].unique()))

    prob_informed_df = prob_informed(gc2_ranks.copy())
    out_gc2 = privacy_loss(prob_informed_df, len(prob_informed_df["p_user_id"].unique()))
    print("Median privacy loss")
    print(
        np.median(out_gc1["privacy_loss"]),
        np.mean(out_gc1["privacy_loss"]),
        "GC2:",
        np.median(out_gc2["privacy_loss"]),
        np.mean(out_gc2["privacy_loss"]),
    )
    print("Decrease of rank")
    print(np.mean(out_gc1["rank"]), "vs random strategy", len(gc1_ranks["p_user_id"].unique()) // 2)
    print(np.mean(out_gc2["rank"]), "vs random strategy", len(gc2_ranks["p_user_id"].unique()) // 2)
    plt.rcParams.update({"font.size": 15})

    plt.figure(figsize=(5, 5))
    plt.boxplot([out_gc1["rank"], out_gc2["rank"]])
    plt.xticks([1, 2], ["Green Class 1", "Green Class 2"])
    plt.ylabel("Rank")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "rank_dist.pdf"))
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.boxplot([out_gc1["privacy_loss"], out_gc2["privacy_loss"]])
    plt.xticks([1, 2], ["Green Class 1", "Green Class 2"])
    plt.ylabel("Privacy loss")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "privacy_loss.pdf"))
    plt.show()


if __name__ == "__main__":
    study = "gc2"

    in_path = os.path.join("outputs", study)
    save_path = os.path.join("1paper", "figures", "results")
    use_metric = "mse_combined"

    # plot matrix
    for acc in ["acc_k0", "acc_k5", "acc_k10"]:
        mean = pd.read_csv(os.path.join(in_path, acc, f"mean_{use_metric}.csv"), index_col="p_duration")
        std = pd.read_csv(os.path.join(in_path, acc, f"std_{use_metric}.csv"), index_col="p_duration")
        out_name = f"{study}_{acc}_{use_metric}.pdf"
        plot_matrix(mean, std, os.path.join(save_path, out_name))

    # Table for feature comparison
    feature_comparison_table("outputs")
