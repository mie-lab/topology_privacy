import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.linear_model import LinearRegression
from utils import get_engine
import statsmodels.api as sm


def plot_matrix(mean, std=None, save_path=None, include_1week=True):
    """
    Main plot: Matrix with reidentification accuracy
    Arguments:
        mean: np array
            mean accuracy for all combinations
        std: Optional np array (same shape as mean)
            Can be provided to be added in the matrix
        save_path: str
            filepath to save the figure
    """
    # set std to nan if not provided
    if std is None:
        std = np.zeros(mean.shape) + np.nan

    def data_to_label(data, text):
        """Construct cell labels from mean and std"""
        label = []
        for data, text in zip(data.flatten(), text.flatten()):
            if pd.isna(text):
                label.append("{0:.2f}".format(data))
            else:
                label.append("{0:.2f}\n".format(data) + "\u00B1" + "{0:.2f}".format(text))
        return np.asarray(label).reshape(mean.shape)

    sns.set(font_scale=1.5)
    # construct labels
    labels = data_to_label(np.array(mean), np.array(std))
    print(labels.shape, mean.shape)
    # make heatmap with labels in the cells
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean, annot=labels, fmt="", cmap="YlGnBu")
    if include_1week:
        plt.xticks(np.arange(9) + 0.5, [1, 2] + np.arange(4, 32, 4).tolist(), fontsize=20)
        plt.yticks(np.arange(9) + 0.5, [1, 2] + np.arange(4, 32, 4).tolist(), fontsize=20)
    else:
        plt.xticks(np.arange(7) + 0.5, np.arange(4, 32, 4), fontsize=20)
        plt.yticks(np.arange(7) + 0.5, np.arange(4, 32, 4), fontsize=20)
    plt.xlabel("Duration test user", fontsize=25)
    plt.ylabel("Duration pool", fontsize=25)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def feature_comparison_table(in_path, out_path="1journal_paper"):
    """
    Compare which features enable reidentification
    in_path: str
        Path to matrices with reidentification performances
    """
    feature_comp_dict = {}
    # For all metrics and features
    for metric in ["kldiv", "mse", "wasserstein", "all"]:
        for feats in ["transition", "in_degree", "out_degree", "shortest_path", "centrality", "combined"]:
            if metric == "all" and feats != "combined":
                continue
            feature_comp_dict[(metric, feats)] = {}
            # for all accuracy conditions
            for use_acc in [0, 1, 5, 10]:
                # read mean accuracy values
                mean_acc = pd.read_csv(
                    os.path.join(in_path, study, "acc_k" + str(use_acc), f"mean_{metric}_{feats}.csv"),
                    index_col="p_duration",
                )
                if use_acc == 0:
                    lab = "Recip. rank"
                else:
                    lab = str(use_acc) + "-Accuracy"
                # save in dictionary
                feature_comp_dict[(metric, feats)].update(
                    {(lab, "Mean"): np.nanmean(mean_acc.values), (lab, "Max"): np.nanmax(mean_acc.values),}
                )
    # output table in latex format
    df_raw = pd.DataFrame(feature_comp_dict)
    df = df_raw.swapaxes(1, 0)
    print(df)
    with open(out_path, "w") as outfile:
        print(df.to_latex(float_format="%.2f"), file=outfile)


def plot_intra_inter(rank_df, save_path="1journal_paper"):
    """
    Compare intra and inter user differences - 
    Is the variance explained by differences between users or by differences between time periods?
    Arguments:
        rank_df: Dataframe with the rank of the ground-truth match for each user in each time period
        save_path: str - Filepath
    Saves bar plot if save_path is provided
    """
    rank_df = rank_df[(rank_df["u_duration"] != 2) & (rank_df["p_duration"] != 2)]
    # intra user
    std_rank_per_user = (
        rank_df.groupby(["p_duration", "u_user_id"])
        .agg({"rank": "std", "u_user_id": "count"})
        .rename(columns={"rank": "rank_std", "u_user_id": "nr_samples"})
    )
    intra_user = std_rank_per_user.reset_index().groupby("p_duration").agg({"rank_std": ["mean", "std"]})

    # inter user
    std_rank_per_bin = (
        rank_df.groupby(["p_duration", "u_duration", "u_filename", "p_filename"])
        .agg({"rank": "std"})
        .rename(columns={"rank": "rank_std"})
    )
    inter_user = std_rank_per_bin.reset_index().groupby("p_duration").agg({"rank_std": ["mean", "std"]})

    # Barplot
    width = 1
    offset = width / 2
    plt.bar(inter_user.index - offset, inter_user[("rank_std", "mean")], width=width, label="Inter-user variance")
    plt.bar(intra_user.index + offset, intra_user[("rank_std", "mean")], width=width, label="Intra-user variance")
    plt.xticks(inter_user.index, inter_user.index, fontsize=15)
    plt.xlabel("Tracking period of pool", fontsize=15)
    plt.ylabel("Standard deviation of rank", fontsize=15)
    # plt.ylim(0, 45)
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def privacy_loss_plot(gc1_ranks, gc2_ranks, save_path="1journal_paper"):
    """Plot privacy loss according to Manousakas et al (Figure 3 in paper)

    Parameters
    ----------
    gc1_ranks : Rank of real user id in matching 
    gc2_ranks : Rank of real user id in matching (gc2 dataset)
    save_path : std
        filepath to save output figure
    """
    # implementation of privacy loss according to paper
    def privacy_loss(rank_df, nr_users):
        rank_df["recip_rank"] = 1 / rank_df["rank"]
        sum_all_recip_ranks = np.sum([1 / (i + 1) for i in range(nr_users)])
        # sum_recip_rank = rank_df.groupby("u_user_id").agg({"recip_rank": "sum"})
        # rank_df = rank_df.merge(sum_recip_rank, how="left", left_on="u_user_id", right_index=True)
        rank_df["prob_informed"] = rank_df["recip_rank"] / sum_all_recip_ranks
        # divide the informed prob by the uninformed prob (just random pick of the users, so 1 / nr of users)
        rank_df["privacy_loss"] = rank_df["prob_informed"] / (1 / nr_users) - 1
        return rank_df

    out_gc1 = privacy_loss(gc1_ranks.copy(), gc1_ranks["u_user_id"].nunique())
    out_gc2 = privacy_loss(gc2_ranks.copy(), nr_users=gc2_ranks["u_user_id"].nunique())
    print("Median privacy loss")
    print(
        np.median(out_gc1["privacy_loss"]),
        np.mean(out_gc1["privacy_loss"]),
        "GC2:",
        np.median(out_gc2["privacy_loss"]),
        np.mean(out_gc2["privacy_loss"]),
    )
    print("Decrease of rank")
    print(np.mean(out_gc1["rank"]), "vs random strategy", gc1_ranks["u_user_id"].nunique() // 2)
    print(np.mean(out_gc2["rank"]), "vs random strategy", gc2_ranks["u_user_id"].nunique() // 2)
    plt.rcParams.update({"font.size": 15})

    plt.figure(figsize=(5, 5))
    widths = [0.25, 0.25]
    plt.boxplot([out_gc1["rank"], out_gc2["rank"]], widths=widths)
    plt.plot([1 - widths[0], 1 + widths[0]], [139 / 2, 139 / 2], c="green", label="random reference", linestyle="--")
    plt.plot([2 - widths[0], 2 + widths[0]], [49 / 2, 49 / 2], c="green", linestyle="--")
    plt.legend()
    plt.xticks([1, 2], ["Green Class 1", "Green Class 2"])
    plt.ylabel("Rank")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "rank_dist.pdf"))
    # plt.show()

    plt.figure(figsize=(5, 5))
    plt.boxplot([out_gc1["privacy_loss"], out_gc2["privacy_loss"]])
    plt.xticks([1, 2], ["Green Class 1", "Green Class 2"])
    plt.ylabel("Privacy loss")
    # plt.ylim(-1, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "privacy_loss.pdf"))
    # plt.show()


def regression_analysis(in_path="../topology_privacy/outputs/gc1", out_path="1journal_paper"):
    """
    Analyze the influence of pool and test tracking duration on the matching accuracy
    Loads the MSE matrix, applies regression analysis and prints the result as a latex table.
    """
    week_bins = [1, 2] + [(i + 1) * 4 for i in range(7)]
    out_df = {}
    for k in [0, 1, 5, 10]:
        # load matrix
        res = pd.read_csv(os.path.join(in_path, f"acc_k{k}", "mean_mse_combined.csv"), index_col="p_duration")
        res_arr = np.array(res)
        # Run linear regression
        X = np.array([[(w, v) for w in week_bins] for v in week_bins]).reshape(-1, 2)
        diff = np.abs(X[:, 0] - X[:, 1])
        X = np.hstack((X, np.expand_dims(diff, 1)))
        Y = res_arr.flatten()
        X = X[~np.isnan(Y)]
        Y = Y[~np.isnan(Y)]
        reg = LinearRegression().fit(X, Y)
        # save coefficients
        out_df[k] = reg.coef_.tolist() + [reg.intercept_] + [reg.score(X, Y)]

        X2 = sm.add_constant(X)
        est = sm.OLS(Y, X2)
        est2 = est.fit()
        p_values = est2.pvalues.tolist()
        # p_value_dict[k] = p_values
        print("All p values significant?", np.array(p_values) < 0.05)

    # rename for clean table
    out_df = (
        pd.DataFrame(out_df)
        .rename(
            columns={0: "MRR", 1: "1-Accuracy", 5: "5-Accuracy", 10: "10-Accuracy"},
            index={
                0: "test duration",
                1: "pool duration",
                2: "Absolute difference between pool and test duration",
                3: "Intercept",
                4: "R2 score",
            },
        )
        .swapaxes(1, 0)
    )
    # output
    print(out_df)

    with open(out_path, "w") as outfile:
        print(out_df.to_latex(float_format="%.2f"), file=outfile)


def inbetween_analysis(df_rank, out_path="1journal_paper"):
    assert all(df_rank["u_filename"] > df_rank["p_filename"])
    # convert to datatime
    df_rank["p_filename"] = pd.to_datetime(df_rank["p_filename"])
    df_rank["u_filename"] = pd.to_datetime(df_rank["u_filename"])
    df_rank["Weeks between pool-bin and user-bin"] = (
        (df_rank["u_filename"] - df_rank["p_filename"]).dt.total_seconds() / (3600 * 24 * 7)
    ).astype(int)

    top10acc = lambda x: sum(x <= 10) / len(x)
    top10_acc_plot = df_rank.groupby(["Weeks between pool-bin and user-bin"]).agg({"rank": top10acc}).reset_index()
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.figure(figsize=(9, 4.5))
    sns.barplot(
        data=top10_acc_plot,
        x="Weeks between pool-bin and user-bin",
        y="rank",
        hatch="/",
        edgecolor="black",
        color="lightgrey",
    )
    plt.ylabel("Top-10-Accuracy")
    # # reciprocal rank instead
    # df_rank["Reciprocal rank"] = 1 / df_rank["rank"]
    # sns.barplot(
    #     data=df_rank,
    #     x="Weeks between pool-bin and user-bin",
    #     y="Reciprocal rank",
    #     hatch="/",
    #     edgecolor="black",
    #     color="lightgrey",
    # )
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(os.path.join(out_path, "inbetween_experiment.pdf"))
    else:
        plt.show()


if __name__ == "__main__":
    engine = get_engine(DBLOGIN_FILE="dblogin.json")
    study = "gc1"
    out_path = "1journal_paper"

    in_path = os.path.join("outputs", study)
    use_metric = "mse_combined"

    df_rank = pd.read_sql(f"SELECT * FROM {study}.user_ranking_{use_metric}", engine)
    plot_intra_inter(df_rank, os.path.join(out_path, f"{study}_inter_intra.pdf"))

    # plot matrix for each accuracy
    for acc in ["acc_k0", "acc_k1", "acc_k5", "acc_k10"]:
        # read mean and std of matching performance
        mean = pd.read_csv(os.path.join(in_path, acc, f"mean_{use_metric}.csv"), index_col="p_duration")
        std = pd.read_csv(os.path.join(in_path, acc, f"std_{use_metric}.csv"), index_col="p_duration")
        out_name = f"{study}_{acc}_{use_metric}.pdf"
        plot_matrix(mean, std, os.path.join(out_path, out_name))

    # Table for feature comparison
    feature_comparison_table("outputs", out_path=os.path.join(out_path, f"{study}_feature_comparison.txt"))

    # Regression analysis (Table 1 in paper)
    regression_analysis(
        in_path=os.path.join("outputs", study), out_path=os.path.join(out_path, f"{study}_regression_analysis.txt")
    )

    # # privacy loss plot
    # gc1_ranks = pd.read_sql(
    #     f"SELECT * FROM gc1.user_ranking_{use_metric}  WHERE u_duration=28 AND p_duration=28", engine
    # )
    # gc2_ranks = pd.read_sql(
    #     f"SELECT * FROM gc2.user_ranking_{use_metric} WHERE u_duration=28 AND p_duration=28", engine
    # )
    # privacy_loss_plot(gc1_ranks, gc2_ranks, out_path)

    # # duration inbetween plot
    # df_rank = pd.read_sql(f"SELECT * FROM gc1.user_ranking_inbetween_experiment", engine)
    # inbetween_analysis(df_rank, out_path)
