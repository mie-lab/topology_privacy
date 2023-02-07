import pandas as pd
from utils import get_engine
import os
from visualization import plot_intra_inter


def calculate_reciprocal_rank(df_rank_filtered, k=10, return_reciprocal=False):
    """ 
    Calculate reciprocal rank 
    """
    # compute reciprocal ranks
    if return_reciprocal:
        df_rank_filtered["reciprocal_rank"] = 1 / df_rank_filtered["rank"]
        matrix_elements = df_rank_filtered.groupby(by=["p_duration", "u_duration"])["reciprocal_rank"].agg(
            ["mean", "std"]
        )
        mean_matrix = matrix_elements["mean"].unstack(level=-1, fill_value=None)
        std_matrix = matrix_elements["std"].unstack(level=-1, fill_value=None)
        return mean_matrix, std_matrix

    # Compute topk accuracy
    df_rank_filtered["topk"] = (df_rank_filtered["rank"] <= k).astype(int)
    df_rank_top_acc = df_rank_filtered.groupby(by=["u_duration", "p_duration", "u_filename", "p_filename"]).agg(
        {"topk": "mean"}
    )
    matrix_elements = df_rank_top_acc.groupby(by=["p_duration", "u_duration"])["topk"].agg(["mean", "std"])
    # times 100 for accuracy
    mean_matrix = matrix_elements["mean"].unstack(level=-1, fill_value=None) * 100
    std_matrix = matrix_elements["std"].unstack(level=-1, fill_value=None) * 100
    return mean_matrix, std_matrix


def calculate_topk_accuracy(df, k, distance_column="distance"):
    """Alternative function --> the same can also be computed with the reciprocal rank function"""
    # get best guesses per group ranks
    # get minimum row index per group for each block of [p_duration, p_user, p_start_date(=p_filename) u_duration]
    # https://stackoverflow.com/a/71130117/16232216

    min_row_ix_by_group = (
        df.groupby(by=["p_duration", "u_user_id", "u_duration", "p_filename", "u_filename"])[distance_column]
        .nsmallest(k, keep="all")
        .index.get_level_values(-1)
    )
    # top k guesses
    min_by_group = df.loc[min_row_ix_by_group]

    # Group by group and calculate if a single group had a correct guess (yes or no). Column same user is boolean
    correct_guess_by_group = (
        min_by_group.groupby(by=["p_duration", "u_user_id", "u_duration", "p_filename", "u_filename"])["same_user"]
        .sum()
        .reset_index()
    )

    # Groups are currently: ['p_duration', 'p_user_id', 'u_duration', 'p_file_name']
    # Groups for matrix are: ['p_duration', 'u_duration']
    # we now aggregate to the final matrix and calculate mean + standard deviation

    # aggregate users for accuracy
    acc_by_group = correct_guess_by_group.groupby(by=["u_duration", "p_duration", "u_filename", "p_filename"]).agg(
        {"same_user": "mean"}
    )

    # aggregate to matrix elements
    matrix_elements = acc_by_group.groupby(by=["p_duration", "u_duration"])["same_user"].agg(["mean", "std"])

    mean_matrix = matrix_elements["mean"].unstack(level=-1, fill_value=None) * 100
    std_matrix = matrix_elements["std"].unstack(level=-1, fill_value=None) * 100

    return mean_matrix, std_matrix


if __name__ == "__main__":

    STUDY = "gc1"
    engine = get_engine(DBLOGIN_FILE="dblogin.json")

    # make output dir
    output_base_path = os.path.join("outputs", STUDY)
    os.makedirs(output_base_path, exist_ok=True)

    # RUN for all:
    # collect all possibilities
    possible_cols = []
    for metric in ["kldiv", "mse", "wasserstein"]:
        for feats in ["in_degree", "out_degree", "shortest_path", "centrality", "transition", "combined"]:
            possible_cols.append(metric + "_" + feats)
    possible_cols.append("all_combined")

    # iterate over top-k --> 0-accuracy means reciprocal rank
    for k in [0, 1, 5, 10]:
        out_path = os.path.join(output_base_path, "acc_k" + str(k))
        os.makedirs(out_path, exist_ok=True)
        # Run on all dist types
        for dist_col in possible_cols:
            df_rank = pd.read_sql(f"SELECT * FROM {STUDY}.user_ranking_{dist_col}", engine)
            if k == 0:  # encoding for reciprocal rank:
                mean_matrix, std_matrix = calculate_reciprocal_rank(df_rank, return_reciprocal=True)
            else:
                # accuracy can also be calculated via reciprocal rank function
                mean_matrix, std_matrix = calculate_reciprocal_rank(df_rank, k=k)
            # save mean and std as output
            mean_matrix.to_csv(os.path.join(out_path, "mean_" + dist_col + ".csv"))
            std_matrix.to_csv(os.path.join(out_path, "std_" + dist_col + ".csv"))
