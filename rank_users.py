import pandas as pd
from utils import get_engine


def clean_impossible_matches(df):
    """Delete impossible tasks from data (user not in pool)."""
    df_ix = df.index
    df_ = df.set_index(["p_duration", "u_duration", "p_filename", "u_filename", "u_user_id"])
    df_["df_ix"] = df_ix

    sum_same_user_by_task = df.groupby(by=["p_duration", "u_duration", "p_filename", "u_filename", "u_user_id"])[
        "same_user"
    ].sum()
    impossible_task_ix = sum_same_user_by_task[sum_same_user_by_task < 1].index

    df_ix_to_delete = df_.loc[impossible_task_ix, "df_ix"]
    return df.drop(df_ix_to_delete)


def rank_users(engine, study="gc1", table_name="distance_1w"):

    distances_query = f"SELECT * FROM {study}.{table_name}"  # WHERE p_duration=28 and u_duration=28"
    feature_cross_product_df = pd.read_sql(distances_query, con=engine)
    feature_cross_product_df["same_user"] = (
        feature_cross_product_df["p_user_id"] == feature_cross_product_df["u_user_id"]
    )
    print("Read distances")
    feature_cross_product_df = clean_impossible_matches(feature_cross_product_df)
    print("Cleaned impossible matches")

    # Compute combined distances
    for metric in ["kldiv", "mse", "wasserstein"]:
        feature_cross_product_df[f"{metric}_combined"] = (
            feature_cross_product_df[f"{metric}_in_degree"]
            + feature_cross_product_df[f"{metric}_out_degree"]
            + feature_cross_product_df[f"{metric}_shortest_path"]
            + feature_cross_product_df[f"{metric}_transition"]
        )
    # Sum up all similarities
    feature_cross_product_df["all_combined"] = (
        feature_cross_product_df["kldiv_combined"]
        + feature_cross_product_df["mse_combined"]
        + feature_cross_product_df["wasserstein_combined"]
    )
    # collect all possibilities
    possible_cols = []
    for metric in ["kldiv", "mse", "wasserstein"]:
        for feats in ["in_degree", "out_degree", "shortest_path", "centrality", "transition", "combined"]:
            possible_cols.append(metric + "_" + feats)
    possible_cols.append("all_combined")

    grouped_feature_distances = feature_cross_product_df.groupby(
        by=["u_user_id", "u_duration", "u_filename", "p_duration", "p_filename"]
    )
    print("Grouped by duration and filename")
    for dist_col in possible_cols:
        feature_cross_product_df["rank"] = grouped_feature_distances[dist_col].rank()
        df_rank_filtered = feature_cross_product_df[feature_cross_product_df["same_user"]]
        df_rank_filtered = df_rank_filtered[
            ["p_duration", "p_filename", "u_user_id", "u_duration", "u_filename", "rank"]
        ]
        df_rank_filtered.to_sql("user_ranking_" + dist_col, engine, study, if_exists="replace")
        print("Ranks written to DB for distance", dist_col, len(df_rank_filtered), df_rank_filtered["rank"].mean())


if __name__ == "__main__":
    engine = get_engine(DBLOGIN_FILE="dblogin.json")
    rank_users(engine, "gc1")
