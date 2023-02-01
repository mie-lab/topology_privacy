import pandas as pd
import numpy as np
import os
import sqlalchemy
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, pearsonr

from graph_trackintel.graph_utils import get_largest_component
from graph_trackintel.io import read_graphs_from_postgresql
from graph_trackintel.analysis.graph_features import (
    journey_length,
    degree_beta,
    transition_beta,
    hub_size,
    median_trip_distance,
    highest_decile_distance,
)
from utils import get_engine
from fill_matrix import clean_impossible_matches


def get_q_for_col(col, questions):
    if col[0] == "q":
        col_to_qname = "Q" + col.split("_")[0][1:]
    else:
        col_to_qname = col
    try:
        corresponding_q = questions.loc[col_to_qname]["question"]
    except KeyError:
        corresponding_q = col
    if not isinstance(corresponding_q, str):
        corresponding_q = corresponding_q.iloc[0]
    return corresponding_q


def compute_graph_features():
    print("----------------- GET TIME BINS FOR GC 1 and 2 --------------------")
    con = get_engine(return_con=True, DBLOGIN_FILE="dblogin.json")
    engine = get_engine(DBLOGIN_FILE="dblogin.json")

    for study in ["gc1", "gc2"]:
        # initialize output table
        table_precomputed_feats = []
        # Run for each time period
        for weeks in [28]:
            print("processing weeks:", weeks, "STUDY", study)
            cur = con.cursor()

            # get the timebin names
            cur.execute(f"SELECT name FROM {study}.dur_{weeks}w")

            table_name = f"dur_{weeks}w"
            all_names = cur.fetchall()

            # iterate over all tables for this specific period
            for name in all_names:
                file_name = name[0]
                # get graphs
                graph_dict = read_graphs_from_postgresql(
                    graph_table_name=table_name,
                    psycopg_con=con,
                    graph_schema_name=study,
                    file_name=file_name,
                    decompress=True,
                )

                # preprocess and compute features
                for user_id, activity_graph in graph_dict.items():
                    graph = get_largest_component(activity_graph.G)
                    feat_dict = {
                        "user_id": str(user_id),
                        "duration": int(weeks),
                        "file_name": str(file_name),
                        "journey_length": journey_length(graph),
                        "transition_beta": transition_beta(graph),
                        "degree_beta": degree_beta(graph),
                        "hub_size": hub_size(graph),
                        "median_trip_distance": median_trip_distance(graph),
                        "highest_decile_distance": highest_decile_distance(graph),
                    }
                    print(feat_dict)
                    # compute features
                    table_precomputed_feats.append(feat_dict)

            # write to db
            df_feats = pd.DataFrame(table_precomputed_feats)
            df_feats.to_sql("graph_features", engine, study, if_exists="replace")
            print("written to db")


def relate_to_features():
    all_features_studies = []
    # con = get_engine(return_con=True, DBLOGIN_FILE="dblogin.json")
    engine = get_engine(DBLOGIN_FILE="dblogin.json")
    for study in ["gc1", "gc2"]:

        # get graph features
        graph_feats = pd.read_sql(f"SELECT * FROM {study}.graph_features", engine)
        print("loaded graph feats", graph_feats["user_id"].nunique(), len(graph_feats))

        # get classical features
        classical_feats = pd.read_sql(f"SELECT * FROM {study}.classical_features", engine)
        print("loaded raw feats", classical_feats["user_id"].nunique(), len(classical_feats))
        classical_feats["user_id"] = classical_feats["user_id"].astype(str)

        # get user info
        user_info = pd.read_sql_query(sql=f"SELECT * FROM {study}.user_info_clean".format(study), con=engine)
        user_info = user_info[~pd.isna(user_info["user_id"])]
        user_info["user_id"] = user_info["user_id"].astype(str)
        print("Loaded user info", len(user_info))

        # get user matching difficulty (i.e., the rank)
        rank_query = (
            f"SELECT u_user_id, u_filename, rank FROM {study}.user_ranking WHERE p_duration=28 AND u_duration=28"
        )
        ranks = pd.read_sql(rank_query, con=engine).rename(columns={"u_user_id": "user_id"})
        print("Loaded ranks", ranks["user_id"].nunique(), len(ranks))

        # merge rank with user info and graph feats
        together = ranks.merge(user_info, how="left", left_on="user_id", right_on="user_id")
        together = together.merge(classical_feats, how="left", left_on="user_id", right_on="user_id")
        together = together.merge(
            graph_feats, how="left", left_on=["user_id", "u_filename"], right_on=["user_id", "file_name"]
        )
        print("final", together.columns, len(together))

        # correlate rank with user survey features and graph feats
        # together.to_csv(f"feature_analysis_{study}.csv", index=False)
        all_features_studies.append(together)
    all_features_studies = pd.concat(all_features_studies)

    # Pass the combined df to the analyze function
    analyze_features(all_features_studies)


def rank_users():
    engine = get_engine(DBLOGIN_FILE="dblogin.json")
    for study in ["gc1", "gc2"]:
        distances_query = f"SELECT * FROM {study}.distance"  # WHERE p_duration=28 and u_duration=28"
        feature_cross_product_df = pd.read_sql(distances_query, con=engine)
        feature_cross_product_df["same_user"] = (
            feature_cross_product_df["p_user_id"] == feature_cross_product_df["u_user_id"]
        )
        metric = "mse"
        feature_cross_product_df[f"{metric}_combined"] = (
            feature_cross_product_df[f"{metric}_in_degree"]
            + feature_cross_product_df[f"{metric}_out_degree"]
            + feature_cross_product_df[f"{metric}_shortest_path"]
            + feature_cross_product_df[f"{metric}_transition"]
        )
        feature_cross_product_df = clean_impossible_matches(feature_cross_product_df)
        feature_cross_product_df["rank"] = feature_cross_product_df.groupby(
            by=["u_user_id", "u_duration", "u_filename", "p_duration", "p_filename"]
        )["mse_combined"].rank()
        df_rank_filtered = feature_cross_product_df[feature_cross_product_df["same_user"]]
        df_rank_filtered = df_rank_filtered[
            ["p_user_id", "p_duration", "p_filename", "u_user_id", "u_duration", "u_filename", "rank"]
        ]
        df_rank_filtered.to_sql("user_ranking", engine, study, if_exists="replace")
        print("Ranks written to DB")


def analyze_features(res):
    # # Load from csv
    # res = []
    # for study in ["gc1", "gc2"]:
    #     res.append(pd.read_csv(f"feature_analysis_{study}.csv"))
    # res = pd.concat(res)

    test_results = []
    for col in res.columns:
        # 1) Remove NaNs
        rows_nan = pd.isna(res[col])
        targets = res.loc[~rows_nan, "rank"]
        test_var = res.loc[~rows_nan, col]
        uni_val = col

        # 2) skip some columns
        if col in ["user_id", "u_filename", "rank", "index", "duration", "file_name"]:
            continue
        # 3) case disti/nction by datatype
        if (test_var.dtype == float or test_var.dtype == int) and test_var.nunique() > 2:
            r, p = pearsonr(test_var, targets)
            test_results.append(
                {
                    "name": col,
                    "nr_samples": len(test_var),
                    "p-value": p,
                    "mean_rank_low": targets[test_var <= test_var.median()].mean(),
                    "mean_rank_high": targets[test_var > test_var.median()].mean(),
                    "stat": r,
                    "test": "pearsonr",
                }
            )
        else:
            for uni_val in test_var.unique():
                nr_samp_val = sum(test_var == uni_val)
                r, p = mannwhitneyu(targets[test_var == uni_val], targets[test_var != uni_val])
                test_results.append(
                    {
                        "name": col,
                        "value": uni_val,
                        "nr_samples": nr_samp_val,
                        "p-value": p,
                        "stat": r,
                        "mean_rank_low": targets[test_var != uni_val].mean(),
                        "mean_rank_high": targets[test_var == uni_val].mean(),
                        "test": "mannwhitneyu",
                    }
                )
    pd.DataFrame(test_results).sort_values("p-value").to_csv(os.path.join("outputs", "test_results_features.csv"))


if __name__ == "__main__":
    # # compute graph features for each user and write to a new table
    # compute_graph_features()

    # # Compute the ranks of each user
    # rank_users()

    # # merge with user survey features
    relate_to_features()
