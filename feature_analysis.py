import pandas as pd
import numpy as np
import os
import sqlalchemy
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, pearsonr

import statsmodels.api as sm

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
            df_feats.to_sql("graph_features", engine, study, if_exists="append")
            print("written to db")


def load_features_for_correlation():
    all_features_studies = []
    # con = get_engine(return_con=True, DBLOGIN_FILE="dblogin.json")
    engine = get_engine(DBLOGIN_FILE="dblogin.json")
    for study in ["gc1", "gc2"]:

        # get graph features TODO: for now only using 28-bin features
        graph_feats = pd.read_sql(f"SELECT * FROM {study}.graph_features WHERE duration=28", engine)
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
        rank_query = f"SELECT u_user_id, u_filename, rank FROM {study}.user_ranking_mse_combined WHERE p_duration=4 AND u_duration=4"
        ranks = pd.read_sql(rank_query, con=engine).rename(columns={"u_user_id": "user_id"})
        print("Loaded ranks", ranks["user_id"].nunique(), len(ranks))

        # merge with graph feats on correct bin
        together = ranks.merge(  # TODO: on "u_duration", "u_filename" and "duration", "file_name"
            graph_feats, how="left", left_on=["user_id"], right_on=["user_id"]
        )
        # group by user
        agg_dict = {col: "mean" for col in graph_feats.columns if col not in ["user_id", "file_name", "duration"]}
        agg_dict["rank"] = "mean"
        together = together.groupby("user_id").agg(agg_dict).reset_index()

        # merge rank with user info and graph feats
        together = together.merge(user_info, how="left", left_on="user_id", right_on="user_id")
        together = together.merge(classical_feats, how="left", left_on="user_id", right_on="user_id")
        print("final", together.columns, len(together))

        # correlate rank with user survey features and graph feats
        # together.to_csv(f"feature_analysis_{study}.csv", index=False)
        all_features_studies.append(together)
    all_features_studies = pd.concat(all_features_studies)

    # Pass the combined df to the analyze function
    analyze_features_correlation(all_features_studies)


def analyze_features_correlation(res):
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


def analyze_features_regression(out_path):
    # selet suitable subset of features because otherwise it requires a lot of explanation
    use_features = (
        ["radius_of_gyration", "random_entropy", "real_entropy", "median_trip_distance"]
        + ["journey_length", "hub_size", "degree_beta", "transition_beta"]  # common normal features
        + ["age", "sex", "HT", "GA"]  # graph features  # socio-demographics
    )
    out_by_dur = []
    for duration in [1, 2] + [4 * (i + 1) for i in range(7)]:
        all_features_studies = []
        # con = get_engine(return_con=True, DBLOGIN_FILE="dblogin.json")
        engine = get_engine(DBLOGIN_FILE="dblogin.json")
        # over studies
        both_studies = []
        for study in ["gc1", "gc2"]:

            # get graph features TODO: for now only using 28-bin features
            graph_feats = pd.read_sql(f"SELECT * FROM {study}.graph_features WHERE duration=28", engine)
            print("loaded graph feats", graph_feats["user_id"].nunique(), len(graph_feats))
            # group by user and average the graph features
            agg_dict = {col: "mean" for col in graph_feats.columns if col not in ["user_id", "file_name", "duration"]}
            graph_feats = graph_feats.groupby("user_id").agg(agg_dict).reset_index()

            # get classical features
            classical_feats = pd.read_sql(f"SELECT * FROM {study}.classical_features", engine)
            print("loaded classical feats", classical_feats["user_id"].nunique(), len(classical_feats))
            classical_feats["user_id"] = classical_feats["user_id"].astype(str)

            # get user info
            user_info = pd.read_sql_query(sql=f"SELECT * FROM {study}.user_info_clean".format(study), con=engine)
            user_info = user_info[~pd.isna(user_info["user_id"])]
            user_info["user_id"] = user_info["user_id"].astype(str)
            print("Loaded user info", len(user_info))

            # get user matching difficulty (i.e., the rank)
            rank_query = f"SELECT u_user_id, u_filename, rank FROM {study}.user_ranking_mse_combined WHERE p_duration={duration} AND u_duration={duration}"
            ranks = pd.read_sql(rank_query, con=engine).rename(columns={"u_user_id": "user_id"})
            print("Loaded ranks", ranks["user_id"].nunique(), len(ranks))
            nr_unique_users = ranks["user_id"].nunique()
            ranks["rank"] = ranks["rank"] / nr_unique_users * 100
            ranks = ranks.groupby("user_id").agg({"rank": "mean"})

            # merge with graph feats on correct bin
            together = ranks.merge(  # TODO: on "u_duration", "u_filename" and "duration", "file_name"
                graph_feats, how="left", left_on=["user_id"], right_on=["user_id"]
            )
            # merge rank with user info and graph feats
            together = together.merge(user_info, how="left", left_on="user_id", right_on="user_id")
            together = together.merge(classical_feats, how="left", left_on="user_id", right_on="user_id")
            #             print("final", together.columns, len(together))

            # restrict to the feature set and append
            both_studies.append(together[use_features + ["rank"]].reset_index())
        both_studies = pd.concat(both_studies)

        # preprocess
        both_studies["sex"] = both_studies["sex"].map({"Male": 0, "Female": 1})
        both_studies["PT"] = 0
        both_studies.loc[both_studies["HT"] == 1, "PT"] = 0.5
        both_studies.loc[both_studies["GA"] == 1, "PT"] = 1
        both_studies.drop(["GA", "HT", "index"], axis=1, inplace=True)
        both_studies["id"] = np.arange(len(both_studies))
        both_studies.set_index("id", inplace=True)
        #         print(both_studies)
        print("Before", len(both_studies), "after dropping nans", len(both_studies.dropna()))
        # convert to X, y
        notna_index = both_studies.dropna().index
        X = both_studies.drop("rank", axis=1).loc[notna_index]
        y = both_studies.loc[notna_index, "rank"]

        # check for collinearity
        corr = both_studies.dropna().corr()
        print("Highest absolute correlation value:", np.sort(np.abs(np.array(corr).flatten()))[-len(corr) - 1])
        # both_studies.to_csv(f"test_collinearity_{duration}.csv")

        X_normed = (X - X.mean()) / (X.std() + 1e-7)
        X_normed.shape
        X2 = sm.add_constant(X_normed)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        # postprocessing
        out = est2.summary2().tables[1][["Coef.", "P>|t|"]].round(3).swapaxes(1, 0)
        out.index.name = "M"
        out.reset_index(inplace=True)
        out["R squared"] = est2.rsquared
        out["duration"] = duration
        out_by_dur.append(out)
    out_by_dur = pd.concat(out_by_dur)

    coef_rounded = out_by_dur[out_by_dur["M"] == "Coef."].set_index(["duration", "M"]).round(2)
    coef_as_str = coef_rounded.values.astype(str)
    p_vals_significant = out_by_dur[out_by_dur["M"] != "Coef."].set_index(["duration", "M"]) < 0.05  # .values
    coef_as_str[p_vals_significant] = np.char.add(coef_as_str[p_vals_significant], " (*)")
    final_df_only_coef = (
        pd.DataFrame(coef_as_str, columns=coef_rounded.columns, index=coef_rounded.index)
        .reset_index()
        .drop("M", axis=1)
        .set_index("duration")
    )
    final_df_only_coef.to_csv(out_path[:-3] + "csv")
    with open(out_path, "w") as outfile:
        print(final_df_only_coef.to_latex(), file=outfile)


if __name__ == "__main__":
    # # compute graph features for each user and write to a new table
    # compute_graph_features()

    # # Compute the ranks of each user
    # rank_users()

    # # merge with user survey features
    analyze_features_regression(out_path="outputs/feature_regression.txt")
