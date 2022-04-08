import pandas as pd
import numpy as np
import sqlalchemy
from graph_trackintel.io import read_graphs_from_postgresql
from utils import get_engine
from pandas import DataFrame
import networkx as nx
from sklearn.preprocessing import normalize
import os
import psycopg2
import pickle
import warnings
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error


def calculate_reciprocal_rank(df, k=10, distance_column="distance"):
    # TODO: computing top-k-accuracy with this function vs the other function - results don't match

    df["rank"] = df.groupby(by=["p_duration", "p_user_id", "u_duration", "p_filename", "u_filename"])[
        distance_column
    ].rank()
    df["reciprocal_rank"] = 1 / df["rank"]
    df_rank_filtered = df[df["same_user"]]

    # for top 10
    # df_rank_filtered["topk"] = (df_rank_filtered["rank"] <= k).astype(int)
    # matrix_elements = df_rank_filtered.groupby(by=["p_duration", "u_duration"])["topk"].agg(["mean", "std"])

    matrix_elements = df_rank_filtered.groupby(by=["p_duration", "u_duration"])["reciprocal_rank"].agg(["mean", "std"])

    mean_matrix = matrix_elements["mean"].unstack(level=-1, fill_value=None) * 100
    std_matrix = matrix_elements["std"].unstack(level=-1, fill_value=None) * 100
    return mean_matrix, std_matrix


def calculate_topk_accuracy(df, k, distance_column="distance"):
    # get best guesses per group ranks
    # get minimum row index per group for each block of [p_duration, p_user, p_start_date(=p_filename) u_duration]
    # https://stackoverflow.com/a/71130117/16232216

    min_row_ix_by_group = (
        df.groupby(by=["p_duration", "p_user_id", "u_duration", "p_filename"])[distance_column]
        .nsmallest(k)
        .index.get_level_values(-1)
    )
    # top k guesses
    min_by_group = df.loc[min_row_ix_by_group]

    # Group by group and calculate if a single group had a correct guess (yes or no). Column same user is boolean
    correct_guess_by_group = (
        min_by_group.groupby(by=["p_duration", "p_user_id", "u_duration", "p_filename"])["same_user"]
        .sum()
        .reset_index()
    )

    # Groups are currently: ['p_duration', 'p_user_id', 'u_duration', 'p_file_name']
    # Groups for matrix are: ['p_duration', 'u_duration']
    # we now aggregate to the final matrix and calculate mean + standard deviation

    # aggregate to matrix elements
    matrix_elements = correct_guess_by_group.groupby(by=["p_duration", "u_duration"])["same_user"].agg(["mean", "std"])

    mean_matrix = matrix_elements["mean"].unstack(level=-1, fill_value=None) * 100
    std_matrix = matrix_elements["std"].unstack(level=-1, fill_value=None) * 100

    return mean_matrix, std_matrix


if __name__ == "__main__":

    STUDY = "gc1"
    DIST_COL = "kldiv_in_degree"

    engine = get_engine(DBLOGIN_FILE=os.path.join("dblogin.json"))
    print("download distances")
    distances_query = f"SELECT * FROM {STUDY}.distance"
    feature_cross_product_df = pd.read_sql(distances_query, con=engine)

    # calculate same_user_flag (important for topk acc)
    feature_cross_product_df["same_user"] = (
        feature_cross_product_df["p_user_id"] == feature_cross_product_df["u_user_id"]
    )

    mean_matrix, std_matrix = calculate_topk_accuracy(
        feature_cross_product_df, k=10, distance_column=DIST_COL
    )  # uses the column distance

    print("\nMEAN")
    print(mean_matrix)
    # print("\nSTD")
    # print(std_matrix)

    print("RECIPROCAL")
    mean_matrix, std_matrix = calculate_reciprocal_rank(
        feature_cross_product_df, k=10, distance_column=DIST_COL
    )  # uses the column distance

    print("\nMEAN")
    print(mean_matrix)
    # print("\nSTD")
    # print(std_matrix)
