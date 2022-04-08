



import pandas as pd

import numpy as np
import sqlalchemy
from graph_trackintel.io import read_graphs_from_postgresql
from utils import get_engine
from pandas import DataFrame

import sparse
import scipy
import networkx as nx
from sklearn.preprocessing import normalize
import future_trackintel
from future_trackintel import activity_graph
from sparse._compressed.compressed import uncompress_dimension
import itertools
import matplotlib.pyplot as plt
import sqlalchemy
from collections import Counter
import os
import psycopg2
import pickle
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error

#SQL query
    # this query performs a join on the feature rows. For every row it joins all possible combinations of durations
    # under the following conditions: Musst be from the same user and it musst be after the end of the current time bin.

    # To extent this query to join all possible time bin combinations add the t2.filename into the DISTINCT
    # statement. Furthermore you have to make sure that both intervals do not overlap (e.g., by extending the
    #  t1.enddate <= t2.filename::timestamp check.

    # query can be optimized by the following actions:
        # precompute enddate
        # cast types
        # add indices (critical speed improvement)

# alternative with statement (replaces create temp table)

    # WITH table_with_end_date AS
    #     (SELECT *, file_name::timestamp +  (duration::text || ' week')::interval AS enddate
    #     FROM gc1.dur_features)

feature_cross_product_query = """
CREATE TEMP TABLE IF NOT EXISTS table_with_end_date AS
	(SELECT *, file_name::timestamp +  (duration::text || ' week')::interval AS enddate
    FROM gc1.dur_features);
	
CREATE INDEX IF NOT EXISTS temp_table_end_date_index on table_with_end_date (enddate);
CREATE INDEX IF NOT EXISTS temp_table_user_id on table_with_end_date (user_id);
CREATE INDEX IF NOT EXISTS temp_table_duration on table_with_end_date (duration);
CREATE INDEX IF NOT EXISTS temp_table_file_name on table_with_end_date (file_name);

SELECT
    DISTINCT
    ON(t1.user_id, t1.duration, t1.file_name, t2.user_id, t2.duration)
    t1. *, t2. *
    FROM table_with_end_date t1
    join
        gc1.dur_features t2
        ON
        t1.enddate <= t2.file_name::timestamp
    ORDER
    BY t1.user_id, t1.duration, t1.file_name, t2.user_id, t2.duration, t2.file_name::timestamp"""


if __name__ == '__main__':

    engine = get_engine(DBLOGIN_FILE=os.path.join("dblogin.json"))
    print("get matrix")
    feature_cross_product_df = pd.read_sql(feature_cross_product_query, con=engine)

    # fix duplicate column names
    col_names = feature_cross_product_df.columns.to_list()
    start_right = int((len(col_names) -1) /2 + 1) # pool has additional column 'enddate'

    prefixed_names_pool = ['p_' + x for x in col_names[0:start_right]]
    prefixed_names_user = ['u_' + x for x in col_names[start_right:]]
    feature_cross_product_df.columns = prefixed_names_pool + prefixed_names_user

    # calculate same_user_flag
    feature_cross_product_df['same_user'] = feature_cross_product_df['p_user_id'] == feature_cross_product_df[
        'u_user_id']

    # solve unstructured data problems (lists are stored as stings
    print("data cleaning")
    feature_cross_product_df['p_in_degree_feats'] = feature_cross_product_df['p_in_degree_feats'].apply(
        lambda x: eval(x.replace('{', '[').replace('}', ']')))

    feature_cross_product_df['u_in_degree_feats'] = feature_cross_product_df['u_in_degree_feats'].apply(
        lambda x: eval(x.replace('{', '[').replace('}', ']')))

    # make sure that everything has the same length
    zero_len_flag = ((feature_cross_product_df['u_in_degree_feats'].apply(len) < 10) |
                     (feature_cross_product_df['p_in_degree_feats'].apply(len) < 10))
    feature_cross_product_df = feature_cross_product_df.loc[~zero_len_flag]

    # calculate a specific distance
    def distance_wrapper(df, distance_type):
        if distance_type == 'in_degree':
            return [mean_squared_error(*a, squared=False) for a in tuple(
                zip(df["p_in_degree_feats"], df["u_in_degree_feats"]))]


    # the distance wrapper function should return a distance vector given the dataframe.
    print("calc distance")
    feature_cross_product_df['distance'] = distance_wrapper('in_degree')

    print("get k best guesses per group")

    def calculate_topk_accuracy(df, k):
        # get best guesses per group ranks
            # get minimum row index per group for each block of [p_duration, p_user, p_start_date(=p_filename) u_duration]
            # https://stackoverflow.com/a/71130117/16232216

        min_row_ix_by_group = df.groupby(by=['p_duration', 'p_user_id', 'u_duration', 'p_file_name']
                                         ).distance.nsmallest(k).index.get_level_values(-1)
        # top k guesses
        min_by_group = df.loc[min_row_ix_by_group]

        # Group by group and calculate if a single group had a correct guess (yes or no). Column same user is boolean
        correct_guess_by_group = min_by_group.groupby(by=['p_duration', 'p_user_id', 'u_duration', 'p_file_name'])[
            'same_user'].sum().reset_index()

        # Groups are currently: ['p_duration', 'p_user_id', 'u_duration', 'p_file_name']
        # Groups for matrix are: ['p_duration', 'u_duration']
        # we now aggregate to the final matrix and calculate mean + standard deviation

        # aggregate to matrix elements
        matrix_elements = correct_guess_by_group.groupby(by=['p_duration', 'u_duration'])['same_user'].agg(['mean',
                                                                                                                'std'])

        mean_matrix = matrix_elements['mean'].unstack(level=- 1, fill_value=None) *100
        std_matrix = matrix_elements['std'].unstack(level=- 1, fill_value=None) * 100

        return mean_matrix, std_matrix


    mean_matrix, std_matrix = calculate_topk_accuracy(feature_cross_product_df, k=10)




