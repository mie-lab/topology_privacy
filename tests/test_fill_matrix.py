from fill_matrix import calculate_reciprocal_rank, calculate_topk_accuracy
import pandas as pd
import os
from utils import get_engine

engine = get_engine(DBLOGIN_FILE=os.path.join("dblogin.json"))
con = get_engine(return_con=True, DBLOGIN_FILE=os.path.join("dblogin.json"))
k = 1

# df = pd.read_csv(os.path.join("..", "test_data_matrix_aggregation.csv"), sep=";")
df = pd.read_sql(
    "select * from gc1.dur_features_cross_join where u_user_id::int <= 1597::int and p_user_id::int <= "
    "1597::int and p_duration in(20, 24) and u_duration in (20, 24)",
    con=engine,
)
columns_to_drop = [
    "p_shortest_path_feats",
    "p_centrality_feats",
    "p_in_degree_feats",
    "p_out_degree_feats",
    "u_shortest_path_feats",
    "u_centrality_feats",
    "u_in_degree_feats",
    "u_out_degree_feats",
    "enddate",
    "postgres_id",
]
df.drop(columns_to_drop, inplace=True, axis=1)

# recode filename columns

recode_filename_dict = {"2016-11-23": 0, "2017-04-12": 1, "2017-08-30": 2, "2017-05-10": 3}
df["p_filename"] = df["p_filename"].map(recode_filename_dict)
df["u_filename"] = df["u_filename"].map(recode_filename_dict)

# create testdata
df = df.sort_values(["p_duration", "u_duration", "u_filename", "p_filename", "u_user_id", "p_user_id"])
# df = df.sort_values(['p_filename', 'u_filename','p_duration', 'u_duration',  'u_user_id', 'p_user_id'])
df["distance"] = 0
df["same_user"] = df["p_user_id"] == df["u_user_id"]


def count_matching_tasks(df):
    return df.groupby(["p_duration", "u_duration", "p_filename", "u_filename", "u_user_id"]).size().shape[0]


def list_matching_tasks(df):
    return df.groupby(["p_duration", "u_duration", "p_filename", "u_filename", "u_user_id"]).size().index


def set_task_as_success(df, p_duration, u_duration, p_filename, u_filename, u_user_id):
    ix = df[
        (df["p_duration"] == p_duration)
        & (df["u_duration"] == u_duration)
        & (df["p_filename"] == p_filename)
        & (df["u_filename"] == u_filename)
        & (df["u_user_id"] == u_user_id)
        & (df["p_user_id"] != u_user_id)
    ].index
    df.loc[ix, "distance"] = range(1, len(ix) + 1)


def filter_by_tuple(df, p_duration, u_duration, p_filename, u_filename, u_user_id):
    ix = df[
        (df["p_duration"] == p_duration)
        & (df["u_duration"] == u_duration)
        & (df["p_filename"] == p_filename)
        & (df["u_filename"] == u_filename)
        & (df["u_user_id"] == u_user_id)
    ].index
    return df.loc[ix]


def set_task_as_failure(df, p_duration, u_duration, p_filename, u_filename, u_user_id):
    ix = df[
        (df["p_duration"] == p_duration)
        & (df["u_duration"] == u_duration)
        & (df["p_filename"] == p_filename)
        & (df["u_filename"] == u_filename)
        & (df["u_user_id"] == u_user_id)
        & (df["p_user_id"] != u_user_id)
    ].index
    df.loc[ix, "distance"] = range(0, len(ix))
    nb_users_in_task = len(ix)

    ix = df[
        (df["p_duration"] == p_duration)
        & (df["u_duration"] == u_duration)
        & (df["p_filename"] == p_filename)
        & (df["u_filename"] == u_filename)
        & (df["u_user_id"] == u_user_id)
        & (df["p_user_id"] == u_user_id)
    ].index
    df.loc[ix, "distance"] = nb_users_in_task + 10


def set_task_as_success_non_unique_ranks(df, p_duration, u_duration, p_filename, u_filename, u_user_id):
    ix = df[
        (df["p_duration"] == p_duration)
        & (df["u_duration"] == u_duration)
        & (df["p_filename"] == p_filename)
        & (df["u_filename"] == u_filename)
        & (df["u_user_id"] == u_user_id)
        & (df["p_user_id"] != u_user_id)
    ].index
    df.loc[ix, "distance"] = range(0, len(ix))


def set_task_as_failure_non_unique_ranks(df, p_duration, u_duration, p_filename, u_filename, u_user_id):
    ix = df[
        (df["p_duration"] == p_duration)
        & (df["u_duration"] == u_duration)
        & (df["p_filename"] == p_filename)
        & (df["u_filename"] == u_filename)
        & (df["u_user_id"] == u_user_id)
        & (df["p_user_id"] != u_user_id)
    ].index
    df.loc[ix, "distance"] = [0] + list(range(0, len(ix) - 1))
    nb_users_in_task = len(ix)

    ix = df[
        (df["p_duration"] == p_duration)
        & (df["u_duration"] == u_duration)
        & (df["p_filename"] == p_filename)
        & (df["u_filename"] == u_filename)
        & (df["u_user_id"] == u_user_id)
        & (df["p_user_id"] == u_user_id)
    ].index
    df.loc[ix, "distance"] = nb_users_in_task - 2


# top left elements (acc 2/3 with high std within users)
df_ = df[(df["p_duration"] == 20) & (df["u_duration"] == 20)].sort_values(["u_user_id", "p_filename", "u_filename"])

# sort index_of_matching_tasks by u_user_id to alternate matching success by u_user_id
index_of_matching_tasks = list_matching_tasks(df_).sortlevel(-1)
alternator_flag = True
for ix_tuple in list_matching_tasks(df_):
    if alternator_flag:
        set_task_as_success_non_unique_ranks(df, *ix_tuple)
        alternator_flag = False
    else:
        set_task_as_failure_non_unique_ranks(df, *ix_tuple)
        alternator_flag = True

# lower left elements (acc 1/2 with non unique ranks)
df_ = df[(df["p_duration"] == 24) & (df["u_duration"] == 20)].sort_values(["u_user_id", "p_filename", "u_filename"])
index_of_matching_tasks = list_matching_tasks(df_).sortlevel(-1)
alternator_flag = True
for ix_tuple in list_matching_tasks(df_):
    if alternator_flag:
        set_task_as_success(df, *ix_tuple)
        alternator_flag = False
    else:
        set_task_as_failure(df, *ix_tuple)
        alternator_flag = True


count_matching_tasks(df_)

df_ = filter_by_tuple(df, *(20, 4, "2016-11-23", "2017-04-12", "1737"))

ix_tuple = (20, 4, "2016-11-23", "2017-04-12", "1685")
p_duration, u_duration, p_filename, u_filename, u_user_id = ix_tuple
ix = df[
    (df["p_duration"] == p_duration)
    & (df["u_duration"] == u_duration)
    & (df["p_filename"] == p_filename)
    & (df["u_filename"] == u_filename)
    & (df["u_user_id"] == u_user_id)
].index
df_ = df.loc[ix]

m1, m2 = calculate_reciprocal_rank(df_, k=k, distance_column="distance")
print(m1)
n1, n2 = calculate_topk_accuracy(df_, k)
print(n1)
# acc = df_.groupby(by=[])


# df['']
#
#
# df['same_user'] = df['u_user_id'] == df['p_user_id']
#
# mean_matrix, std_matrix = calculate_topk_accuracy(df, k)
# print(mean_matrix)
# print(std_matrix)
# mean_matrix, std_matrix = calculate_reciprocal_rank(df, k=k, distance_column='distance')
#
# print(mean_matrix)
# print(std_matrix)
#
a = df.groupby(by=["p_duration", "u_duration"]).size()
