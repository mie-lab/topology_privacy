# create a dummy feature dataset that has an entry for every user, graph duration and start time combination

import pandas as pd

import numpy as np
import os
from utils import get_engine

engine = get_engine(DBLOGIN_FILE=os.path.join("dblogin.json"))
con = get_engine(return_con=True, DBLOGIN_FILE=os.path.join("dblogin.json"))


# all combinations of
# user_id, duration, file_name, random feature

# get all timebins with all start dates
time_bins = [4, 8, 12, 16, 20, 24, 28]
df_list = []
for time_bin in time_bins:
    query = "select name from gc1.dur_{}w".format(time_bin)
    df = pd.read_sql(query, con=engine)
    df["duration"] = time_bin
    df_list.append(df)

df_dates = pd.concat(df_list, axis=0)

# get all users
query = "select user_id from gc1.included_users"
df_users = pd.read_sql(query, con=engine)

# merge with cross product
df_all = df_dates.merge(df_users, how="cross")

# add features
df_all["random_value"] = np.random.randint(1, 1000, df_all.shape[0])

df_all.rename({"name": "file_name"}, axis=1, inplace=True)
df_all.to_sql(name="dur_features_test", con=engine, schema="gc1", if_exists="replace")
