import pandas as pd
import numpy as np
import sqlalchemy

from graph_trackintel.graph_utils import get_largest_component
from graph_trackintel.io import read_graphs_from_postgresql
from graph_trackintel.analysis.graph_features import (
    sp_length,
    betweenness_centrality,
    get_degrees,
)
from utils import get_engine


def normed_list(vals):
    return (vals / np.sum(vals)).tolist()


def degree_dist(graph, mode="in", cutoff=20):
    """
    mode: in (=indegree), out (outdegree) or all (both)
    """
    degree_vals = get_degrees(graph, mode=mode)
    all_vals = sorted(degree_vals)
    topk_vals = np.zeros(cutoff)  # for padding
    topk_vals[-len(all_vals) :] = all_vals[-cutoff:]
    return normed_list(topk_vals)


def centrality_dist(graph, centrality_fun=betweenness_centrality):
    """
    Compute distribution of centrality in fixed size histogram vector
    Returns:
        1D np array of length 10
    """
    centrality = centrality_fun(graph)
    centrality_vals = list(centrality.values())
    # construct log distribution from 0 to 1
    log_space = np.logspace(0, 1.05, 10) / 10 - 0.1
    # distribution of centrality values
    centrality_dist, _ = np.histogram(centrality_vals, log_space)
    return normed_list(centrality_dist)


def shortest_path_distribution(graph, max_len=10):
    sp_counts = sp_length(graph, max_len=max_len)
    return normed_list(sp_counts)


def transition_distribution(graph, cutoff=20):
    transitions = np.array([edge[2]["weight"] for edge in graph.edges(data=True)])
    all_vals = sorted(transitions)
    topk_vals = np.zeros(cutoff)  # for padding
    topk_vals[-len(all_vals) :] = all_vals[-cutoff:]
    return normed_list(topk_vals)


def precompute_features():
    print("----------------- GET TIME BINS FOR GC 1 and 2 --------------------")
    con = get_engine(return_con=True, DBLOGIN_FILE="dblogin.json")
    engine = get_engine(DBLOGIN_FILE="dblogin.json")

    dtype_dict = {
        "transition_feats": sqlalchemy.ARRAY(sqlalchemy.types.REAL),
        "centrality_feats": sqlalchemy.ARRAY(sqlalchemy.types.REAL),
        "in_degree_feats": sqlalchemy.ARRAY(sqlalchemy.types.REAL),
        "out_degree_feats": sqlalchemy.ARRAY(sqlalchemy.types.REAL),
        "shortest_path_feats": sqlalchemy.ARRAY(sqlalchemy.types.REAL),
    }

    for study in ["gc1"]:
        # initialize output table
        table_precomputed_feats = []
        # Make new directory for this duration data
        # Run
        for weeks in [4 * (i + 1) for i in range(7)]:
            print("processing weeks:", weeks, "STUDY", study)
            cur = con.cursor()

            # get the timebin names
            cur.execute(f"SELECT name FROM {study}.dur_{weeks}w")

            table_name = f"dur_{weeks}w"
            all_names = cur.fetchall()

            for name in all_names:
                file_name = name[0]
                graph_dict = read_graphs_from_postgresql(
                    graph_table_name=table_name,
                    psycopg_con=con,
                    graph_schema_name=study,
                    file_name=file_name,
                    decompress=True,
                )
                for user_id, activity_graph in graph_dict.items():
                    graph = get_largest_component(activity_graph.G)
                    feat_dict = {
                        "user_id": str(user_id),
                        "duration": int(weeks),
                        "file_name": str(file_name),
                        "transition_feats": transition_distribution(graph),
                        "shortest_path_feats": shortest_path_distribution(graph),
                        "centrality_feats": centrality_dist(graph),
                        "in_degree_feats": degree_dist(graph),
                        "out_degree_feats": degree_dist(graph, mode="out"),
                    }
                    table_precomputed_feats.append(feat_dict)

            # write to db
            df_feats = pd.DataFrame(table_precomputed_feats)
            df_feats.to_sql("dur_features", engine, study, if_exists="replace", dtype=dtype_dict)
            print("written to db")


if __name__ == "__main__":
    precompute_features()
