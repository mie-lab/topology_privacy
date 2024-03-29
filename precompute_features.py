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
    """Normalize to get a distribution and convert to list"""
    return (vals / np.sum(vals)).tolist()


def degree_dist(graph, mode="in", cutoff=20):
    """
    Arguments:
        graph: ActivityGraph
        mode: {in, out, all} 
            which degrees to use: in (=indegree), out (outdegree) or all (both)
        cutoff: int
            Maximum considered degree

    Returns: 
        list with the normalized distribution of highest degrees
    """
    degree_vals = get_degrees(graph, mode=mode)
    all_vals = sorted(degree_vals)
    # put into a fixed-size array
    topk_vals = np.zeros(cutoff)  # for padding
    topk_vals[-len(all_vals) :] = all_vals[-cutoff:]
    # noralize distribution
    return normed_list(topk_vals)


def centrality_dist(graph, centrality_fun=betweenness_centrality):
    """
    Compute distribution of centrality in fixed size histogram vector

    Arguments:
        graph: ActivityGraph
        centrality_fun: function dependent on which centrality to use
    Returns:
        1D np array of length 10, distribution of centralities in graph
    """
    centrality = centrality_fun(graph)
    centrality_vals = list(centrality.values())
    # construct log distribution from 0 to 1
    log_space = np.logspace(0, 1.05, 10) / 10 - 0.1
    # distribution of centrality values
    centrality_dist, _ = np.histogram(centrality_vals, log_space)
    return normed_list(centrality_dist)


def shortest_path_distribution(graph, max_len=10):
    """
    Distribution all pair shortest path lengths
    Arguments:
        graph:ActivityGraph
        max_len: Int
            maximum considered sp length
    Returns: 
        list of normalized distritbuion of sp lengths
    """
    sp_counts = sp_length(graph, max_len=max_len)
    return normed_list(sp_counts)


def transition_distribution(graph, cutoff=20):
    """
    Distribution of edge transition weights
    Arguments:
        graph:ActivityGraph
        cutoff: Int
            maximum considered transition count
    Returns: 
        list of normalized distritbuion of highest transition counts
    """
    # get all transition counts
    transitions = np.array([edge[2]["weight"] for edge in graph.edges(data=True)])
    all_vals = sorted(transitions)
    # transfer to fixed size array
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

    for study in ["gc1", "gc2"]:
        # initialize output table
        table_precomputed_feats = []
        # Run for each time period
        for weeks in [1, 2]  + [4 * (i + 1) for i in range(7)]:
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
                    if graph.number_of_nodes() < 2:
                        print("Not enought nodes")
                        continue
                    feat_dict = {
                        "study": study,
                        "user_id": str(user_id),
                        "duration": int(weeks),
                        "file_name": str(file_name),
                        "transition_feats": transition_distribution(graph),
                        "shortest_path_feats": shortest_path_distribution(graph),
                        "centrality_feats": centrality_dist(graph),
                        "in_degree_feats": degree_dist(graph),
                        "out_degree_feats": degree_dist(graph, mode="out"),
                    }
                    # compute features
                    table_precomputed_feats.append(feat_dict)

        # write to db
        df_feats = pd.DataFrame(table_precomputed_feats)
        df_feats.to_sql("dur_features_1w", engine, study, if_exists="append", dtype=dtype_dict)
        print("written to db")


if __name__ == "__main__":
    precompute_features()
