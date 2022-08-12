import numpy as np
import pandas as pd
import time
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

from utils import get_engine


def kl_div(dist1, dist2):
    """
    KL divergence between two distributions
    dist1, dists2: 2D numpy arrays of shape (nr_samples, distribution_size)
    """
    element_wise_entropy = rel_entr(dist1, dist2)
    # need to mask out the infs
    element_wise_entropy[element_wise_entropy == np.inf] = 0
    return np.sum(element_wise_entropy, axis=1)


def mse(dist1, dist2):
    """
    MSE between two arrays
    dist1, dists2: 2D numpy arrays of shape (nr_samples, distribution_size)
    """
    return np.sum((dist1 - dist2) ** 2, axis=1)


def wasserstein(dist1, dist2):
    """
    Wasserstein distance between two distributions
    dist1, dists2: 2D numpy arrays of shape (nr_samples, distribution_size)
    """
    div = []
    for i in range(len(dist1)):
        # compute distance row-wise
        div.append(wasserstein_distance(dist1[i], dist2[i]))
    return div


def get_div_array(dist1, dist2, metric="kldiv"):
    """
    Get inverse similarity (divergence) between two distributions
    Arguments:
        dist1, dist2: 2D numpy arrays of shape (nr_samples, distribution_size)
        metric: {kldiv, mse, wasserstein}
            Chosen metric for comparison
    Returns:
        1D numpy array with row-wise similarity    
    """
    # define functions from above for each method
    metric_dict = {"kldiv": kl_div, "mse": mse, "wasserstein": wasserstein}
    metric_fun = metric_dict[metric]
    # get row-wise divergence for an array
    div = metric_fun(dist1, dist2)
    return div


def get_similarity(df, columns_for_similarity, metric="kldiv"):
    """Compute divergence of user from pool"""

    def get_array_from_df(cols):
        # helper function to transform database lists to array
        vals_col = df[cols].values.tolist()
        vals_col = np.array(vals_col).reshape((len(vals_col), -1))
        return vals_col

    for col in columns_for_similarity:
        # transform df input cols to numpy arrays
        feats_pool = get_array_from_df("p_" + col)
        feats_user = get_array_from_df("u_" + col)
        print(feats_user.shape, feats_pool.shape)

        # compute divergence
        for metric in ["kldiv", "mse", "wasserstein"]:
            out_col_name = f"{metric}_{col[:-6]}"
            df[out_col_name] = get_div_array(feats_pool, feats_user, metric=metric)

    # clean table
    cols_to_drop = [col for col in df.columns if "feats" in col]
    df_out = df.drop(columns=cols_to_drop)
    return df_out


if __name__ == "__main__":

    study = "gc1"

    # retrieve data
    engine = get_engine(DBLOGIN_FILE="dblogin.json")

    # load cross join table
    tic = time.time()
    df = pd.read_sql(f"SELECT * FROM {study}.dur_features_cross_join", engine)
    print("loaded data...")

    # Compute similarities for each combination in cross join table
    df_out = get_similarity(
        df, ["in_degree_feats", "shortest_path_feats", "transition_feats", "out_degree_feats", "centrality_feats"]
    )
    print("similarity computed...")

    # write to database
    df_out.to_sql("distance", engine, study, if_exists="replace")
    print("written to db, time for processing", time.time() - tic)
