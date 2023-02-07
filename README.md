# Influence of tracking duration on the privacy of individual mobility graphs

This is the open-source code for our paper presented at LBS 2022, the 17th International Conference on Location Based Services in Munich. 
Our paper analyzes how privacy-preserving are graph representations of mobility. 

This repository includes all source code used to produce the results from the paper; however, the dataset can not be published due to data privacy. 

### Installation

The following packages are required to run this code:
- [trackintel](https://pypi.org/project/trackintel/)
- [graph_trackintel](https://github.com/mie-lab/graph-trackintel)
- pandas, numpy, scipy
- matplotlib, seaborn
- networkx
- psycopg2, sqlalchemy

## Analysis

Our analysis comprises the following steps:
1) [**precompute features**](precompute_features.py)
  A script that precomputes the graph features used in the publication.
    - transition_feats: The distribution of transition weights over the 20 most popular trips.
    - shortest_path_feats: The distribution of shortest-path lengths in the graph.
    - centrality_feats: The betweenness centrality of a node denotes its centrality with respect to other 
      nodes.
    - in_degree_feats: Distribution of (unweighted) node in-degrees.
    - out_degree_feats: Similar to the in-degree, the distribution of out-degrees over the 20 locations
with the highest out-degree is computed.    
2)  [**create cross join table**](create_cross_join_table.py)
    - Loads feature table
    - Combines all pairs of subsequent time periods for reidentification tests
3) [**compute similarity**](similarity.py)
    - Loads the cross-joined pairs
    - Computes similarity with several metrics for each pair
    - Writes the result to the database
4) [**rank users**](rank_users.py)
    - Loads the similarities
    - For each duration-bin combination, rank the users from the pool by their distance to the current user
    - Write the rank of the matched user to the database
5) [**fill_matrix**](fill_matrix.py):
    - Loads similarities for all combinations (all users & time period bins)
    - Computes reidentification accuracy for all time-bin combinations (over users)
    - Computes reciprocal ranks
6) [**visualization**](visualization.py): Functions for visualizing / summarizing all results reported in the paper, namely
    - The reidentification accuracies (Figure 2)
        - How much is the reidentification top-k accuracy for differnt tracking periods
    - Regression analysis (Table 1)
        - performs the regression analysis to evaluate the effect of pool- and test-user tracking duration on the
    matching performance
    - Feature analysis (Table 2)
        - which features improve user reidentification performance?
    - Privacy loss analysis (Figure 3)
        - What is the privacy loss due to reidentifying users?
    - Intra and inter user differences (Figure 4)
        - Is the variance explained by differences between users or by differences between time periods?



