# Influence of tracking duration on the privacy of individual mobility graphs

This is the open-source code for our paper presented at LBS 2022, the 17th International Conference on Location Based Services in Munich. 
Our paper analyzes how privacy-preserving are graph representations of mobility. 

This repository includes all source code used to produce the results from the paper; however, the dataset can not be published due to data privacy. 

Our analysis comprises the following steps:
- [**precompute features**](precompute_features.py)
  A script that precomputes the graph features used in the publication.
    - transition_feats: The distribution of transition weights over the 20 most popular trips.
    - shortest_path_feats: The distribution of shortest-path lengths in the graph.
    - centrality_feats: The betweenness centrality of a node denotes its centrality with respect to other 
      nodes.
    - in_degree_feats: Distribution of (unweighted) node in-degrees.
    - out_degree_feats: Similar to the in-degree, the distribution of out-degrees over the 20 locations
with the highest out-degree is computed.    
- [**create cross join table**](create_cross_join_table.py)
- [**compute similarity**](similarity.py)
- [**fill_matrix**](fill_matrix.py)
- [**visualization**](visualization.py): Functions for visualizing / summarizing all results reported in the paper, namely
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

Further analysis:
    - regression_analysis.py
    This script 


