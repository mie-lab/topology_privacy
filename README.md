# topology_privacy
How privacy-preserving are graph representations of mobiltiy


- precompute features
  A script that precomputes the graph features used in the publication.
    - transition_feats: The distribution of transition weights over the 20 most popular trips.
    - shortest_path_feats: The distribution of shortest-path lengths in the graph.
    - centrality_feats: The betweenness centrality of a node denotes its centrality with respect to other 
      nodes.
    - in_degree_feats: Distribution of (unweighted) node in-degrees.
    - out_degree_feats: Similar to the in-degree, the distribution of out-degrees over the 20 locations
with the highest out-degree is computed.
      
- create cross join table
- similarity.py
- fill_matrix.py

Further analysis:
    - regression_analysis.py
    This script performs the regression analysis to evaluate the effect of pool- and test-user tracking duration on the
    matching performance (table 1).