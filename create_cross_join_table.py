
import sqlalchemy
from utils import get_engine
import os
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


if __name__ == '__main__':

    datasets = ['gc1', 'gc2']
    engine = get_engine(DBLOGIN_FILE=os.path.join("dblogin.json"))
    con = engine.connect()

    for ds in datasets:
        print(ds)
        sql_query = f"""
        DROP TABLE if EXISTS {ds}.dur_features_cross_join;
        
        create table {ds}.dur_features_cross_join as
        
        WITH table_with_end_date AS
            (SELECT *, file_name::timestamp +  (duration::text || ' week')::interval AS enddate
            FROM {ds}.dur_features)
        
        SELECT
            DISTINCT
            ON(t1.user_id, t1.duration, t1.file_name, t2.user_id, t2.duration)
            t1.user_id as p_user_id, t1.duration as p_duration, t1.file_name as p_filename, 
            t1.shortest_path_feats as p_shortest_path_feats, t1.centrality_feats as p_centrality_feats, 
            t1.in_degree_feats as p_in_degree_feats, t1.out_degree_feats as p_out_degree_feats,
            t1.enddate as enddate,
            t2.user_id as u_user_id, t2.duration as u_duration, t2.file_name as u_filename, 
            t2.shortest_path_feats as u_shortest_path_feats, t2.centrality_feats as u_centrality_feats, 
            t2.in_degree_feats as u_in_degree_feats, t2.out_degree_feats as u_out_degree_feats
            FROM table_with_end_date t1
            join
                {ds}.dur_features t2
                ON
                t1.enddate <= t2.file_name::timestamp
            ORDER
            BY t1.user_id, t1.duration, t1.file_name, t2.user_id, t2.duration, t2.file_name::timestamp;
        """

        con.execute(sql_query)

