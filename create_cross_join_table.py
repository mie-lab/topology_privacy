import sqlalchemy
from utils import get_engine
import os

# SQL query
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


if __name__ == "__main__":
    datasets = ["gc2", "gc1"]

    # Threshold
    # query to determine threshold:
    # "select p_duration, p_filename, u_duration, u_filename, count(*) as count_el from gc2.dur_features_cross_join
    # group by p_duration, p_filename, u_duration, u_filename order by p_duration, u_duration, p_filename;"

    dset_thresholds = {"gc1": 5000, "gc2": 1000}
    engine = get_engine(DBLOGIN_FILE=os.path.join("dblogin.json"))
    con = engine.connect()

    for ds in datasets:
        print(ds)
        sql_query = f"""
        DROP TABLE if EXISTS {ds}.dur_features_cross_join_1w;
        
        create table {ds}.dur_features_cross_join_1w as
        
        WITH table_with_end_date AS
            (SELECT *, file_name::timestamp +  (duration::text || ' week')::interval AS enddate
            FROM {ds}.dur_features_1w)
        
        SELECT
            DISTINCT
            ON(t1.user_id, t1.duration, t1.file_name::timestamp, t2.user_id, t2.duration, t2.file_name::timestamp)
            t1.user_id as p_user_id, t1.duration as p_duration, t1.file_name as p_filename, 
            t1.shortest_path_feats as p_shortest_path_feats, t1.centrality_feats as p_centrality_feats, 
            t1.in_degree_feats as p_in_degree_feats, t1.out_degree_feats as p_out_degree_feats,
            t1.transition_feats as p_transition_feats,
            t1.enddate as enddate,
            t2.user_id as u_user_id, t2.duration as u_duration, t2.file_name as u_filename, 
            t2.shortest_path_feats as u_shortest_path_feats, t2.centrality_feats as u_centrality_feats, 
            t2.in_degree_feats as u_in_degree_feats, t2.out_degree_feats as u_out_degree_feats,
            t2.transition_feats as u_transition_feats
            FROM table_with_end_date t1
            join
                {ds}.dur_features_1w t2
                ON
                t1.enddate <= t2.file_name::timestamp
            ORDER
            BY t1.user_id, t1.duration, t1.file_name::timestamp, t2.user_id, t2.duration, t2.file_name::timestamp;  
        """

        con.execute(sql_query)
        con.execute(f"ALTER TABLE {ds}.dur_features_cross_join_1w ADD COLUMN postgres_id SERIAL PRIMARY KEY;")

        # Delete (p_duration, p_filename, u_duration, u_filename) with very few observations.
        # The SQL query always choses the next possible non-overlapping time bin (=u_filename) for a
        # p_duration, p_filename, u_duration combination. Because of the way the graphs are created and filtered by
        # trackign quality (min days tracked) some combinations of (u_filename, u_duration) are missing for some users.
        # In such a case
        # There are p_duration, p_filename, u_duration, u_filename combinations with very few elements.

        sql_query_postprocessing = f"""
        WITH 
            ids_by_count as (
                SELECT array_agg(postgres_id) AS postgres_ids, p_duration, p_filename, u_duration, u_filename, 
                    count(*) AS count_el 
                FROM {ds}.dur_features_cross_join_1w 
                GROUP BY p_duration, p_filename, u_duration, 
                    u_filename HAVING count(*) < {dset_thresholds[ds]} 
                ORDER BY p_duration, u_duration, p_filename),
            invalid_id_list as (
                SELECT array_agg(postgres_ids_unnest) AS id_array 
                FROM ids_by_count, unnest(postgres_ids) AS postgres_ids_unnest)
    
        DELETE FROM {ds}.dur_features_cross_join_1w
        WHERE postgres_id = ANY(
            SELECT unnest(id_array) FROM invalid_id_list);
        """
        con.execute(sql_query_postprocessing)

        # ensure that nina has access to tables.
        try:
            sql_access = f"""GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {ds} TO wnina;"""
            con.execute(sql_access)
        except:
            print("could not grant access to wnina")
