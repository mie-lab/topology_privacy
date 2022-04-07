import json
import os
import psycopg2
from graph_trackintel.io import read_graphs_from_postgresql


def get_con():
    DBLOGIN_FILE = os.path.join("dblogin.json")
    with open(DBLOGIN_FILE) as json_file:
        LOGIN_DATA = json.load(json_file)

    con = psycopg2.connect(
        dbname=LOGIN_DATA["database"],
        user=LOGIN_DATA["user"],
        password=LOGIN_DATA["password"],
        host=LOGIN_DATA["host"],
        port=LOGIN_DATA["port"],
    )
    return con


def get_timebins():
    print("----------------- GET TIME BINS FOR GC 1 and 2 --------------------")
    con = get_con()

    for study in ["gc1", "gc2"]:
        # Make new directory for this duration data
        # Run
        for weeks in [4 * (i + 1) for i in range(7)]:
            print("processing weeks:", weeks, "STUDY", study)
            cur = con.cursor()
            # get the timebin names
            cur.execute(f"SELECT name FROM {study}.dur_{weeks}w")
            all_names = cur.fetchall()

            for name in all_names:
                table_name = f"dur_{weeks}w"
                file_name = name[0]
                graph_dict = read_graphs_from_postgresql(
                    graph_table_name=table_name,
                    psycopg_con=con,
                    graph_schema_name=study,
                    file_name=file_name,
                    decompress=True,
                )
                print(graph_dict.keys())


get_timebins()
