import os
from sqlalchemy import create_engine
import psycopg2
import json


def get_engine(return_con=False, DBLOGIN_FILE=os.path.join("..", "dblogin.json")):
    """Crete a engine object for database connection

    study: Used to specify the database for the connection. "yumuv_graph_rep" directs to sbb internal database
    return_con: Boolean
        if True, a psycopg connection object is returned
    """
    # build database login string from file
    # DBLOGIN_FILE = os.path.join("..", "dblogin.json")
    with open(DBLOGIN_FILE) as json_file:
        LOGIN_DATA = json.load(json_file)

    conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**LOGIN_DATA)
    engine = create_engine(conn_string)
    if return_con:
        con = psycopg2.connect(
            dbname=LOGIN_DATA["database"],
            user=LOGIN_DATA["user"],
            password=LOGIN_DATA["password"],
            host=LOGIN_DATA["host"],
            port=LOGIN_DATA["port"],
        )

    if return_con:
        return con
    else:
        return engine
