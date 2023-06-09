import pandas as pd
from sqlalchemy import create_engine
from abc import ABC


class Database():

    def __init__(self, db=None, username="info_team", pwd="Make_recipes0430"):
        self.username = username
        self.pwd = pwd
        self.engine = None
        self.name = None
        self.table = None
        if db is not None:
            self.connect(db)

    def connect(self, db: str):
        conn_string = f'mysql+mysqlconnector://{self.username}:{self.pwd}@192.168.1.91:3306/{db}'
        self.engine = create_engine(conn_string)
        self.name = db

    def disconnect(self):
        del self.name
        self.name = None

    def pull(self, table="", remove_index=True):
        assert self.engine is not None
        engine = self.engine
        data = pd.read_sql_table(table, engine)
        if remove_index:
            data = self._remove_index(data)
        return data
    
    def push(self, data=None, table=""):
        ...

    def _remove_index(self, data):
        _data = data.copy()
        if "index" in _data.columns:
            _data = _data.drop('index', axis=1)
        elif "id" in data.columns:
            _data = _data.drop('id', axis=1)
        return _data


if __name__ == "__main__":
    db = Database(db="test_db")
    df = db.pull(table="half_cell_classifier_test")