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


class AutomatDataSet(ABC):
    """The base model of pandas dataframe tailored for Automat Solutions, Inc.
    """
    def __init__(self, df:pd.DataFrame=None):
        super().__init__()
        self._df = df.copy()
    
    @property
    def dataframe(self):
        return self._df


class LiquidMasterDataSet(AutomatDataSet):
    def __init__(self, df:pd.DataFrame):
        super().__init__(df=df)
        self.lce_name = "LCE"
        self.chemical_names = [
            'LiPF6',
            'LiTFSI', 'LiFSI', 'LiPO2F2', 'Lithium Perchlorate', 'LiBOB', 'LiBF4',
            'LiTf', 'L1', 'LiTDI', 'LiDFOB', 'EC', 'DEC', 'DMC', 'EMC',
            '1,3-dioxolane', '1,2-dimethoxyethane', 'PC', 'diglyme', 'TTE',
            'Sulfolane', 'TMP', 'BTC', 'DMDMOS', 'VC', 'FEC', 'SN', 'PS', 'DTO',
            'TMSPi', 'BTFE', 'AN', 'TFEO', 'BTFEC', 'TMSNCS', 'LiAlO2',
            'Montmorillonite', 'Aluminium Oxide', 'LiNO3', 'P2O5', 'Li2S', 'CsNO3',
            'S1', 'S2', 'S3', 'C1', 'C2', 'C3', 'E1', 'DTD', 'Li2O', 'TFMB',
        ]
        self._name_dict = {
            "1,2-dimethoxyethane": "DME",
            "1,3-dioxolane": "DOL",
            "Aluminium Oxide": "Al2O3",
            "Lithium Perchlorate": "LiClO4"
        }
    def map_names(self, name_dict:dict=None, inplace=True):
        """ Rename the main dataframe in the dataset according to `name_dict`.
        Parameters
            name_dict:  python dictionary. If `None` use the internal `_name_dict`.
        Returns
            dataframe: The renamed dataframe. 
        """
        df_copy = self._df.copy()
        if name_dict is None:
            name_dict = self._name_dict
        df_copy.rename(columns=name_dict, inplace=True)
        if inplace:
            # Update chemical names
            for old, new in self._name_dict.items():
                if old in self.chemical_names:
                    self.chemical_names.remove(old)
                    self.chemical_names.append(new)
            # Update dataframe
            self._df = df_copy
        else:
            return df_copy

    def normalize_components(self, by="total_mass(g)", inplace=False):
        df_copy = self._df.copy()
        group = self.chemical_names
        total_mass = df_copy.loc[:, [by]].to_numpy()
        components = df_copy.loc[:, group].to_numpy()
        normalized_components = components / total_mass
        df_copy[by] = 1
        df_copy[group] = normalized_components
        if inplace:
            self._df = df_copy
        else:
            return df_copy.fillna(0)
    
    @property
    def chemicals(self):
        df_copy = self._df.copy()
        df_copy.fillna(0, inplace=True)
        return df_copy.loc[:, self.chemical_names]

    @property
    def lce(self):
        df_copy = self._df.copy()
        return df_copy.loc[:, [self.lce_name]]

if __name__ == "__main__":
    # conn = make_connection(db="test_db", username="zliang", pwd="Automat46305!")
    db = Database(db="test_db")
    df = db.pull(table="half_cell_classifier_test")
    