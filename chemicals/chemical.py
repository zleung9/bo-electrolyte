from abc import ABC
import pandas as pd

class Chemical(ABC):
    def __init__(self):
        self._name = None
        self._iupac = None
        self._smiles = None
        self._cas_num = None
    
    def to_dict(self):
        """Return a dict object to be stored in MangoDB.
        """
        info_dict = {
            "name": self._name,
            "IUPAC": self._iupac,
            "SMILES": self._smiles,
            "CAS #": self._cas_num,
        }
        return info_dict
        
    @classmethod
    def from_csv(cls, file_path="./chemicals.csv"):
        df = pd.read_csv(file_path)
        return df


class Solvent(Chemical):
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_csv(cls, file_path="./chemicals.csv"):
        df = pd.read_csv(file_path)
