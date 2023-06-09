from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from btgenerate.database.database import Database


class AutomatDataSet(Dataset):
    """The base model of pandas dataframe tailored for Automat Solutions, Inc.
    """
    def __init__(self, df:pd.DataFrame=None):
        super().__init__()
        self._df = df.copy()
    
    @property
    def dataframe(self):
        return self._df.copy()

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx=None):    
        return self._df.loc[idx, :]
    

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
        group = list(set(self.chemical_names) & set(df_copy.columns))
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
        real_chemical_names = list(set(self.chemical_names) & set(df_copy.columns))
        df_copy.fillna(0, inplace=True)
        return df_copy.loc[:, real_chemical_names]

    @property
    def lce(self):
        df_copy = self._df.copy()
        return df_copy.loc[:, [self.lce_name]]
    
    def pull_data(self, subset="all"):
        _df = pd.concat([self.chemicals, self.lce], axis=1)
        if subset == "lab":
            _df = _df.loc[~self._df["lab_batch"].isna()]
        elif subset == "ml":
            _df = _df.loc[self._df["lab_batch"].isna()]
        elif subset != "all":
            raise Exception("Subset must be one of the following: all, lab, ml")
        _df.dropna(axis=0, how="any", subset=self.lce_name, inplace=True)
        X = torch.from_numpy(_df.iloc[:, :-1].to_numpy(dtype=float))
        y = torch.from_numpy(_df.iloc[:, [-1]].to_numpy(dtype=float))
        return X, y

if __name__ == "__main__":
    db = Database(db="FMT")
    df = db.pull(table="Liquid Master Table")
    ds = LiquidMasterDataSet(df=df)
    X, y = ds.pull_data(subset="lab")
    