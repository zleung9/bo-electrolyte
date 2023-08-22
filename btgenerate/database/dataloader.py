from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
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
    
    
class ChemicalInputDataSet(AutomatDataSet):
    
    def __init__(self, df:pd.DataFrame=None):
        self.database = "mars_db"
        self.table = "chemical_input"
        if df is None:
            df=Database(db=self.database).pull(table=self.table)
        super().__init__(df=df)
    
    @property
    def chemical_names(self):
        return self._df["chemical"].tolist()


class ManualMaterialsDataSet(AutomatDataSet):
    def __init__(self, df:pd.DataFrame=None):
        self.database = "mars_db"
        self.table = "manual_materials"
        if df is None:
            df=Database(db=self.database).pull(table=self.table)
        super().__init__(df=df)
        self.info_columns = [
            'generation_id', 'electrolyte_id', 'note', 'generation_project', 'experiment',
            'generation_method', 'total_mass(g)',
        ]
        self.chemical_names = df.columns[~df.columns.isin(self.info_columns)].tolist()
    
    @property
    def chemicals(self):
        df_copy = self._df.copy()
        df_copy.fillna(0, inplace=True)
        return df_copy.loc[:, self.chemical_names]

    @property
    def electrolyte_ids(self):
        df_copy = self._df.copy()
        return df_copy.loc[:, "electrolyte_id"]
    
    
class LiquidMasterTableDataSet(AutomatDataSet):
    def __init__(self, df:pd.DataFrame=None):
        self.database = "FMT"
        self.table = "Liquid Master Table"
        if df is None:
            df = Database(db=self.database).pull(table=self.table)
        super().__init__(df=df)
        self.info_columns = [ # columns except chemicals
            'Electrolyte ID', 'lab_batch', 'note', 'total_mass(g)', 'generation_method', 
            'generation_project', 'experiment', 'Conductivity', 'Voltage', 'Cycles', 'LCE', 
            'Initial Li efficiency', 'generation_id', 'Predicted Conductivity', 
            'Predicted Voltage', 'Predicted LCE'
        ]
        self.chemical_names = df.columns[~df.columns.isin(self.info_columns)].tolist()

    def map_names(self, name_dict:dict=None, inplace=True):
        """ Rename the main dataframe in the dataset according to `name_dict`.
        Parameters
            name_dict:  python dictionary. If `None` use the internal `_name_dict`.
        Returns
            dataframe: The renamed dataframe. 
        """

        _name_dict = {
            "1,2-dimethoxyethane": "DME",
            "1,3-dioxolane": "DOL",
            "Aluminium Oxide": "Al2O3",
            "Lithium Perchlorate": "LiClO4"
        }

        df_copy = self._df.copy()
        if name_dict is not None:
            _name_dict = name_dict
        df_copy.rename(columns=_name_dict, inplace=True)
        if inplace:
            # Update chemical names
            for old, new in _name_dict.items():
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
        df_copy.fillna(0, inplace=True)
        return df_copy.loc[:, self.chemical_names]
    
    @property
    def electrolyte_ids(self):
        df_copy = self._df.copy()
        return df_copy.loc[:, "Electrolyte ID"]
    
    
    def target(self, target="LCE"):
        df_copy = self._df.copy()
        return df_copy.loc[:, target]
    
    def pull_data(self, subset="all", target="LCE"):
        _df = pd.concat([self.chemicals, self.target(target)], axis=1)
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

    def reduced_space(self, electrolyte_ids, target="LCE"):
        """Given a list of electrolyte ID's, return only their non-zero components, with LCE.
        """
        _df = self.dataframe
        indices = _df.index[_df["Electrolyte ID"].isin(electrolyte_ids)]
        chemicals = self.chemicals.loc[indices]
        absent_chemicals = chemicals.columns[(chemicals == 0).all(axis=0)]
        df_reduced = pd.concat(
            [
                self.electrolyte_ids.loc[indices],
                # Only chemicals in presence.
                self.chemicals.loc[indices, ~chemicals.columns.isin(absent_chemicals)],
                self.target(target)[indices]
            ], axis=1
        )
    
        return df_reduced

    def parallel_plot(self, electrolyte_ids, target="LCE", title=None):
        """Generate a parallel plot of non-zero components and LCE given a list of electrolyte ID's.
        """
        _df = self.reduced_space(electrolyte_ids, target=target)
        fig, ax = plt.subplots(figsize=(_df.shape[1]-1, 4))
        pd.plotting.parallel_coordinates(_df, class_column="Electrolyte ID", ax=ax)
        ax.set_title(title)
        ax.set_ylim([0,1])
        ax.legend(loc="best")
        plt.close()
        return fig


if __name__ == "__main__":
    db = Database(db="FMT")
    df = db.pull(table="Liquid Master Table")
    ds = LiquidMasterTableDataSet(df=df)
    X, y = ds.pull_data(subset="lab")
    