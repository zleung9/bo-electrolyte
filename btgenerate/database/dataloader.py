from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from btgenerate.database.database import Database

_name_dict = {
    "1,2-dimethoxyethane": "DME",
    "1,3-dioxolane": "DOL",
    "Aluminium Oxide": "Al2O3",
    "Lithium Perchlorate": "LiClO4"
}

class RecipeDataset(Dataset):
    """The base model of pandas dataframe tailored for Automat Solutions, Inc.
    """
    def __init__(self, df:pd.DataFrame=None):
        super().__init__()
        self._df = df.copy()
        if "electrolyte_id" in df.columns:
            self.electrolyte_id_col = "electrolyte_id"
        elif "Electrolyte ID" in df.columns:
            self.electrolyte_id_col = "Electrolyte ID"
        else:
            self.electrolyte_id_col = None

    @property
    def dataframe(self):
        return self._df.copy()

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx=None):    
        return self._df.loc[idx, :]

    @property
    def info_columns(self):
        raise NotImplementedError("Must define self.info_columns!")

    @property
    def chemical_names(self):
        """ Return column names that are chemicals
        """
        return self._df.columns[~self._df.columns.isin(self.info_columns)].tolist()
    
    @property
    def chemicals(self):
        df_copy = self.dataframe.fillna(0)
        return df_copy.loc[:, self.chemical_names]
    
    @property
    def electrolyte_ids(self):
        df_copy = self._df.copy()
        return df_copy.loc[:, self.electrolyte_id_col]

    def map_names(self, name_dict:dict=None, inplace=True):
        """ Rename the main dataframe in the dataset according to `name_dict`.
        Parameters
            name_dict:  python dictionary. If `None` use the internal `_name_dict`.
        Returns
            dataframe: The renamed dataframe. 
        """
        global _name_dict
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
        _df = self._df.copy()
        total_mass = _df[self.chemical_names].sum(axis=1).to_numpy().reshape(-1, 1)
        components = _df[self.chemical_names].to_numpy()
        normalized_components = components / total_mass
        if by is not None:
            _df[by] = 1
        _df[self.chemical_names] = normalized_components
        if inplace:
            self._df = _df
        else:
            return _df.fillna(0)

    def find_by_component(self, space, inclusive=False, reduced=True, target="LCE"):
        """Given a component space, select all the electrolytes that only contains chemicals in
        this space. If not `inclusive`, all components in `space` should be non-zero, otherwise 
        subspaces of `space` is also selected. 
        """
        _df = self.dataframe.fillna(0)
        present_chemicals = [space] if type(space) is str else list(space)
        absent_chemicals = [c for c in self.chemical_names if c not in space]
        select = (_df.loc[:, absent_chemicals] == 0).all(axis=1)
        if not inclusive: # chemicals in space should be all non-zero
            select = select & (_df.loc[:, present_chemicals] > 0).all(axis=1)
        eids = self.electrolyte_ids.loc[select].tolist()
        return self.find_by_eid(eids, target=target)

    def find_by_eid(self, electrolyte_ids, show_space=False, target="LCE"):
        """Given a list of electrolyte ID's, return only their non-zero components, with LCE.
        If `show_space` is `True`, return the the chemical names of the common space.
        """
        if type(electrolyte_ids) is str:
            electrolyte_ids = [electrolyte_ids]
        _df = self.dataframe.fillna(0)
        indices = _df.index[self.electrolyte_ids.isin(electrolyte_ids)]
        all_chemicals = self.chemicals.loc[indices]
        present_chemicals = all_chemicals.columns[(all_chemicals > 0).all(axis=0)].tolist()
        if show_space:
            return present_chemicals
        absent_chemicals = all_chemicals.columns[(all_chemicals == 0).all(axis=0)].tolist()
        if target == "all":
            target = ["LCE", "Voltage", "Conductivity"]
        elif type(target) is str or target is None:
            target =  [target]
        target = [t for t in target if t in self.info_columns]
        df_reduced = pd.concat(
            [
                self.electrolyte_ids.loc[indices],
                # Only chemicals in presence.
                self.chemicals.loc[indices, ~all_chemicals.columns.isin(absent_chemicals)],
                _df.loc[indices, target]
            ], axis=1
        )
        return df_reduced


    def parallel_plot(self, electrolyte_ids, target="LCE", title=None):
        """Generate a parallel plot of non-zero components and LCE given a list of electrolyte ID's.
        """
        _df = self.find_by_eid(electrolyte_ids, target=target)
        fig, ax = plt.subplots(figsize=(_df.shape[1]-1, 4))
        pd.plotting.parallel_coordinates(
            _df, 
            class_column=self.electrolyte_id_col, 
            ax=ax,
            colormap="tab10"
        )
        ax.set_title(title)
        ax.set_ylim([0,1])
        ax.legend(loc="best", ncol=6, fontsize=8)
        plt.close()
        return fig
    
class ChemicaltDataset():
    
    def __init__(self, df:pd.DataFrame=None):
        self.database = "mars_db"
        self.table = "chemical_input"
        if df is None:
            df=Database(db=self.database).pull(table=self.table)
        self._df = df.copy()


    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx=None):    
        return self._df.loc[idx, :]
    
    @property
    def dataframe(self):
        return self._df.copy()
    
    @property
    def chemical_names(self):
        return self._df["chemical"].tolist()

    def MW(self, chemicals):
        if type(chemicals) is str:
            chemicals = [chemicals]
        mw = self._df.loc[self._df["chemical"].isin(chemicals), "MW"].values
        if len(mw) == 1:
            mw = mw[0]
        return mw

    def map_names(self, name_dict:dict=None, inplace=True):
        """ Rename the main dataframe in the dataset according to `name_dict`.
        Parameters
            name_dict:  python dictionary. If `None` use the internal `_name_dict`.
        Returns
            dataframe: The renamed dataframe. 
        """
        global _name_dict
        df_copy = self._df.copy()
        if name_dict is not None:
            _name_dict = name_dict
        
        for old_name, new_name in _name_dict.items():
            df_copy.loc[df_copy["chemical"] == old_name, "chemical"] = new_name        

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

class ManualMaterialsDataset(RecipeDataset):
    def __init__(self, df:pd.DataFrame=None):
        self.database = "mars_db"
        self.table = "manual_materials"
        if df is None:
            df=Database(db=self.database).pull(table=self.table)
        super().__init__(df=df)
        self._df["note"] = self._df["note"].apply(lambda x: str(x, "UTF-8"))
    
    @property
    def info_columns(self):
        return [
            'generation_id', 'electrolyte_id', 'note', 'generation_project', 'experiment',
            'generation_method', 'total_mass(g)',
        ]
    

class LiquidMasterTableDataset(RecipeDataset):
    def __init__(self, df:pd.DataFrame=None):
        self.database = "FMT"
        self.table = "Liquid Master Table"
        if df is None:
            df = Database(db=self.database).pull(table=self.table)
        super().__init__(df=df)

    @property
    def info_columns(self):
        return [ # columns except chemicals
            'Electrolyte ID', 'lab_batch', 'note', 'total_mass(g)', 'generation_method', 
            'generation_project', 'experiment', 'Conductivity', 'Voltage', 'Cycles', 'LCE', 
            'Initial Li efficiency', 'generation_id', 'Predicted Conductivity', 
            'Predicted Voltage', 'Predicted LCE'
        ]
    
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


if __name__ == "__main__":
    # run:
    # >>> export QT_QPA_PLATFORM=offscreen 
    ds_lmt = LiquidMasterTableDataset()
    ds_lmt.normalize_components(inplace=True)
    ds_lmt.map_names()
    print(ds_lmt.find_similar("21-7-583", space_only=True))