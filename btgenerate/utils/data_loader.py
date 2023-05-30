from __future__ import print_function, division
import os
import torch
import pandas as pd

import numpy as np

from torch.utils.data import Dataset, DataLoader

from sklearn import preprocessing
import re

from btgenerate.chemicals.chemical import Chemical
from btgenerate.database.database import Database, LiquidMasterDataSet

class RecipeDataset(Dataset):
    """RecipeDataset. API: torch.utils.data.Dataset"""

    def __init__(
            self, 
            csv_path=None, 
            target="LCE",
            db = "FMT", 
            table="Liquid Master Table",
            source="lab_batch"
    ):
        if csv_path is not None:
            self.data = pd.read_csv(csv_path)
        else:
            self._load_database(db=db, table=table)

    def _load_database(self, db="FMT", table="Liquid Master Table", source="lab_batch"):
        if db is None or table is None:
            db = "FMT"
            table = "Liquid Master Table" 
        db = Database(db=db)
        df = db.pull(table=table)
        ds_lm = LiquidMasterDataSet(df)
        ds_lm.map_names()
        ds_lm.normalize_components(by="total_mass(g)", inplace=True)
        
        self.data = ds_lm.dataframe
        if source == "lab_batch":
            self.data = self.data.dropna(subset="lab_batch")

    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx=None):
        
        if idx and torch.is_tensor(idx):
            idx = idx.tolist()
            self.data = self.data.iloc[idx, :]

        return self.data
    
    def pull_data(self, data=None):
        if not data:
            data = self.data
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y
