from __future__ import print_function, division
import os
import torch
import pandas as pd

import numpy as np

from torch.utils.data import Dataset, DataLoader

from sklearn import preprocessing
import re

import db_connect.connector as db
import fmt_access.fmt_driver as fmt

class RecipeDataset(Dataset):
    """RecipeDataset. API: torch.utils.data.Dataset"""

    def __init__(self, csv_file=None, project_prefix="./", target="LCE",
                             omit_lab_batches=[60, 61], only_lab_approved_chems=True,
                             project="DOE_electrolyte", 
                             table="Liquid Master Table",
                             y_log_transform=False):
        """
        Arguments:
            ...
            csv_file (string, optional): Path to the csv file with annotations.
            root_dir (string, optional): Directory with all the images.
            y_log_transform (bool, optional): Optional transform to be applied
                on a sample.
            ...
        """
        if table == "Liquid Master Table":
            self.data = fmt.pull_data(target=target,
                                 omit_lab_batches=omit_lab_batches, only_lab_approved_chems=only_lab_approved_chems,
                                #  project="DOE_electrolyte", 
                                 table="Liquid Master Table")
            self.data = self.data.drop(columns=['electrolyte_id', 'generation_method', 'total_mass'])
            self.data = self.data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        else:
#             self.data = db.
            ...
            ...
        self.project_prefix = project_prefix
        
        if y_log_transform:
            newy = self.log_transform_y(self.data.iloc[:, [-1]].values)
            self.data.iloc[:, [-1]] = newy.reshape(newy.shape+(1,))

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
    
    def log_transform_y(self, y, minimal_y=1e-9):#min 1e-13
        '''adapt from materialRL chemutil
        y: ndarray
        return: ndarray, 1D
        '''
        y = np.where(y == 0, minimal_y, y)
        y = pd.DataFrame(y)
        transformer = preprocessing.FunctionTransformer(np.log, np.exp, validate=True) #np.log1p
        # scaler_y = preprocessing.MaxAbsScaler()#MinMaxScaler #StandardScaler
        scaler_y = preprocessing.MinMaxScaler()
        # y_conductivity = scaler_y.fit_transform(y_conductivity.reshape(-1,1))
        # y_conductivity = y_conductivity.flatten()

        newy = transformer.fit_transform(y.values.reshape(-1,1))
        joblib.dump(transformer, self.project_prefix+'saved_models/FunctionTransformer.pkl')

        newy = scaler_y.fit_transform(newy.flatten().reshape(-1,1))
        joblib.dump(scaler_y, self.project_prefix+'saved_models/StandardScaler.pkl')
        newy = newy.flatten()
    #     newy = pd.DataFrame(newy)
        return newy