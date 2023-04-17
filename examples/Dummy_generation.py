

import pandas as pd
from abc import ABC
import torch
from torch import nn

from ..search.recipe_generator import BaseRecipeGenerator, BaseRecipePredictor
from ..utils import Parameters
from ..utils.data_loader import RecipeDataset
from torch.utils.data import Dataset
from botorch.models.model import Model, ModelList
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
DTYPE = torch.double


class BaseModel(GPyTorchModel, ABC):
    """ Base model that contains Automat related interface.
    """
    def __init__(self):
        ...

class BaseGenerator(ABC):
    """ Base generator that contains Automat related interface.
    """
    ...

class DummyRecipePredictor(BaseModel):
    """ It is an DNN model but can also interact with Automat workflow.
    """
    def __init__(self):
        super().__init__()
    
    def train(self, train_x, train_y):
        assert train_x is not None
        assert train_y is not None

    def predict(self, x):
        return torch.ones_like(x)
    

class DummyRecipeGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()
    
    def get_model(self, model):
        self.model = model
    
    def get_data(self, data: Dataset):
        self.data = data
    
    def train(self):
        return 0
    
    def generate_batch(self):
        self.model.predict()


data = RecipeDataset()
model = DummyRecipePredictor()
trainer = DummyRecipeGenerator()

trainer.get_model(model)
trainer.get_data(data)
trainer.train()
new_data = trainer.generae_batch()
# measurements....


