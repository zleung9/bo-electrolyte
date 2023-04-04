from abc import ABC, abstractmethod
import pandas as pd


class BaseRecipeGenerator(ABC):
    
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_next = None
        self.y_next = None
        self.model = None
        
    @abstractmethod
    def pull_data(self, data):
        """Get `x_train` and `y_train` from the `data` source, which could be of any form.
        User must override this method. Must return the following: 
        train_x: array_like
            Training features of shape (N, M) where N is the number of points and M is the number
            of features
        train_y: array_like
            Training target of shape (N, K) where N is the number of points and K is the number of 
            targets to optimize (output dimension). Usually K=1.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_model(self, x=None, y=None):
        """Train model based on the given data: inputs/outputs.
        User must override this method. Must take `train_x` and `train_y` as input and must return
        a `RecipePredictor` object.
        """
        raise NotImplementedError
    
    def update_model(self, x=None, y=None):
        """Train model based on the given data: inputs/outputs.
        User must override this method. Must take `train_x` and `train_y` as input and must return
        a `BaseRecipePredictor` object.
        
        Returns
        -------
        self.model : `BaseRecipeRedictor`
        """
        ...

    @abstractmethod
    def generate_batch(self):
        """Takes in the model and existing data points, generate a new batch of data points.
        User must override this method.
        """
        raise NotImplementedError


    def push_data(self):
        """ Push the predicted value to the database.
        """
        ...

    def notify_slack(self):
        ...


class BaseRecipePredictor(ABC):
    """ An Automat wrapper of any model that predicts/suggest the optimized 
    """
    def __init__(self):
        ...
    
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        """A method that predicts the new points given input features.
        Returns
        ------- 
        self.y_pred : array_like
            An array of shape (N, M) where N is the number of predictions, M is the dimension.
        """
        raise NotImplementedError



class DragonflyRecipeGenerator(BaseRecipeGenerator):
    def __init__(self):
        super().__init__()

    

class RGPERecipeGenerator(BaseRecipeGenerator):
    def __init__(self):
        super().__init__()
