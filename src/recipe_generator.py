from abc import ABC, abstractmethod
import pandas as pd
from .utils import Parameters
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from torch import Tensor
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.constraints import Interval
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean

class BaseRecipeGenerator(ABC):
    
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_next = None
        self.y_next = None
        self.model = None
        
    def load_config(self, config_path):
        """Load parameters from a yaml file and populate to local scope.
        """
        config = Parameters.from_yaml(config_path)
        self.__dict__.update(config.to_dict())

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


class BaseRecipePredictor(ExactGP, GPyTorchModel, ABC):
    """ An Automat wrapper of any model that predicts/suggest the optimized 
    """
    _num_outputs = 1  # to inform GPyTorchModel API
    

    def __init__(self, train_X=None, train_Y=None, **kwargs):
        """https://botorch.org/tutorials/custom_botorch_model_in_ax"""
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
#             MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1], lengthscale_constraint=Interval(0.005, 4.0))
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype
    
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

    def _transform_prediction(self):
        """
        """
        ...
    

class DragonflyRecipeGenerator(BaseRecipeGenerator):
    def __init__(self):
        super().__init__()

    

class RGPERecipeGenerator(BaseRecipeGenerator):
    def __init__(self):
        super().__init__()
