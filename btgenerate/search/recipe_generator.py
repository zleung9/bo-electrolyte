from abc import ABC, abstractmethod
import pandas as pd
import torch
from torch import nn
from btgenerate.utils.utils import Parameters
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


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
    def load_data(self, data):
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

    def train(self, model, train_X=None, train_Y=None):
        """Train model based on the given data: inputs/outputs.
        User must override this method. Must take `train_x` and `train_y` as input and must return
        a `RecipePredictor` object.
        
        Returns
        -------
        self.model : `BaseRecipeRedictor`
        """
        raise NotImplementedError
    
    @abstractmethod
    def generate_batch(self):
        """Takes in the model and existing data points, generate a new batch of data points.
        User must override this method.

        Returns
        -------
        self.next_x : array_like must be a tensor object
            Training features of shape (N, M) where N is the number of points and M is the number of 
        features.
        """
        raise NotImplementedError


    def push_data(self):
        """ Push the predicted value to the database.
        """
        raise NotImplementedError

    def notify_slack(self):
        raise NotImplementedError

    def _transform_prediction(self):
        """
        """
        ...

class SimplePredictor(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.ones(x.shape[:-1])

class SimpleGenerator(BaseRecipeGenerator):
    
    def __init__(self):
        super().__init__()
    
    def load_data(self, X, y):
        """Load data.
        """
        self.x_train = X
        self.y_train = y
        assert len(self.x_train.shape) == 2
        self.dim = self.x_train.shape[1]
        self.n_candidates = min(5000, max(2000, 200 * self.dim))
        return self.x_train, self.y_train

    def train(self, model, x_train=None, y_train=None):
        model.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # fit_gpytorch_mll(mll)
        pass
        model.eval()
        return model
    
    def generate_batch(
            self, 
            data, 
            model, 
            batch_size=None, 
            num_restarts=10,
            raw_samples=512,
        ):
        """Use model and 
        """
        X_train, y_train = data
        if batch_size is None:
            batch_size = self.batch_size
        assert X_train.min() >= 0.0 and X_train.max() <= 1.0 and torch.all(torch.isfinite(y_train))

        # Create a batch
        ei = qExpectedImprovement(model, y_train.max())
        dim = X_train.shape[1]
        bounds = torch.stack(
            [
                torch.zeros(dim, dtype=DTYPE, device=DEVICE),
                torch.ones(dim, dtype=DTYPE, device=DEVICE),
            ]
        )
        X_next, _ = optimize_acqf(
            ei,
            bounds=bounds,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        posterior = model.posterior(X_next)
        y_next = posterior.sample(sample_shape=torch.Size([100])).mean(axis=0)
        
        self.x_next = X_next
        self.y_next = y_next

        return self.x_next, self.y_next