

import pandas as pd

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import db_connect.connector as db
import fmt_access.fmt_driver as fmt


from .batch_generation import TurboState, generate_batch, update_state
from .recipe_generator import BaseRecipeGenerator, BaseRecipePredictor
from .utils import Parameters




# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
DTYPE = torch.double


class TurboRecipeGenerator(BaseRecipeGenerator):
    
    def __init__(self):
        super().__init__()
        self.state = None
        self.config = None
        self.dim = None # dimensionality of the problem
        self.n_candidates = None
        self.raw_samples = None

    def load_config(self, config_path):
        """Load configuration to an existing generator
        """
        self.config = Parameters.from_yaml(config_path)
        self.batch_size = self.config.batch_size
        if self.config.max_cholesky_size == "inf":
            self.max_cholesky_size = float("inf")
        self.raw_samples = self.config.raw_samples
        self.n_restarts = self.config.num_restarts

    def initialize_state(self, initialize=True, state_path=None):
        self._update_state(initialize=initialize, state_path=state_path)
        
    def _update_state(self, initialize=False, state_path=None):
        """Update the Turbo state of the generator upon new data points. 
        """
        if initialize:
            if state_path is None:
                assert self.dim is not None, "Please load data first"
                assert self.batch_size is not None, "Please check 'batch_size'"
                self.state = TurboState(self.dim, batch_size=self.batch_size)
            else:
                ...
        elif self.y_next is None:
            print("State not updated. y_next is None")
        else:
            assert self.state is not None
            assert self.y_next is not None
            self.state = update_state(self.state, self.y_next) 

    def get_model(self):
        self.model = GaussianProcessModel(
            max_cholesky_size=self.max_cholesky_size
        )
        return self.model

    def pull_data(self, data: pd.DataFrame):
        """Load data.
        """
        df = data
        self.x_train = df.iloc[:, :-1].values
        self.y_train = df.iloc[:, [-1]].values
        assert len(self.x_train.shape) == 2
        self.dim = self.x_train.shape[1]
        self.n_candidates = min(5000, max(2000, 200 * self.dim))
        return self.x_train, self.y_train
    
    def generate_batch(self, data, model, batch_size=None, acquisition="ei"):
        """Use model and 
        """
        x_train, y_train = data
        if batch_size is None:
            batch_size = self.batch_size
        
        # Create a batch
        x_next = generate_batch(
            state=self.state,
            model=model.model,
            X=x_train,
            Y=y_train,
            batch_size=batch_size,
            n_candidates=self.n_candidates,
            num_restarts=self.n_restarts,
            raw_samples=self.raw_samples,
            acqf=acquisition,
            device=DEVICE
        )
        y_next = model.predict(x_next)
        
        self.x_next = x_next
        self.y_next = y_next
        self._update_state()

        return self.x_next, self.y_next
        


class GaussianProcessModel(BaseRecipePredictor):
    
    def __init__(self, max_cholesky_size):
        super().__init__()
        self.max_cholesky_size = max_cholesky_size
        self.num_outputs=1
    
    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.dim = x_train.shape[1]
        self.likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        self.covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, 
                ard_num_dims=self.dim, 
                lengthscale_constraint=Interval(0.005, 4.0)
            )
        )
        self.model = SingleTaskGP(
            self.x_train, 
            self.y_train,
            covar_module=self.covar_module, 
            likelihood=self.likelihood
        )
        self.posterior = self.model.posterior(x_train)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        # Do the fitting inside the Cholesky context
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            # Fit the model
            fit_gpytorch_mll(self.mll)

    def predict(self, x_next, sample_size=100):
        # Record the difference between prediction and ground truth
        posterior = self.model.posterior(x_next)
        y_pred = posterior.sample(
            sample_shape=torch.Size([sample_size])
        ).mean(axis=0)
        return y_pred
    


if __name__ == "__main__":
    turbo = TurboRecipeGenerator()
    turbo.load_config("./src/config.yaml")
    
    data = fmt.pull_data(target="Voltage",
                         omit_lab_batches=[60, 61], only_lab_approved_chems=False,
                         table="Liquid Master Table")
    data = data.drop(columns=['electrolyte_id', 'generation_method', 'total_mass'])

    x_train, y_train = turbo.pull_data(data)
    
    turbo.initialize_state(initialize=True)
    

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    
    model = turbo.get_model()
    model.train(x_train, y_train)
    x_next, y_next = turbo.generate_batch((x_train, y_train), model)
    print(x_next, y_next)
#     turbo.push_data((x_next, y_next))
#     turbo.notify_slack()
