import os
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP


import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from .turbo_state import *
from .batch_generation import *
from .objective_function import *

SMOKE_TEST = os.environ.get("SMOKE_TEST") # what is smoke test?
max_cholesky_size = float("inf")  # Always use Cholesky

# X_turbo = get_initial_points(dim, n_init)
# Y_turbo = torch.tensor(
#     [eval_objective(x) for x in X_turbo], dtype=dtype, device=device
# ).unsqueeze(-1)


def loop_generation(dim: int, batch_size: int, device):
    state = TurboState(dim, batch_size=batch_size)

    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4
    N_CANDIDATES = min(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4

    torch.manual_seed(0)
    pred_true_diff = []
    while not state.restart_triggered:  # Run until TuRBO converges
        # Fit a GP model
        train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Do the fitting and acquisition function optimization inside the Cholesky context
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            fit_gpytorch_mll(mll)

            # Create a batch
            X_next = generate_batch(
                state=state,
                model=model,
                X=X_turbo,
                Y=train_Y,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                acqf="ts",
            )

        Y_next = torch.tensor(
            [eval_objective(x) for x in X_next], dtype=dtype, device=device
        ).unsqueeze(-1)

        # Record the difference between prediction and ground truth
        posterior = model.posterior(X_next)
        Y_pred = posterior.sample(sample_shape=torch.Size([100])).mean(axis=0)
        pred_true_diff.extend(abs(Y_pred - Y_next).tolist())
        # Update state
        state = update_state(state=state, Y_next=Y_next)

        # Append data
        X_turbo = torch.cat((X_turbo, X_next), dim=0)
        Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

        # Print current status
        print(
            f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
        )