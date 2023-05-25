import torch
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
dtype = torch.double

DIM = 5


def eval_objective(x, device, batch_size: int):
    fun = Ackley(dim=DIM, negate=True).to(dtype=dtype, device=device)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    lb, ub = fun.bounds
    n_init = 2 * DIM
    max_cholesky_size = float("inf")  # Always use Cholesky

    """This is a helper function we use to unnormalize and evalaute a point"""
    return fun(unnormalize(x, fun.bounds))