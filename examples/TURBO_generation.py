import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.acquisition.objective import ConstrainedMCObjective
from btgenerate.search.recipe_generator import SimpleGenerator, SimplePredictor
from btgenerate.database.dataloader import LiquidMasterDataSet
from btgenerate.database.database import Database
from gpytorch.likelihoods import GaussianLikelihood



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


def test_generator():    
    
    # db = Database(db="FMT")
    # df = db.pull(table="Liquid Master Table")
    # ds = LiquidMasterDataSet(df=df)
    # ds.normalize_components(inplace=True)
    # x_train, y_train = ds.pull_data()

    x_train = torch.rand(size=(100, 20), device=DEVICE, dtype=DTYPE)
    x_train = x_train / x_train.sum(axis=1).unsqueeze(-1)
    y_train = torch.rand(size=(100, 1), device=DEVICE, dtype=DTYPE)

    generator = SimpleGenerator()
    generator.load_config("./config.yaml")
    generator.load_data(x_train, y_train)
    
    predictor = SimplePredictor()
    model = SingleTaskGP(
        x_train, 
        y_train,
        mean_module=predictor
    )
    constraint1 = (torch.arange(x_train.shape[1]), torch.ones(x_train.shape[1], dtype=DTYPE), 1)

    model = ModelListGP(model)
    model = generator.train_model(model)

    x_next, y_next = generator.generate_batch(
        data=(x_train, y_train), 
        model=model,
        equality_constraints=[constraint1]
    )
    print(x_next.shape, y_next.shape)

def test_loop():
    torch.manual_seed(0)
    from botorch.test_functions import Ackley
    from botorch.utils.transforms import unnormalize
    from torch.quasirandom import SobolEngine
    import matplotlib.pyplot as plt
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.constraints import Interval

    dim = 5
    fun = Ackley(dim=dim, negate=True).to(dtype=DTYPE, device=DEVICE)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    n_init = 2 * dim

    def eval_objective(x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return fun(unnormalize(x, fun.bounds))

    def get_initial_points(dim, n_pts, seed=0):
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts).to(dtype=DTYPE, device=DEVICE)
        return X_init
    
    x_train = get_initial_points(dim, n_init)
    y_train = torch.tensor(
        [eval_objective(x) for x in x_train], dtype=DTYPE, device=DEVICE
    ).unsqueeze(-1)
    y_pred = []
    generator = SimpleGenerator()
    generator.load_config("./config.yaml")
    generator.load_data(x_train, y_train)
    
    predictor = SimplePredictor()
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )
    for i in range(100):
        
        y_train_norm = (y_train - y_train.mean()) / y_train.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        model = SingleTaskGP(
            x_train, 
            y_train_norm,
            # mean_module=predictor,
            likelihood=likelihood
        )
        # constraint1 = (torch.arange(x_train.shape[1]), torch.ones(x_train.shape[1], dtype=DTYPE), 1)
        # model = ModelListGP(model)
        model = generator.train_model(model)

        x_next, y_pred = generator.generate_batch(
            data=(x_train, y_train_norm), 
            model=model,
            # equality_constraints=[constraint1]
        )
        y_next = torch.tensor(
            [eval_objective(x) for x in x_next], dtype=DTYPE, device=DEVICE
        ).unsqueeze(-1)

        x_train = torch.cat((x_train, x_next), dim=0)
        y_train = torch.cat((y_train, y_next), dim=0)
        print(f"{i}/100\t, {abs(y_pred-y_next).mean()}")
    plt.plot(y_train)
    plt.savefig("test.png")   
    pass


def test_loop1():
    torch.manual_seed(0)
    from botorch.test_functions import Ackley
    from botorch.utils.transforms import unnormalize
    from torch.quasirandom import SobolEngine
    import matplotlib.pyplot as plt
    from gpytorch.constraints import Interval
    from botorch.optim import optimize_acqf
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition import qExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from botorch.test_functions import Ackley
    from botorch.utils.transforms import unnormalize
    from torch.quasirandom import SobolEngine
    dim = 5
    fun = Ackley(dim=dim, negate=True).to(dtype=DTYPE, device=DEVICE)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    n_init = 2 * dim

    def eval_objective(x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return fun(unnormalize(x, fun.bounds))

    def get_initial_points(dim, n_pts, seed=0):
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts).to(dtype=DTYPE, device=DEVICE)
        return X_init
    
    X_ei = get_initial_points(dim, n_init)
    Y_ei = torch.tensor(
        [eval_objective(x) for x in X_ei], dtype=DTYPE, device=DEVICE
    ).unsqueeze(-1)

    for i in range(100):
        train_Y = (Y_ei - Y_ei.mean()) / Y_ei.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        model = SingleTaskGP(X_ei, train_Y, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Create a batch
        ei = qExpectedImprovement(model, train_Y.max())
        candidate, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack(
                [
                    torch.zeros(dim, dtype=DTYPE, device=DEVICE),
                    torch.ones(dim, dtype=DTYPE, device=DEVICE),
                ]
            ),
            q=4,
            num_restarts=10,
            raw_samples=512,
        )
        Y_next = torch.tensor(
            [eval_objective(x) for x in candidate], dtype=DTYPE, device=DEVICE
        ).unsqueeze(-1)

        # Append data
        X_ei = torch.cat((X_ei, candidate), axis=0)
        Y_ei = torch.cat((Y_ei, Y_next), axis=0)

        # Print current status
        print(f"{len(X_ei)}) Best value: {Y_ei.max().item():.2e}")
if __name__ == "__main__":
    ######
    # run:
    # >>> export QT_QPA_PLATFORM=offscreen 
    ######
    test_loop1()
    # test_generator()