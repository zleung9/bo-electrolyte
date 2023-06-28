import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.acquisition.objective import ConstrainedMCObjective
from btgenerate.search.recipe_generator import SimpleGenerator, SimplePredictor
from btgenerate.database.dataloader import LiquidMasterDataSet
from btgenerate.database.database import Database



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


def test_generator():    
    def outcome_constraint(X):
        """L1 constraint; feasible if less than or equal to zero."""
        return X.sum(dim=-1) - 1
    
    db = Database(db="FMT")
    df = db.pull(table="Liquid Master Table")
    ds = LiquidMasterDataSet(df=df)
    ds.normalize_components(inplace=True)
    x_train, y_train = ds.pull_data()

    generator = SimpleGenerator()
    generator.load_config("./config.yaml")
    generator.load_data(x_train, y_train)
    
    predictor = SimplePredictor()
    model = SingleTaskGP(
        x_train, 
        y_train,
        mean_module=predictor
    )
    c_train = outcome_constraint(x_train).unsqueeze(-1) # add output dimension
    constraint = SingleTaskGP(
        x_train,
        c_train,
    ).to(x_train)

    model_with_constraint = ModelListGP(model, constraint)
    model_with_constraint = generator.train_model(
        model_with_constraint, 
        x_train=x_train, 
        y_train=y_train
    )
    constrained_objective = ConstrainedMCObjective(
        objective=lambda Z: Z[..., 0],
        constraints=[lambda Z: Z[..., 1]],
    )
    x_next, y_next = generator.generate_batch(
        data=(x_train, y_train), 
        model=model_with_constraint,
        objective=constrained_objective
    )
    print(x_next.shape, y_next.shape)


if __name__ == "__main__":
    
    test_generator()