import torch
from botorch.models import SingleTaskGP

from btgenerate.search.recipe_generator import SimpleGenerator, SimplePredictor
from btgenerate.database.dataloader import LiquidMasterDataSet
from btgenerate.database.database import Database


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


def test_generator():    
    
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
    generator.train(model, x_train=x_train, y_train=y_train)

    x_next, y_next = generator.generate_batch((x_train, y_train), model)
    print(x_next.shape, y_next.shape)


if __name__ == "__main__":
    test_generator()