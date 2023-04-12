

import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Conv1D, Lambda, concatenate, MaxPooling1D, BatchNormalization,Input
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Model

from sklearn.model_selection import train_test_split


from ..recipe_generator import BaseRecipeGenerator, BaseRecipePredictor
from ..data_loader import RecipeDataset


def BLOCK(seq, filters):
    cnn = Conv1D(filters*2, 3, padding='SAME', dilation_rate=1, activation='relu')(seq)
    cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
    cnn = Conv1D(filters*2, 3, padding='SAME', dilation_rate=2, activation='relu')(cnn)
    cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
    cnn = Conv1D(filters*2, 3, padding='SAME', dilation_rate=4, activation='relu')(cnn)
    cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
    if int(seq.shape[-1]) != filters:
        seq = Conv1D(filters, 1, padding='SAME')(seq)
    seq = concatenate([seq, cnn])
    return seq

def test_su_1d_model(input_dim):

    input_tensor = Input(shape=input_dim)
    seq = input_tensor

    seq = BLOCK(seq, 16)
    seq = MaxPooling1D(2)(seq)
    seq = Dropout(0.5)(seq)
    seq = BLOCK(seq, 16)
    seq = MaxPooling1D(2)(seq)
    seq = BLOCK(seq, 32)
    # seq = MaxPooling1D(2)(seq)
    seq = BLOCK(seq, 32)
    # seq = MaxPooling1D(2)(seq)
    seq = BLOCK(seq, 64)
    # seq = MaxPooling1D(2)(seq)
    seq = BLOCK(seq, 64)
    # seq = MaxPooling1D(2)(seq)
    seq = BLOCK(seq, 128)
    # seq = MaxPooling1D(2)(seq)
    seq = BLOCK(seq, 128)
    # seq = GlobalMaxPooling1D()(seq)
    seq = Flatten()(seq)
    seq = Dropout(0.5)(seq)
    seq = Dense(128, activation='relu')(seq)

    output_tensor = Dense(1, activation='linear')(seq)

    model = Model(inputs=[input_tensor], outputs=[output_tensor])
    model.summary()
    model.compile(loss='mean_squared_error',
              optimizer=Adam(1e-3))
    return model
    
class DNN(BaseRecipePredictor):   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model =  None
        
    def dnn_regression(self, input_dim, batch_size=64, trainable=True, lr = 1e-3, layer=[128,256,512,1024], batch_norm_after_layer=False, **kwargs) -> keras.models.Model:
        self.BATCH_SIZE = batch_size
        input_tensor = Input(shape=input_dim)
        seq = input_tensor

        for i in range(len(layer)):
            seq = Dense(layer[i], kernel_initializer='he_normal', activation='relu')(seq)
            if batch_norm_after_layer:
                seq = BatchNormalization()(seq)

        seq = Dropout(0.5)(seq)
        seq = Flatten()(seq)
        output_tensor = Dense(1, kernel_initializer='he_normal', activation='linear')(seq)

        model = Model(inputs=[input_tensor], outputs=[output_tensor])
        if trainable is False:
            for l in model.layers[1:-1]: 
                l.trainable=False

        model.summary()
        model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr))
        self.model = model
        return model
    
    def train(self, skip_chemical: list=[], target: str = "LCE",
                plot_results: bool = True,
                no_sediment: bool = False,
                only_lab_approved_chems: bool = True, 
                load_scaler=False, model_type="dnn", estimator="dnn", project_prefix="/home/dongpeng/automat_proj/automat_predictor/src/single_target/train_model/",random_state=42,BATCH_SIZE=32):

        dataset = RecipeDataset(target="LCE",
                             omit_lab_batches=[60, 61], only_lab_approved_chems=False,
                             project="DOE_electrolyte", 
                             table="Liquid Master Table",
                             y_log_transform=False, project_prefix=project_prefix)
        X, y = dataset.pull_data()

        # splitting of the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

        # X_train, X_test, y_train, y_test = train_test_split(x, y_conductivity, test_size=0.1, shuffle=True, random_state=random_state )

        X_train = X_train.values.reshape(X_train.shape+(1,))
        # X_val = X_val.reshape(X_val.shape+(1,))
        y_train = y_train.values.reshape(y_train.shape+(1,))
        # y_val = y_val.reshape(y_val.shape+(1,))
        X_test = X_test.values.reshape(X_test.shape+(1,))
        y_test = y_test.values.reshape(y_test.shape+(1,))


        self.model = self.dnn_regression(X_train.shape[1:])

        checkpoint = ModelCheckpoint(project_prefix+"saved_models/critic_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='max')
        print(X_train)


        history = self.model.fit(X_train, y_train, validation_split=0.1, batch_size=BATCH_SIZE, callbacks=[], epochs=500, verbose=1)
        y_pred = self.model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

        
        max_Test_R2 = float("-inf")
        recorded_mse = float("-inf")
        recorded_mae = float("-inf")
        recorded_corr = float("-inf")
        recorded_pVal = float("-inf")
        best_params = None

        Current_Test_R2 = r2_score(y_test, y_pred)
        print("msescore: {:.4f} ".format(mean_squared_error(y_test, y_pred)))
        print("spearmanr ", stats.spearmanr(y_pred, y_test))
        print("R2 score ", Current_Test_R2)
        print(X_train.shape)
        print(y_test.shape)
        print(y_pred.shape)



        return model


    def predict(self, x, loadweight=None):
        """A method that predicts the new points given input features.
        Returns
        ------- 
        self.y_pred : array_like
            An array of shape (N, M) where N is the number of predictions, M is the dimension.
        """
        self.y_pred = self.model.predict(x, batch_size=self.BATCH_SIZE, verbose=1)
        return self.y_pred