from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,Dropout,BatchNormalization,Lambda,LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras import layers
from tensorflow import keras
from scipy import stats


class SurrogateModel:
    def __init__(self, name, learning_rate = 0.001, model='nn'):
        self.dims    = 40
        self.learning_rate = learning_rate
        self.model = model
        self.name = name

    def fit(self, X, y, verbose = False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(Conv1D(128, kernel_size=3, strides=1, padding='same', activation='elu', input_shape=(self.dims, 1)))
        model.add(MaxPooling1D(pool_size=2, strides=1))
        model.add(Dropout(0.2))
        model.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation='elu'))
        model.add(MaxPooling1D(pool_size=2, strides=1))
        model.add(Dropout(0.2))
        model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation='elu'))
        model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', activation='elu'))
        model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', activation='elu'))
        model.add(Flatten())
        model.add(Dense(128,activation='elu'))
        model.add(Dense(64,activation='elu'))
        model.add(Dense(1,activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')

        mc = ModelCheckpoint(f"{self.name}_NN.h5", monitor='val_loss', mode='min', verbose=verbose, save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        model.fit(X_train.reshape(len(X_train),self.dims,1), y_train, batch_size=64, epochs=500, validation_data=(X_test.reshape(len(X_test),self.dims,1), y_test), callbacks=[early_stop,mc],verbose=verbose)

        model =  keras.models.load_model(f"{self.name}_NN.h5")
        y_pred = model.predict(X_test.reshape(len(X_test),self.dims,1))

        R=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
        R=np.asarray(R).round(3)
        MAE= metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
        MAPE= metrics.mean_absolute_percentage_error(y_test.reshape(-1), y_pred.reshape(-1))
        MAE=np.asarray(MAE).round(5)
        MAPE=np.asarray(MAPE).round(5)
        print("Model performance: R2",R**2,"MAE",MAE,"MAPE",MAPE)

        return model
 