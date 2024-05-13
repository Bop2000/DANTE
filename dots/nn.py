from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,Dropout,BatchNormalization,Lambda,LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras import layers
from scipy import stats
from sklearn import metrics
import numpy as np
import pandas as pd
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt

################################ Define the CNN model ################################

class model_training:
    def __init__(self, f = None, dims=10, learning_rate = 0.001):
        self.f = f
        self.dims    = dims
        self.learning_rate = learning_rate

    def __call__(self, X,y, verbose = False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.f == 'ackley':
            if self.dims <= 100:
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
            else:
                model = Sequential()
                model.add(Conv1D(128, kernel_size=3, strides=1, padding='same', activation='elu', input_shape=(self.dims, 1)))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Dropout(0.2))
                model.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation='elu'))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Dropout(0.2))
                model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation='elu'))
                model.add(MaxPooling1D(pool_size=2, strides=1))
                model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', activation='elu'))
                model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', activation='elu'))
                model.add(Conv1D(4, kernel_size=3, strides=1, padding='same', activation='elu'))
                model.add(Flatten())
                model.add(Dense(64,activation='elu'))
                model.add(Dense(1,activation='linear'))
                model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')

        elif self.f == 'rastrigin':
            model = Sequential([
                layers.Conv1D(256,kernel_size=5,strides=1,padding='same', activation='elu', input_shape=(self.dims,1)),
                layers.LayerNormalization(),
                layers.Conv1D(128,kernel_size=5,strides=2, padding='same', activation='elu'),
                layers.Conv1D(64,kernel_size=3,strides=2, padding='same', activation='elu'),
                layers.Conv1D(32,kernel_size=3,strides=1, padding='same', activation='elu'),
                layers.Conv1D(16,kernel_size=3,strides=1, padding='same', activation='elu'),
                layers.Conv1D(8,kernel_size=3,strides=1, padding='same', activation='elu'),
                layers.Flatten(),
                Dense(128, activation='elu'),
                Dense(64, activation='elu'),
                Dense(1, activation='linear')
            ])
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_absolute_percentage_error')

        elif self.f == 'rosenbrock':
            model = Sequential()
            model.add(Conv1D(128, kernel_size=3, strides=1, padding='same', activation='elu', input_shape=(self.dims, 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
            model.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
            model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(MaxPooling1D(pool_size=2, strides=1))
            model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(Conv1D(4, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(Flatten())
            model.add(Dense(64,activation='elu'))
            model.add(Dense(1,activation='linear'))
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')


        elif self.f == 'griewank':
            model = Sequential()
            model.add(Lambda(lambda x: x / 600, input_shape=(self.dims, 1)))
            model.add(Conv1D(128, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
            model.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
            model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(MaxPooling1D(pool_size=2, strides=1))
            model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(Conv1D(4, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(Flatten())
            model.add(Dense(64,activation='elu'))
            model.add(Dense(1,activation='linear'))
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')


        elif self.f == 'levy':
            model = Sequential([
                layers.Lambda(lambda x: x / 10, input_shape=(self.dims, 1)),
                layers.Conv1D(256,kernel_size=5,strides=1,padding='same', activation='elu'),
                # layers.LayerNormalization(),
                layers.Conv1D(128,kernel_size=5,strides=2, padding='same', activation='elu'),
                layers.Conv1D(64,kernel_size=3,strides=2, padding='same', activation='elu'),
                layers.Conv1D(32,kernel_size=3,strides=1, padding='same', activation='elu'),
                layers.Conv1D(16,kernel_size=3,strides=1, padding='same', activation='elu'),
                layers.Conv1D(8,kernel_size=3,strides=1, padding='same', activation='elu'),
                layers.Flatten(),
                Dense(128, activation='elu'),
                Dense(64, activation='elu'),
                Dense(1, activation='linear')
            ])
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_absolute_percentage_error')


        elif self.f == 'schwefel':
            model = Sequential([
                layers.Lambda(lambda x: x / 1000, input_shape=(self.dims, 1)),
                layers.Conv1D(256,kernel_size=5,padding='same', activation='elu'),
                # layers.LayerNormalization(),
                layers.Conv1D(128,kernel_size=5, padding='same', activation='elu'),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64,kernel_size=5, padding='same', activation='elu'),
                layers.Conv1D(32,kernel_size=5, padding='same', activation='elu'),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(16,kernel_size=5, padding='same', activation='elu'),
                layers.Conv1D(8,kernel_size=5, padding='same', activation='elu'),
                layers.Conv1D(4,kernel_size=5, padding='same', activation='elu'),
                layers.Flatten(),
                Dense(128, activation='elu'),
                Dense(64, activation='elu'),
                Dense(32, activation='elu'),
                Dense(16, activation='elu'),
                Dense(8, activation='elu'),
                Dense(1, activation='linear')
            ])
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_absolute_percentage_error')

        elif self.f == 'michalewicz':
            model = Sequential()
            model.add(Lambda(lambda x: x / np.pi, input_shape=(self.dims, 1)))
            model.add(Conv1D(128, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(MaxPooling1D(pool_size=2, strides=1))
            model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', activation='elu'))
            model.add(Flatten())
            model.add(Dense(64,activation='elu'))
            model.add(Dense(1,activation='linear'))
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        else:
            model = Sequential()
            model.add(Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(self.dims, 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
            model.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
            model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2, strides=1))
            model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', activation='relu'))
            model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', activation='relu'))
            model.add(Conv1D(4, kernel_size=3, strides=1, padding='same', activation='relu'))
            model.add(Flatten())
            model.add(Dense(64,activation='relu'))
            model.add(Dense(1,activation='linear'))
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')


        mc = ModelCheckpoint("NN.h5", monitor='val_loss', mode='min', verbose=verbose, save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        model.fit(X_train.reshape(len(X_train),self.dims,1), y_train, batch_size=64, epochs=500, validation_data=(X_test.reshape(len(X_test),self.dims,1), y_test), callbacks=[early_stop,mc],verbose=verbose)

        model =  keras.models.load_model("NN.h5")
        y_pred = model.predict(X_test.reshape(len(X_test),self.dims,1))

        R=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
        R=np.asarray(R).round(3)
        MAE= metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
        MAPE= metrics.mean_absolute_percentage_error(y_test.reshape(-1), y_pred.reshape(-1))
        MAE=np.asarray(MAE).round(5)
        MAPE=np.asarray(MAPE).round(5)
        print("Model performance: R2",R**2,"MAE",MAE,"MAPE",MAPE)

        plt.figure()
        sns.set()
        sns.regplot(x=y_pred, y=y_test, color='k')
        plt.xlabel('Predicted value')
        plt.ylabel('Ground truth')

        return model
 