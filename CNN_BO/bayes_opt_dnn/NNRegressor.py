import numpy as np
from scipy.optimize import NonlinearConstraint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from typing import Any, Mapping, Callable, Optional, Union, Tuple
import random
from scipy import stats
from sklearn import metrics

from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    Lambda,
)
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor


class CNNRegressor:
    """CNN surrogate model with Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(
        self,
        input_dim: int,
        dropout_rate: float = 0.2,
        n_forward_passes: int = 5,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        random_state: Optional[int] = None
    ):
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.n_forward_passes = n_forward_passes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.models = []
        for _ in range(n_forward_passes):
            self.models.append(self._build_model())
        
        if random_state is not None:
            tf.random.set_seed(random_state)
            np.random.seed(random_state)
    
    def _build_model(self) -> tf.keras.Model:
        """Construct the CNN architecture with MC Dropout layers."""
        model = Sequential([
            Conv1D(16, kernel_size=3, strides=1, padding="same", 
                   activation="relu", input_shape=(self.input_dim, 1)),
            MaxPooling1D(pool_size=2),
            Dropout(self.dropout_rate),
            Conv1D(8, kernel_size=3, strides=1, padding="same", activation="elu"),
            Conv1D(4, kernel_size=3, strides=1, padding="same", activation="elu"),
            Flatten(),
            Dense(32, activation="elu"),
            Dense(1, activation="linear")
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        print('model built!')
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the CNN model with MC Dropout enabled."""
        # Reshape input to 3D: (samples, features, channels)
        X_reshaped = X.reshape(-1, self.input_dim, 1)
        index_random = np.arange(X_reshaped.shape[0])
        random.shuffle(index_random)
        
        for i in range(len(self.models)):
            model = self.models[i]
            # slice the data to 'n_model' parts
            ind=index_random[
                round(i*len(index_random)/self.n_forward_passes):
                    round((1+i)*len(index_random)/self.n_forward_passes)]
            ind2=np.setdiff1d(index_random, ind)
            x_train,x_test,y_train,y_test = X_reshaped[ind2],X_reshaped[ind],y[ind2],y[ind]
            # print(X_reshaped.shape)
            model.fit(x_train,y_train, 
                      epochs=self.epochs, 
                      batch_size=self.batch_size,
                      validation_data=(x_test, y_test),
                      verbose=0)
            
            y_pred = model.predict(x_test, verbose=0)
            self.evaluate_model(y_test, y_pred)

    
    def evaluate_model(self, y_test, y_pred):
        """
        Evaluate the model's performance and plot results.

        This method calculates various performance metrics and creates a regression plot.

        Args:
            y_test (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
        """
        r = stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
        r = np.asarray(r).round(3)
        r_squared = metrics.r2_score(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
        mape = metrics.mean_absolute_percentage_error(
            y_test.reshape(-1), y_pred.reshape(-1)
        )
        mae = np.asarray(mae).round(5)
        mape = np.asarray(mape).round(5)
        print(
            f"Model performance: R2 {r_squared:.3f}, MAE {mae:.5f}, MAPE {mape:.5f}"
        )

    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with uncertainty estimation using MC Dropout."""
        X_reshaped = X.reshape(-1, self.input_dim, 1)
        predictions = []
        
        # Enable dropout at test time for uncertainty estimation
        for model in self.models:
            preds = model(X_reshaped, training=True).numpy().flatten()
            predictions.append(preds)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        # print(mean,std)
        return (mean, std) if return_std else mean

