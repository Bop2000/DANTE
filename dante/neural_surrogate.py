"""
This module provides classes for training neural network models for various synthetic functions.
It includes an abstract base class and specific implementations for different synthetic functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    Lambda,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


@dataclass
class SurrogateModel(ABC):
    """
    Abstract base class for surrogate model implementations.

    Attributes:
        input_dims (int): The input dimensions for the model.
        test_size (float): The proportion of the dataset to include in the test split.
        train_test_split_random_state (int): Random state for reproducible train-test splits.
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The number of samples per gradient update.
        epochs (int): The number of epochs to train the model.
        patience (int): Number of epochs with no improvement after which training will be stopped.
    """

    input_dims: int = 10
    learning_rate: float = 0.001
    check_point_path: Path = field(default_factory=lambda: Path("NN.keras"))
    test_size: float = 0.2
    train_test_split_random_state: int = 42
    batch_size: int = 64
    epochs: int = 500
    patience: int = 30
    _model: keras.Model | None = None

    @abstractmethod
    def create_model(self) -> keras.Model:
        """
        Create and return a Keras model.

        This method should be implemented by subclasses to define the specific
        architecture of the neural network model.

        Returns:
            keras.Model: The created Keras model.
        """
        pass

    def __call__(self, x, y, verbose=0):
        """
        Train the model on the given data.

        This method handles the entire training process, including data splitting,
        model creation, training, and evaluation.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            verbose (bool): If True, print detailed output during training. Defaults to False.

        Returns:
            keras.Model: The trained Keras model.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=self.train_test_split_random_state,
        )

        self.model = self.create_model()

        mc = ModelCheckpoint(
            self.check_point_path,
            monitor="val_loss",
            mode="min",
            verbose=verbose,
            save_best_only=True,
        )
        early_stop = EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        self.model.fit(
            x_train.reshape(len(x_train), self.input_dims, 1),
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_test.reshape(len(x_test), self.input_dims, 1), y_test),
            callbacks=[early_stop, mc],
            verbose=verbose,
        )

        self.model = keras.models.load_model(self.check_point_path)
        y_pred = self.model.predict(
            x_test.reshape(len(x_test), self.input_dims, 1), verbose=verbose
        )

        self.evaluate_model(y_test, y_pred)

        return self.model

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

        plt.figure()
        sns.regplot(x=y_pred, y=y_test, color="k")
        plt.xlabel("Predicted value")
        plt.ylabel("Ground truth")


class AckleySurrogateModel(SurrogateModel):
    """
    Surrogate model implementation for the Ackley function.
    """

    def create_model(self) -> keras.Model:
        if self.input_dims <= 100:
            model = Sequential(
                [
                    Conv1D(
                        128,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        activation="elu",
                        input_shape=(self.input_dims, 1),
                    ),
                    MaxPooling1D(pool_size=2, strides=1),
                    Dropout(0.2),
                    Conv1D(
                        64, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    MaxPooling1D(pool_size=2, strides=1),
                    Dropout(0.2),
                    Conv1D(
                        32, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    Conv1D(
                        16, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    Conv1D(
                        8, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    Flatten(),
                    Dense(128, activation="elu"),
                    Dense(64, activation="elu"),
                    Dense(1, activation="linear"),
                ]
            )
        else:
            model = Sequential(
                [
                    Conv1D(
                        128,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        activation="elu",
                        input_shape=(self.input_dims, 1),
                    ),
                    MaxPooling1D(pool_size=2),
                    Dropout(0.2),
                    Conv1D(
                        64, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    MaxPooling1D(pool_size=2),
                    Dropout(0.2),
                    Conv1D(
                        32, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    MaxPooling1D(pool_size=2, strides=1),
                    Conv1D(
                        16, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    Conv1D(
                        8, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    Conv1D(
                        4, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    Flatten(),
                    Dense(64, activation="elu"),
                    Dense(1, activation="linear"),
                ]
            )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error"
        )
        return model


class RastriginSurrogateModel(SurrogateModel):
    """
    Surrogate model implementation for the Rastrigin function.
    """

    def create_model(self) -> keras.Model:
        model = Sequential(
            [
                layers.Conv1D(
                    256,
                    kernel_size=5,
                    strides=1,
                    padding="same",
                    activation="elu",
                    input_shape=(self.input_dims, 1),
                ),
                layers.LayerNormalization(),
                layers.Conv1D(
                    128, kernel_size=5, strides=2, padding="same", activation="elu"
                ),
                layers.Conv1D(
                    64, kernel_size=3, strides=2, padding="same", activation="elu"
                ),
                layers.Conv1D(
                    32, kernel_size=3, strides=1, padding="same", activation="elu"
                ),
                layers.Conv1D(
                    16, kernel_size=3, strides=1, padding="same", activation="elu"
                ),
                layers.Conv1D(
                    8, kernel_size=3, strides=1, padding="same", activation="elu"
                ),
                layers.Flatten(),
                Dense(128, activation="elu"),
                Dense(64, activation="elu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mean_absolute_percentage_error",
        )
        return model


class RosenbrockSurrogateModel(SurrogateModel):
    """
    Surrogate model implementation for the Rosenbrock function.
    """

    def create_model(self) -> keras.Model:
        model = Sequential(
            [
                Conv1D(
                    128,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation="elu",
                    input_shape=(self.input_dims, 1),
                ),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(64, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(32, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2, strides=1),
                Conv1D(16, kernel_size=3, strides=1, padding="same", activation="elu"),
                Conv1D(8, kernel_size=3, strides=1, padding="same", activation="elu"),
                Conv1D(4, kernel_size=3, strides=1, padding="same", activation="elu"),
                Flatten(),
                Dense(64, activation="elu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error"
        )
        return model


class GriewankSurrogateModel(SurrogateModel):
    """
    Surrogate model training implementation for the Griewank function.
    """

    def create_model(self) -> keras.Model:
        model = Sequential(
            [
                Lambda(lambda x: x / 600, input_shape=(self.input_dims, 1)),
                Conv1D(128, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(64, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(32, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2, strides=1),
                Conv1D(16, kernel_size=3, strides=1, padding="same", activation="elu"),
                Conv1D(8, kernel_size=3, strides=1, padding="same", activation="elu"),
                Conv1D(4, kernel_size=3, strides=1, padding="same", activation="elu"),
                Flatten(),
                Dense(64, activation="elu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error"
        )
        return model


class LevySurrogateModel(SurrogateModel):
    """
    Surrogate model implementation for the Levy function.
    """

    def create_model(self) -> keras.Model:
        model = Sequential(
            [
                layers.Lambda(lambda x: x / 10, input_shape=(self.input_dims, 1)),
                layers.Conv1D(
                    256, kernel_size=5, strides=1, padding="same", activation="elu"
                ),
                layers.Conv1D(
                    128, kernel_size=5, strides=2, padding="same", activation="elu"
                ),
                layers.Conv1D(
                    64, kernel_size=3, strides=2, padding="same", activation="elu"
                ),
                layers.Conv1D(
                    32, kernel_size=3, strides=1, padding="same", activation="elu"
                ),
                layers.Conv1D(
                    16, kernel_size=3, strides=1, padding="same", activation="elu"
                ),
                layers.Conv1D(
                    8, kernel_size=3, strides=1, padding="same", activation="elu"
                ),
                layers.Flatten(),
                Dense(128, activation="elu"),
                Dense(64, activation="elu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mean_absolute_percentage_error",
        )
        return model


class SchwefelSurrogateModel(SurrogateModel):
    """
    Surrogate model implementation for the Schwefel function.
    """

    def create_model(self) -> keras.Model:
        model = Sequential(
            [
                layers.Lambda(lambda x: x / 1000, input_shape=(self.input_dims, 1)),
                layers.Conv1D(256, kernel_size=5, padding="same", activation="elu"),
                layers.Conv1D(128, kernel_size=5, padding="same", activation="elu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=5, padding="same", activation="elu"),
                layers.Conv1D(32, kernel_size=5, padding="same", activation="elu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(16, kernel_size=5, padding="same", activation="elu"),
                layers.Conv1D(8, kernel_size=5, padding="same", activation="elu"),
                layers.Conv1D(4, kernel_size=5, padding="same", activation="elu"),
                layers.Flatten(),
                Dense(128, activation="elu"),
                Dense(64, activation="elu"),
                Dense(32, activation="elu"),
                Dense(16, activation="elu"),
                Dense(8, activation="elu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mean_absolute_percentage_error",
        )
        return model


class MichalewiczSurrogateModel(SurrogateModel):
    """
    Surrogate model implementation for the Michalewicz function.
    """

    def create_model(self) -> keras.Model:
        model = Sequential(
            [
                Lambda(lambda x: x / np.pi, input_shape=(self.input_dims, 1)),
                Conv1D(128, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2),
                Conv1D(64, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2),
                Conv1D(32, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2, strides=1),
                Conv1D(16, kernel_size=3, strides=1, padding="same", activation="elu"),
                Conv1D(8, kernel_size=3, strides=1, padding="same", activation="elu"),
                Flatten(),
                Dense(64, activation="elu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error"
        )
        return model


class DefaultSurrogateModel(SurrogateModel):
    """
    Default surrogate model implementation.
    """

    def create_model(self) -> keras.Model:
        model = Sequential(
            [
                Conv1D(
                    128,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation="relu",
                    input_shape=(self.input_dims, 1),
                ),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu"),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(32, kernel_size=3, strides=1, padding="same", activation="relu"),
                MaxPooling1D(pool_size=2, strides=1),
                Conv1D(16, kernel_size=3, strides=1, padding="same", activation="relu"),
                Conv1D(8, kernel_size=3, strides=1, padding="same", activation="relu"),
                Conv1D(4, kernel_size=3, strides=1, padding="same", activation="relu"),
                Flatten(),
                Dense(64, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error"
        )
        return model


class PredefinedSurrogateModel(Enum):
    ACKLEY = auto()
    RASTRIGIN = auto()
    ROSENBROCK = auto()
    GRIEWANK = auto()
    LEVY = auto()
    SCHWEFEL = auto()
    MICHALEWICZ = auto()
    DEFAULT = auto()


def get_surrogate_model(
    f: PredefinedSurrogateModel | str | None = None,
) -> SurrogateModel:
    """
    Factory function to get the appropriate SurrogateModel.

    Args:
        f (str): The name of the optimization function.

    Returns:
        SurrogateModel: An instance of the appropriate SurrogateModel subclass.
    """
    model_classes = {
        PredefinedSurrogateModel.ACKLEY: AckleySurrogateModel,
        PredefinedSurrogateModel.RASTRIGIN: RastriginSurrogateModel,
        PredefinedSurrogateModel.ROSENBROCK: RosenbrockSurrogateModel,
        PredefinedSurrogateModel.GRIEWANK: GriewankSurrogateModel,
        PredefinedSurrogateModel.LEVY: LevySurrogateModel,
        PredefinedSurrogateModel.SCHWEFEL: SchwefelSurrogateModel,
        PredefinedSurrogateModel.MICHALEWICZ: MichalewiczSurrogateModel,
    }
    return model_classes.get(f, DefaultSurrogateModel)
