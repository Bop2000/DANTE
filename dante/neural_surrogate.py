"""
This module provides classes for training neural network models for various synthetic functions.
It includes an abstract base class and specific implementations for different synthetic functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import gc
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import backend as K
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
    log_interval: int = 50
    plot_dir: Optional[Path] = None
    _model: keras.Model = None
    _fit_call_count: int = field(init=False, default=0)

    class _EpochProgressLogger(keras.callbacks.Callback):
        def __init__(
            self,
            total_epochs: int,
            log_every: int,
            monitor: str,
            checkpoint_path: Path,
        ) -> None:
            super().__init__()
            self.total_epochs = total_epochs
            self.log_every = max(1, log_every)
            self.monitor = monitor
            self.checkpoint_path = checkpoint_path
            self.best = np.inf
            self.last_logged_epoch = 0
            self.last_epoch = 0
            self.last_logs: dict[str, float] = {}
            self.last_improved = False
            self.last_previous_best = np.inf

        def on_epoch_end(self, epoch: int, logs: Optional[dict[str, float]] = None) -> None:
            logs = logs or {}
            current_epoch = epoch + 1
            current_monitor = logs.get(self.monitor)
            previous_best = self.best
            improved = False
            if current_monitor is not None and current_monitor < self.best:
                self.best = current_monitor
                improved = True

            self.last_epoch = current_epoch
            self.last_logs = logs.copy()
            self.last_improved = improved
            self.last_previous_best = previous_best

            if current_epoch % self.log_every == 0:
                self._print_progress(current_epoch, logs, improved, previous_best)

        def on_train_end(self, logs: Optional[dict[str, float]] = None) -> None:
            if self.last_epoch and self.last_epoch != self.last_logged_epoch:
                self._print_progress(
                    self.last_epoch,
                    self.last_logs,
                    self.last_improved,
                    self.last_previous_best,
                )

        def _print_progress(
            self,
            current_epoch: int,
            logs: dict[str, float],
            improved: bool,
            previous_best: float,
        ) -> None:
            self.last_logged_epoch = current_epoch
            loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            loss_str = f"{loss:.5f}" if loss is not None else "NA"
            val_loss_str = f"{val_loss:.5f}" if val_loss is not None else "NA"

            if improved and val_loss is not None:
                if np.isfinite(previous_best):
                    print(
                        f"Epoch {current_epoch}: {self.monitor} improved from "
                        f"{previous_best:.5f} to {val_loss_str}, saving model to "
                        f"{self.checkpoint_path}"
                    )
                else:
                    print(
                        f"Epoch {current_epoch}: {self.monitor} improved to "
                        f"{val_loss_str}, saving model to {self.checkpoint_path}"
                    )
            else:
                print(
                    f"Epoch {current_epoch}/{self.total_epochs} - "
                    f"loss: {loss_str} - {self.monitor}: {val_loss_str}"
                )

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
        # 清理上一轮可能残留的计算图/句柄，避免内存逐轮增长
        try:
            K.clear_session()
        except Exception:
            pass
        gc.collect()

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
            verbose=0,
            save_best_only=True,
        )
        early_stop = EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        callbacks_list = [early_stop, mc]
        if verbose:
            progress_logger = self._EpochProgressLogger(
                total_epochs=self.epochs,
                log_every=self.log_interval,
                monitor="val_loss",
                checkpoint_path=self.check_point_path,
            )
            callbacks_list.append(progress_logger)
        self.model.fit(
            x_train.reshape(len(x_train), self.input_dims, 1),
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_test.reshape(len(x_test), self.input_dims, 1), y_test),
            callbacks=callbacks_list,
            verbose=0,
        )

        self.model = keras.models.load_model(self.check_point_path)

        y_train_pred = self.model.predict(
            x_train.reshape(len(x_train), self.input_dims, 1), verbose=verbose
        ).reshape(-1)
        y_test_pred = self.model.predict(
            x_test.reshape(len(x_test), self.input_dims, 1), verbose=verbose
        ).reshape(-1)

        self._fit_call_count += 1
        self.evaluate_model(
            y_train=y_train,
            y_train_pred=y_train_pred,
            y_test=y_test,
            y_test_pred=y_test_pred,
            iteration=self._fit_call_count,
        )

        # 主动触发一次垃圾回收，降低长期运行的内存峰值
        gc.collect()
        return self.model

    def evaluate_model(
        self,
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        iteration: int,
    ) -> None:
        """
        Evaluate the model's performance and plot results.

        This method calculates various performance metrics and creates a regression plot.

        Args:
            y_train (np.ndarray): Training true values.
            y_train_pred (np.ndarray): Training predictions.
            y_test (np.ndarray): Test true values.
            y_test_pred (np.ndarray): Test predictions.
        """
        train_metrics = self._plot_predictions_vs_truth(
            truth=y_train.reshape(-1),
            predictions=y_train_pred.reshape(-1),
            iteration=iteration,
            title_suffix="(Train Set)",
            filename=(
                f"train_prediction_vs_truth_iter_{iteration:03d}.png"
                if self.plot_dir is not None
                else None
            ),
        )

        test_metrics = self._plot_predictions_vs_truth(
            truth=y_test.reshape(-1),
            predictions=y_test_pred.reshape(-1),
            iteration=iteration,
            title_suffix="(Test Set)",
            filename=(
                f"test_prediction_vs_truth_iter_{iteration:03d}.png"
                if self.plot_dir is not None
                else None
            ),
        )

        if train_metrics is not None:
            mse, rmse, mae, pearson = train_metrics
            print(
                f"Model train performance — MSE: {mse:.5f}, RMSE: {rmse:.5f}, MAE: {mae:.5f}, Pearson: {pearson:.4f}"
            )

        if test_metrics is not None:
            mse, rmse, mae, pearson = test_metrics
            r_squared = metrics.r2_score(y_test.reshape(-1), y_test_pred.reshape(-1))
            mape = metrics.mean_absolute_percentage_error(
                y_test.reshape(-1), y_test_pred.reshape(-1)
            )
            print(
                f"Model test performance — R2: {r_squared:.3f}, MSE: {mse:.5f}, RMSE: {rmse:.5f}, MAE: {mae:.5f}, MAPE: {mape:.5f}, Pearson: {pearson:.4f}"
            )

    def _plot_predictions_vs_truth(
        self,
        truth: np.ndarray,
        predictions: np.ndarray,
        iteration: int,
        title_suffix: str,
        filename: Optional[str] = None,
    ) -> Optional[tuple[float, float, float, float]]:
        if truth.size == 0 or predictions.size == 0:
            return None

        mse = float(np.mean((predictions - truth) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(predictions - truth)))

        if predictions.size >= 2 and truth.size >= 2:
            pearson = float(np.corrcoef(predictions, truth)[0, 1])
            if np.isnan(pearson):
                pearson = 0.0
        else:
            pearson = 0.0

        min_val = float(min(np.min(predictions), np.min(truth)))
        max_val = float(max(np.max(predictions), np.max(truth)))

        plt.figure(figsize=(10, 8))
        plt.scatter(truth, predictions, color="#1f77b4", alpha=0.7)
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal y=x line")

        metrics_text = (
            f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nPearson: {pearson:.4f}"
        )
        plt.annotate(
            metrics_text,
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            ha="left",
            va="top",
        )

        title = f"Predictions vs. Ground Truth (Iteration {iteration})"
        if title_suffix:
            title = f"{title} {title_suffix}"
        title = f"{title}\nMSE: {mse:.2f}, RMSE: {rmse:.2f}, Pearson: {pearson:.4f}"
        plt.title(title)
        plt.xlabel("True Values")
        plt.ylabel("Model Predictions")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if filename and self.plot_dir is not None:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.plot_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close()
        return mse, rmse, mae, pearson


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
    f: PredefinedSurrogateModel = None,
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
