import numpy as np
import pytest
from tensorflow import keras

from dante.neural_surrogate import (
    SurrogateModel,
    AckleySurrogateModel,
    RastriginSurrogateModel,
    RosenbrockSurrogateModel,
    GriewankSurrogateModel,
    LevySurrogateModel,
    SchwefelSurrogateModel,
    MichalewiczSurrogateModel,
    DefaultSurrogateModel,
    PredefinedSurrogateModel,
    get_surrogate_model,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    x = np.random.rand(100, 10)
    y = np.sum(x ** 2, axis=1)
    return x, y


def test_surrogate_model_abstract():
    with pytest.raises(TypeError):
        SurrogateModel()


@pytest.mark.parametrize(
    "model_class",
    [
        AckleySurrogateModel,
        RastriginSurrogateModel,
        RosenbrockSurrogateModel,
        GriewankSurrogateModel,
        LevySurrogateModel,
        SchwefelSurrogateModel,
        MichalewiczSurrogateModel,
        DefaultSurrogateModel,
    ],
)
def test_surrogate_model_creation(model_class: SurrogateModel):
    model = model_class(input_dims=10, learning_rate=0.001, epochs=1)
    assert isinstance(model, SurrogateModel)
    assert model.input_dims == 10
    assert model.learning_rate == 0.001
    assert model.epochs == 1


@pytest.mark.parametrize(
    "model_class",
    [
        AckleySurrogateModel,
        RastriginSurrogateModel,
        RosenbrockSurrogateModel,
        GriewankSurrogateModel,
        LevySurrogateModel,
        SchwefelSurrogateModel,
        MichalewiczSurrogateModel,
        DefaultSurrogateModel,
    ],
)
def test_create_model(model_class):
    model = model_class(input_dims=10, epochs=1)
    keras_model = model.create_model()
    assert isinstance(keras_model, keras.Model)
    assert keras_model.input_shape == (None, 10, 1)
    assert keras_model.output_shape == (None, 1)


def test_surrogate_model_call(sample_data):
    x, y = sample_data
    model = DefaultSurrogateModel(input_dims=10, epochs=1)
    trained_model = model(x, y, verbose=False)
    assert isinstance(trained_model, keras.Model)


def test_evaluate_model(sample_data, capsys):
    x, y = sample_data
    model = DefaultSurrogateModel(input_dims=10)
    trained_model = model(x, y, verbose=False)
    y_pred = trained_model.predict(x.reshape(len(x), 10, 1))
    model.evaluate_model(y, y_pred)
    captured = capsys.readouterr()
    assert "Model performance:" in captured.out


@pytest.mark.parametrize(
    "model_type,expected_class",
    [
        (PredefinedSurrogateModel.ACKLEY, AckleySurrogateModel),
        (PredefinedSurrogateModel.RASTRIGIN, RastriginSurrogateModel),
        (PredefinedSurrogateModel.ROSENBROCK, RosenbrockSurrogateModel),
        (PredefinedSurrogateModel.GRIEWANK, GriewankSurrogateModel),
        (PredefinedSurrogateModel.LEVY, LevySurrogateModel),
        (PredefinedSurrogateModel.SCHWEFEL, SchwefelSurrogateModel),
        (PredefinedSurrogateModel.MICHALEWICZ, MichalewiczSurrogateModel),
        (PredefinedSurrogateModel.DEFAULT, DefaultSurrogateModel),
    ],
)
def test_get_surrogate_model(model_type, expected_class):
    assert get_surrogate_model(model_type) == expected_class


def test_get_surrogate_model_default():
    assert get_surrogate_model("UNKNOWN_MODEL") == DefaultSurrogateModel
