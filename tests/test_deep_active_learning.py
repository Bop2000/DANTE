import pytest
from dante.deep_active_learning import DeepActiveLearning
from dante.obj_functions import Ackley, Rastrigin
from dante.neural_surrogate import AckleySurrogateModel, RastriginSurrogateModel


class TestDeepActiveLearning:
    @pytest.fixture
    def dal_instance(self):
        func = Ackley(dims=3)
        surrogate = AckleySurrogateModel(input_dims=func.dims, epochs=1)
        return DeepActiveLearning(
            func=func,
            num_data_acquisition=30,
            surrogate=surrogate,
            num_init_samples=10,
            num_samples_per_acquisition=10,
        )

    def test_initialization(self, dal_instance: DeepActiveLearning):
        assert dal_instance.dims == 3
        assert dal_instance.num_data_acquisition == 30
        assert dal_instance.num_init_samples == 10
        assert dal_instance.num_samples_per_acquisition == 10
        assert dal_instance.input_x.shape == (10, 3)
        assert dal_instance.input_scaled_y.shape == (10,)

    @pytest.mark.slow
    def test_run_method(self, dal_instance: DeepActiveLearning):
        dal_instance.run()
        assert (
            dal_instance.input_x.shape[0] > 10
        )  # Should have more samples after running
        assert dal_instance.input_scaled_y.shape[0] > 10

    @pytest.mark.slow
    def test_different_objective_function(self):
        func = Rastrigin(dims=5)
        surrogate = RastriginSurrogateModel(input_dims=func.dims, epochs=1)
        dal = DeepActiveLearning(
            func=func,
            num_data_acquisition=80,
            surrogate=surrogate,
            num_init_samples=40,
            num_samples_per_acquisition=8,
        )
        assert dal.dims == 5
        dal.run()
        assert dal.input_x.shape[0] > 40
