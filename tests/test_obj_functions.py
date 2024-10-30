import numpy as np
import pytest

from dante.obj_functions import Ackley, Rastrigin, Rosenbrock, Griewank, Michalewicz, Schwefel


class TestFunctions:
    @pytest.mark.parametrize(
        argnames=("dims", "x", "expected", "scaled_expected"),
        argvalues=[
            (3, [1, 2, 3], 7.016453, 14.231930),
            (2, [2, 2], 6.593599, 15.143257),
        ]
    )
    def test_ackley(self, dims: int, x: list[float], expected: float, scaled_expected: float):
        ackley = Ackley(dims=dims)
        y = ackley(np.array(x))
        assert y == pytest.approx(expected)
        assert ackley.scaled(y) == pytest.approx(scaled_expected)

    @pytest.mark.parametrize(
        argnames=("dims", "x", "expected", "scaled_expected"),
        argvalues=[
            (3, [1, 2, 3], 14.0, -14.0),
            (2, [2, 2], 8.0, -8.0),
        ]
    )
    def test_rastrigin(self, dims: int, x: list[float], expected: float, scaled_expected: float):
        rastrigin = Rastrigin(dims=dims)
        y = rastrigin(np.array(x))
        assert y == pytest.approx(expected)
        assert rastrigin.scaled(y) == pytest.approx(scaled_expected)

    @pytest.mark.parametrize(
        argnames=("dims", "x", "expected", "scaled_expected"),
        argvalues=[
            (3, [1, 2, 3], 201.0, 147.058823),
            (2, [2, 2], 401.0, 49.627791),
        ]
    )
    def test_rosenbrock(self, dims: int, x: list[float], expected: float, scaled_expected: float):
        rosenbrock = Rosenbrock(dims=dims)
        y = rosenbrock(np.array(x))
        assert y == pytest.approx(expected)
        assert rosenbrock.scaled(y) == pytest.approx(scaled_expected)

    @pytest.mark.parametrize(
        argnames=("dims", "x", "expected", "scaled_expected"),
        argvalues=[
            (3, [1, 2, 3], 1.017027, 29.410958),
            (2, [2, 2], 1.066895, 18.710903),
        ]
    )
    def test_griewank(self, dims: int, x: list[float], expected: float, scaled_expected: float):
        griewank = Griewank(dims=dims)
        y = griewank(np.array(x))
        assert y == pytest.approx(expected)
        assert griewank.scaled(y) == pytest.approx(scaled_expected)

    @pytest.mark.parametrize(
        argnames=("dims", "x", "expected", "scaled_expected"),
        argvalues=[
            (3, [1, 2, 3], -0.00033451267210618074, 0.00033451267210618074),
            (2, [2, 2], -0.37015149, 0.37015149),
        ]
    )
    def test_michalewicz(self, dims: int, x: list[float], expected: float, scaled_expected: float):
        michalewicz = Michalewicz(dims=dims)
        y = michalewicz(np.array(x))
        assert y == pytest.approx(expected)
        assert michalewicz.scaled(y) == pytest.approx(scaled_expected)

    @pytest.mark.parametrize(
        argnames=("dims", "x", "expected", "scaled_expected"),
        argvalues=[
            (3, [1, 2, 3], 1251.170617, -12.511706),
            (2, [2, 2], 834.014736, -8.340147),
        ]
    )
    def test_schwefel(self, dims: int, x: list[float], expected: float, scaled_expected: float):
        schwefel = Schwefel(dims=dims)
        y = schwefel(np.array(x))
        assert y == pytest.approx(expected)
        assert schwefel.scaled(y) == pytest.approx(scaled_expected)
