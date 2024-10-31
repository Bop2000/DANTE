from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from dante.utils import Tracker


class BuiltInSyntheticFunction(Enum):
    ACKLEY = auto()
    RASTRIGIN = auto()
    ROSENBROCK = auto()
    GRIEWANK = auto()
    MICHALEWICZ = auto()
    SCHWEFEL = auto()
    LEVY = auto()


@dataclass
class ObjectiveFunction(ABC):
    dims: int
    turn: float
    lb: np.ndarray = field(init=False)
    ub: np.ndarray = field(init=False)
    name: str = "function"
    tracker: Tracker = field(init=False)
    counter: int = 0

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def scaled(self, y: float) -> float:
        pass

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        return x


@dataclass
class Ackley(ObjectiveFunction):
    dims: int = 3
    turn: float = 0.1
    name: str = "ackley"

    def __post_init__(self):
        self.lb = -5 * np.ones(self.dims)
        self.ub = 5 * np.ones(self.dims)
        self.tracker = Tracker(self.name + str(self.dims))

    def scaled(self, y: float) -> float:
        return 100 / (y + 0.01)

    def __call__(self, x: np.ndarray, apply_scaling: bool = False, track: bool = True) -> float:
        x = self._preprocess(x)
        y = float(
            -20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size))
            - np.exp(np.cos(2 * np.pi * x).sum() / x.size)
            + 20
            + np.e
        )
        if track:
            self.tracker.track(y, x)
        return y if not apply_scaling else self.scaled(y)


@dataclass
class Rastrigin(ObjectiveFunction):
    dims: int = 3
    turn: float = 0.1
    a: float = 10
    name: str = "rastrigin"

    def __post_init__(self):
        self.lb = -5 * np.ones(self.dims)
        self.ub = 5 * np.ones(self.dims)
        self.tracker = Tracker(self.name + str(self.dims))

    def scaled(self, y: float) -> float:
        return -1 * y

    def __call__(self, x: np.ndarray, apply_scaling: bool = False, track: bool = True) -> float:
        x = self._preprocess(x)
        n = len(x)
        sum_ = np.sum(x**2 - self.a * np.cos(2 * np.pi * x))
        y = float(self.a * n + sum_)
        if track:
            self.tracker.track(y, x)
        return y if not apply_scaling else self.scaled(y)


@dataclass
class Rosenbrock(ObjectiveFunction):
    dims: int = 3
    turn: float = 0.1
    name: str = "rosenbrock"

    def __post_init__(self):
        self.lb = -5 * np.ones(self.dims)
        self.ub = 5 * np.ones(self.dims)

    def scaled(self, y: float) -> float:
        return 100 / (y / (self.dims * 100) + 0.01)

    def __call__(self, x: np.ndarray, apply_scaling: bool = False, track: bool = True) -> float:
        x = self._preprocess(x)
        y = float(
            np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
        )
        if track:
            self.tracker.track(y, x)
        return y if not apply_scaling else self.scaled(y)

@dataclass
class Griewank(ObjectiveFunction):
    dims: int = 3
    turn: float = 1
    name: str = "griewank"

    def __post_init__(self):
        self.lb = -600 * np.ones(self.dims)
        self.ub = 600 * np.ones(self.dims)
        self.tracker = Tracker(self.name + str(self.dims))

    def scaled(self, y: float) -> float:
        return 10 / (y / self.dims + 0.001)

    def __call__(self, x: np.ndarray, apply_scaling: bool = False, track: bool = True) -> float:
        x = self._preprocess(x)
        sum_term = np.sum(x**2)
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        y = float(1 + sum_term / 4000 - prod_term)
        if track:
            self.tracker.track(y, x)
        return y if not apply_scaling else self.scaled(y)


@dataclass
class Michalewicz(ObjectiveFunction):
    dims: int = 3
    turn: float = 0.01
    name: str = "michalewicz"

    def __post_init__(self):
        self.lb = 0 * np.ones(self.dims)
        self.ub = np.pi * np.ones(self.dims)
        self.tracker = Tracker(self.name + str(self.dims))

    def scaled(self, y: float) -> float:
        return -1 * y

    def __call__(self, x: np.ndarray, apply_scaling:bool=False, m:float=10, track: bool = True) -> float:
        x = self._preprocess(x)
        d = len(x)
        y = 0
        for i in range(d):
            y += np.sin(x[i]) * np.sin((i + 1) * x[i] ** 2 / np.pi) ** (2 * m)
        if track:
            self.tracker.track(y, x)
        return float(-1 * y) if not apply_scaling else self.scaled(float(-1 * y))


@dataclass
class Schwefel(ObjectiveFunction):
    dims: int = 3
    turn: float = 1
    name: str = "schwefel"

    def __post_init__(self):
        self.lb = -500 * np.ones(self.dims)
        self.ub = 500 * np.ones(self.dims)
        self.tracker = Tracker(self.name + str(self.dims))

    def scaled(self, y: float):
        if y == 0.0:
            return 10000.0
        return -1 * y / 100

    def __call__(self, x: np.ndarray, apply_scaling: bool = False, track: bool = True) -> float:
        x = self._preprocess(x)
        dimension = len(x)
        sum_part = np.sum(-x * np.sin(np.sqrt(np.abs(x))))
        if np.all(np.array(x) == 421, axis=0):
            return 0.0
        y = float(418.9829 * dimension + sum_part)
        if track:
            self.tracker.track(y, x)
        return y if not apply_scaling else self.scaled(y)


@dataclass
class Levy(ObjectiveFunction):
    dims: int = 1
    turn: float = 0.1
    round: int = 1
    name: str = "levy"

    def __post_init__(self):
        self.lb = -10 * np.ones(self.dims)
        self.ub = 10 * np.ones(self.dims)
        self.tracker = Tracker(self.name + str(self.dims))

    def scaled(self, y: float) -> float:
        return -1 * y

    def __call__(self, x: np.ndarray, apply_scaling: bool = False, track: bool = True) -> float:
        x = self._preprocess(x)
        w = []
        for idx in range(0, len(x)):
            w.append(1 + (x[idx] - 1) / 4)
        w = np.array(w)

        term1 = (np.sin(np.pi * w[0])) ** 2

        term3 = (w[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[-1])) ** 2)

        term2 = 0
        for idx in range(1, len(w)):
            wi = w[idx]
            new = (wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2)
            term2 = term2 + new

        y = float(term1 + term2 + term3)
        if track:
            self.tracker.track(y, x)
        return y if not apply_scaling else self.scaled(y)

