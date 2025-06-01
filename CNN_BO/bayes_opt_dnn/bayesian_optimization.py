"""Main module.

Holds the `BayesianOptimization` class, which handles the maximization of a
function over a specific target space.
"""

from __future__ import annotations

import json
from collections import deque
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np
from scipy.optimize import NonlinearConstraint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from typing import Any, Mapping, Callable, Optional, Union, Tuple

from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    Lambda,
)

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor




from bayes_opt_dnn import acquisition
from bayes_opt_dnn.NNRegressor import CNNRegressor
from bayes_opt_dnn.constraint import ConstraintModel
from bayes_opt_dnn.domain_reduction import DomainTransformer
from bayes_opt_dnn.logger import ScreenLogger
from bayes_opt_dnn.parameter import wrap_kernel
from bayes_opt_dnn.target_space import TargetSpace
from bayes_opt_dnn.util import ensure_rng

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.random import RandomState
    from numpy.typing import NDArray

    from bayes_opt_dnn.acquisition import AcquisitionFunction
    from bayes_opt_dnn.constraint import ConstraintModel
    from bayes_opt_dnn.domain_reduction import DomainTransformer
    from bayes_opt_dnn.parameter import BoundsMapping, ParamsType

    Float = np.floating[Any]


class BayesianOptimization:
    """Handle optimization of a target function over a specific target space.

    This class takes the function to optimize as well as the parameters bounds
    in order to find which values for the parameters yield the maximum value
    using bayesian optimization.

    Parameters
    ----------
    f: function or None.
        Function to be maximized.

    pbounds: dict
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values.

    constraint: NonlinearConstraint.
        Note that the names of arguments of the constraint function and of
        f need to be the same.

    random_state: int or numpy.random.RandomState, optional(default=None)
        If the value is an integer, it is used as the seed for creating a
        numpy.random.RandomState. Otherwise the random state provided is used.
        When set to None, an unseeded random state is generated.

    verbose: int, optional(default=2)
        The level of verbosity.

    bounds_transformer: DomainTransformer, optional(default=None)
        If provided, the transformation is applied to the bounds.

    allow_duplicate_points: bool, optional (default=False)
        If True, the optimizer will allow duplicate points to be registered.
        This behavior may be desired in high noise situations where repeatedly probing
        the same point will give different answers. In other situations, the acquisition
        may occasionally generate a duplicate point.
    """

    def __init__(
        self,
        f: Callable[..., float] | None,
        pbounds: Mapping[str, tuple[float, float]],
        acquisition_function: AcquisitionFunction | None = None,
        constraint: NonlinearConstraint | None = None,
        random_state: int | RandomState | None = None,
        verbose: int = 2,
        bounds_transformer: DomainTransformer | None = None,
        allow_duplicate_points: bool = False,
        cnn_params: Optional[Mapping[str, Any]] = None
    ):
        self._random_state = ensure_rng(random_state)
        self._allow_duplicate_points = allow_duplicate_points
        self._queue: deque[ParamsType] = deque()

        if acquisition_function is None:
            if constraint is None:
                self._acquisition_function = acquisition.UpperConfidenceBound(
                    kappa=2.576, random_state=self._random_state
                )
            else:
                self._acquisition_function = acquisition.ExpectedImprovement(
                    xi=0.01, random_state=self._random_state
                )
        else:
            self._acquisition_function = acquisition_function

        if constraint is None:
            # Data structure containing the function to be optimized, the
            # bounds of its domain, and a record of the evaluations we have
            # done so far
            self._space = TargetSpace(
                f, pbounds, random_state=random_state, allow_duplicate_points=self._allow_duplicate_points
            )
            self.is_constrained = False
        else:
            constraint_ = ConstraintModel(
                constraint.fun, constraint.lb, constraint.ub, random_state=random_state
            )
            self._space = TargetSpace(
                f,
                pbounds,
                constraint=constraint_,
                random_state=random_state,
                allow_duplicate_points=self._allow_duplicate_points,
            )
            self.is_constrained = True

        # Replace GP with CNN surrogate
        input_dim = len(pbounds)
        default_cnn_params = {
            'input_dim': input_dim,
            'dropout_rate': 0.2,
            'n_forward_passes': 5,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'random_state': random_state
        }
        # print(default_cnn_params)
        if cnn_params:
            default_cnn_params.update(cnn_params)
            
        self._gp = CNNRegressor(**default_cnn_params)

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            if not isinstance(self._bounds_transformer, DomainTransformer):
                msg = "The transformer must be an instance of DomainTransformer"
                raise TypeError(msg)
            self._bounds_transformer.initialize(self._space)

        self._sorting_warning_already_shown = False  # TODO: remove in future version

        # Initialize logger
        self.logger = ScreenLogger(verbose=self._verbose, is_constrained=self.is_constrained)

    @property
    def space(self) -> TargetSpace:
        """Return the target space associated with the optimizer."""
        return self._space

    @property
    def acquisition_function(self) -> AcquisitionFunction:
        """Return the acquisition function associated with the optimizer."""
        return self._acquisition_function

    @property
    def constraint(self) -> ConstraintModel | None:
        """Return the constraint associated with the optimizer, if any."""
        if self.is_constrained:
            return self._space.constraint
        return None

    @property
    def max(self) -> dict[str, Any] | None:
        """Get maximum target value found and corresponding parameters.

        See `TargetSpace.max` for more information.
        """
        return self._space.max()

    @property
    def res(self) -> list[dict[str, Any]]:
        """Get all target values and constraint fulfillment for all parameters.

        See `TargetSpace.res` for more information.
        """
        return self._space.res()

    def register(
        self, params: ParamsType, target: float, constraint_value: float | NDArray[Float] | None = None
    ) -> None:
        """Register an observation with known target.

        Parameters
        ----------
        params: dict or list
            The parameters associated with the observation.

        target: float
            Value of the target function at the observation.

        constraint_value: float or None
            Value of the constraint function at the observation, if any.
        """
        # TODO: remove in future version
        if isinstance(params, np.ndarray) and not self._sorting_warning_already_shown:
            msg = (
                "You're attempting to register an np.ndarray. In previous versions, the optimizer internally"
                " sorted parameters by key and expected any registered array to respect this order."
                " In the current and any future version the order as given by the pbounds dictionary will be"
                " used. If you wish to retain sorted parameters, please manually sort your pbounds"
                " dictionary before constructing the optimizer."
            )
            warn(msg, stacklevel=1)
            self._sorting_warning_already_shown = True
        self._space.register(params, target, constraint_value)
        self.logger.log_optimization_step(
            self._space.keys, self._space.res()[-1], self._space.params_config, self.max
        )

    def probe(self, params: ParamsType, lazy: bool = True) -> None:
        """Evaluate the function at the given points.

        Useful to guide the optimizer.

        Parameters
        ----------
        params: dict or list
            The parameters where the optimizer will evaluate the function.

        lazy: bool, optional(default=True)
            If True, the optimizer will evaluate the points when calling
            maximize(). Otherwise it will evaluate it at the moment.
        """
        # TODO: remove in future version
        if isinstance(params, np.ndarray) and not self._sorting_warning_already_shown:
            msg = (
                "You're attempting to register an np.ndarray. In previous versions, the optimizer internally"
                " sorted parameters by key and expected any registered array to respect this order."
                " In the current and any future version the order as given by the pbounds dictionary will be"
                " used. If you wish to retain sorted parameters, please manually sort your pbounds"
                " dictionary before constructing the optimizer."
            )
            warn(msg, stacklevel=1)
            self._sorting_warning_already_shown = True
            params = self._space.array_to_params(params)
        if lazy:
            self._queue.append(params)
        else:
            self._space.probe(params)
            self.logger.log_optimization_step(
                self._space.keys, self._space.res()[-1], self._space.params_config, self.max
            )

    def suggest(self) -> dict[str, float | NDArray[Float]]:
        """Suggest a promising point to probe next."""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample(random_state=self._random_state))

        # Finding argmax of the acquisition function.
        suggestion = self._acquisition_function.suggest(gp=self._gp, target_space=self._space, fit_gp=True)

        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points: int) -> None:
        """Ensure the queue is not empty.

        Parameters
        ----------
        init_points: int
            Number of parameters to prime the queue with.
        """
        if not self._queue and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            sample = self._space.random_sample(random_state=self._random_state)
            self._queue.append(self._space.array_to_params(sample))


    def maximize(self, init_points: int = 5, n_iter: int = 25) -> None:
        r"""
        Maximize the given function over the target space.

        Parameters
        ----------
        init_points : int, optional(default=5)
            Number of random points to probe before starting the optimization.

        n_iter: int, optional(default=25)
            Number of iterations where the method attempts to find the maximum
            value.

        Warning
        -------
            The maximize loop only fits the GP when suggesting a new point to
            probe based on the acquisition function. This means that the GP may
            not be fitted on all points registered to the target space when the
            method completes. If you intend to use the GP model after the
            optimization routine, make sure to fit it manually, e.g. by calling
            ``optimizer._gp.fit(optimizer.space.params, optimizer.space.target)``.
        """
        # Log optimization start
        self.logger.log_optimization_start(self._space.keys)

        # Prime the queue with random points
        self._prime_queue(init_points)
        iteration = 0
        while self._queue or iteration < n_iter:
            try:
                x_probe = self._queue.popleft()
                # print('yes')
            except IndexError:
                x_probe = self.suggest()
                iteration += 1
                print('finish')
            self.probe(x_probe, lazy=False)

            if self._bounds_transformer and iteration > 0:
                # The bounds transformer should only modify the bounds after
                # the init_points points (only for the true iterations)
                self.set_bounds(self._bounds_transformer.transform(self._space))

        # Log optimization end
        self.logger.log_optimization_end()

    def set_bounds(self, new_bounds: BoundsMapping) -> None:
        """Modify the bounds of the search space.

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def save_state(self, path: str | PathLike[str]) -> None:
        """Save complete state for reconstruction of the optimizer.

        Parameters
        ----------
        path : str or PathLike
            Path to save the optimization state

        Raises
        ------
        ValueError
            If attempting to save state before collecting any samples.
        """
        if len(self._space) == 0:
            msg = (
                "Cannot save optimizer state before collecting any samples. "
                "Please probe or register at least one point before saving."
            )
            raise ValueError(msg)

        random_state = None
        if self._random_state is not None:
            state_tuple = self._random_state.get_state()
            random_state = {
                "bit_generator": state_tuple[0],
                "state": state_tuple[1].tolist(),
                "pos": state_tuple[2],
                "has_gauss": state_tuple[3],
                "cached_gaussian": state_tuple[4],
            }

        # Get constraint values if they exist
        constraint_values = self._space._constraint_values.tolist() if self.is_constrained else None
        acquisition_params = self._acquisition_function.get_acquisition_params()
        state = {
            "pbounds": {key: self._space._bounds[i].tolist() for i, key in enumerate(self._space.keys)},
            # Add current transformed bounds if using bounds transformer
            "transformed_bounds": (self._space.bounds.tolist() if self._bounds_transformer else None),
            "keys": self._space.keys,
            "params": np.array(self._space.params).tolist(),
            "target": self._space.target.tolist(),
            "constraint_values": constraint_values,
            "allow_duplicate_points": self._allow_duplicate_points,
            "verbose": self._verbose,
            "random_state": random_state,
            "acquisition_params": acquisition_params,
        }
        
        # Add CNN weights to state
        weights = [w.tolist() for w in self._gp.model.get_weights()]
        state['cnn_weights'] = weights
        
        with Path(path).open("w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str | PathLike[str]) -> None:
        """Load optimizer state from a JSON file.

        Parameters
        ----------
        path : str or PathLike
            Path to the JSON file containing the optimizer state.
        """
        with Path(path).open("r") as file:
            state = json.load(file)

        params_array = np.asarray(state["params"], dtype=np.float64)
        target_array = np.asarray(state["target"], dtype=np.float64)
        constraint_array = (
            np.array(state["constraint_values"]) if state["constraint_values"] is not None else None
        )

        for i in range(len(params_array)):
            params = self._space.array_to_params(params_array[i])
            target = target_array[i]
            constraint = constraint_array[i] if constraint_array is not None else None
            self.register(params=params, target=target, constraint_value=constraint)

        self._acquisition_function.set_acquisition_params(state["acquisition_params"])

        if state.get("transformed_bounds") and self._bounds_transformer:
            new_bounds = {
                key: bounds for key, bounds in zip(self._space.keys, np.array(state["transformed_bounds"]))
            }
            self._space.set_bounds(new_bounds)
            self._bounds_transformer.initialize(self._space)


        self._gp.fit(self._space.params, self._space.target)

        if state["random_state"] is not None:
            random_state_tuple = (
                state["random_state"]["bit_generator"],
                np.array(state["random_state"]["state"], dtype=np.uint32),
                state["random_state"]["pos"],
                state["random_state"]["has_gauss"],
                state["random_state"]["cached_gaussian"],
            )
            self._random_state.set_state(random_state_tuple)
