"""Implement domain transformation.

In particular, this provides a base transformer class and a sequential domain
reduction transformer as based on Stander and Craig's "On the robustness of a
simple domain reduction scheme for simulation-based optimization"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np

from bayes_opt_dnn.parameter import FloatParameter
from bayes_opt_dnn.target_space import TargetSpace

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Float = np.floating[Any]


class DomainTransformer(ABC):
    """Base class."""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """To override with specific implementation."""

    @abstractmethod
    def initialize(self, target_space: TargetSpace) -> None:
        """To override with specific implementation."""

    @abstractmethod
    def transform(self, target_space: TargetSpace) -> dict[str, NDArray[Float]]:
        """To override with specific implementation."""


class SequentialDomainReductionTransformer(DomainTransformer):
    """Reduce the searchable space.

    A sequential domain reduction transformer based on the work by Stander, N. and Craig, K:
    "On the robustness of a simple domain reduction scheme for simulation-based optimization"

    Parameters
    ----------
    gamma_osc : float, default=0.7
        Parameter used to scale (typically dampen) oscillations.

    gamma_pan : float, default=1.0
        Parameter used to scale (typically unitary) panning.

    eta : float, default=0.9
        Zooming parameter used to shrink the region of interest.

    minimum_window : float or np.ndarray or dict, default=0.0
        Minimum window size for each parameter. If a float is provided,
        the same value is used for all parameters.
    """

    def __init__(
        self,
        parameters: Iterable[str] | None = None,
        gamma_osc: float = 0.7,
        gamma_pan: float = 1.0,
        eta: float = 0.9,
        minimum_window: NDArray[Float] | Sequence[float] | Mapping[str, float] | float = 0.0,
    ) -> None:
        # TODO: Ensure that this is only applied to continuous parameters
        self.parameters = parameters
        self.gamma_osc = gamma_osc
        self.gamma_pan = gamma_pan
        self.eta = eta

        self.minimum_window_value = minimum_window

    def initialize(self, target_space: TargetSpace) -> None:
        """Initialize all of the parameters.

        Parameters
        ----------
        target_space : TargetSpace
            TargetSpace this DomainTransformer operates on.
        """
        if isinstance(self.minimum_window_value, Mapping):
            self.minimum_window_value = [self.minimum_window_value[key] for key in target_space.keys]
        else:
            self.minimum_window_value = self.minimum_window_value

        any_not_float = any([not isinstance(p, FloatParameter) for p in target_space._params_config.values()])
        if any_not_float:
            msg = "Domain reduction is only supported for all-FloatParameter optimization."
            raise ValueError(msg)
        # Set the original bounds
        self.original_bounds = np.copy(target_space.bounds)
        self.bounds = [self.original_bounds]

        self.minimum_window: NDArray[Float] | Sequence[float]
        # Set the minimum window to an array of length bounds
        if isinstance(self.minimum_window_value, (Sequence, np.ndarray)):
            if len(self.minimum_window_value) != len(target_space.bounds):
                error_msg = "Length of minimum_window must be the same as the number of parameters"
                raise ValueError(error_msg)
            self.minimum_window = self.minimum_window_value
        else:
            self.minimum_window = [self.minimum_window_value] * len(target_space.bounds)

        # Set initial values
        self.previous_optimal = np.mean(target_space.bounds, axis=1)
        self.current_optimal = np.mean(target_space.bounds, axis=1)
        self.r = target_space.bounds[:, 1] - target_space.bounds[:, 0]

        self.previous_d = 2.0 * (self.current_optimal - self.previous_optimal) / self.r

        self.current_d = 2.0 * (self.current_optimal - self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d
        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) + self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

        # check if the minimum window fits in the original bounds
        self._window_bounds_compatibility(self.original_bounds)

    def _update(self, target_space: TargetSpace) -> None:
        """Update contraction rate, window size, and window center.

        Parameters
        ----------
        target_space : TargetSpace
            TargetSpace this DomainTransformer operates on.
        """
        # setting the previous
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d

        self.current_optimal = target_space.params_to_array(target_space.max()["params"])

        self.current_d = 2.0 * (self.current_optimal - self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d

        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) + self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

    def _trim(self, new_bounds: NDArray[Float], global_bounds: NDArray[Float]) -> NDArray[Float]:
        """
        Adjust the new_bounds and verify that they adhere to global_bounds and minimum_window.

        Parameters
        ----------
        new_bounds : np.ndarray
            The proposed new_bounds that (may) need adjustment.

        global_bounds : np.ndarray
            The maximum allowable bounds for each parameter.

        Returns
        -------
        new_bounds : np.ndarray
            The adjusted bounds after enforcing constraints.
        """
        # sort bounds
        new_bounds = np.sort(new_bounds)

        pbounds: NDArray[Float]
        # Validate each parameter's bounds against the global_bounds
        for i, pbounds in enumerate(new_bounds):
            # If the one of the bounds is outside the global bounds, reset the bound to the global bound
            # This is expected to happen when the window is near the global bounds, no warning is issued
            if pbounds[0] < global_bounds[i, 0]:
                pbounds[0] = global_bounds[i, 0]

            if pbounds[1] > global_bounds[i, 1]:
                pbounds[1] = global_bounds[i, 1]

            # If a lower bound is greater than the associated global upper bound,
            # reset it to the global lower bound
            if pbounds[0] > global_bounds[i, 1]:
                pbounds[0] = global_bounds[i, 0]
                warn(
                    "\nDomain Reduction Warning:\n"
                    "A parameter's lower bound is greater than the global upper bound."
                    "The offensive boundary has been reset."
                    "Be cautious of subsequent reductions.",
                    stacklevel=2,
                )

            # If an upper bound is less than the associated global lower bound,
            # reset it to the global upper bound
            if pbounds[1] < global_bounds[i, 0]:
                pbounds[1] = global_bounds[i, 1]
                warn(
                    "\nDomain Reduction Warning:\n"
                    "A parameter's lower bound is greater than the global upper bound."
                    "The offensive boundary has been reset."
                    "Be cautious of subsequent reductions.",
                    stacklevel=2,
                )

        # Adjust new_bounds to ensure they respect the minimum window width for each parameter
        for i, pbounds in enumerate(new_bounds):
            current_window_width = abs(pbounds[0] - pbounds[1])

            # If the window width is less than the minimum allowable width, adjust it
            # Note that when minimum_window < width of the global bounds one side
            # always has more space than required
            if current_window_width < self.minimum_window[i]:
                width_deficit = (self.minimum_window[i] - current_window_width) / 2.0
                available_left_space = abs(global_bounds[i, 0] - pbounds[0])
                available_right_space = abs(global_bounds[i, 1] - pbounds[1])

                # determine how much to expand on the left and right
                expand_left = min(width_deficit, available_left_space)
                expand_right = min(width_deficit, available_right_space)

                # calculate the deficit on each side
                expand_left_deficit = width_deficit - expand_left
                expand_right_deficit = width_deficit - expand_right

                # shift the deficit to the side with more space
                adjust_left = expand_left + max(expand_right_deficit, 0)
                adjust_right = expand_right + max(expand_left_deficit, 0)

                # adjust the bounds
                pbounds[0] -= adjust_left
                pbounds[1] += adjust_right

        return new_bounds

    def _window_bounds_compatibility(self, global_bounds: NDArray[Float]) -> None:
        """Check if global bounds are compatible with the minimum window sizes.

        Parameters
        ----------
        global_bounds : np.ndarray
            The maximum allowable bounds for each parameter.

        Raises
        ------
        ValueError
            If global bounds are not compatible with the minimum window size.
        """
        entry: NDArray[Float]
        for i, entry in enumerate(global_bounds):
            global_window_width = abs(entry[1] - entry[0])
            if global_window_width < self.minimum_window[i]:
                error_msg = "Global bounds are not compatible with the minimum window size."
                raise ValueError(error_msg)

    def _create_bounds(self, parameters: Iterable[str], bounds: NDArray[Float]) -> dict[str, NDArray[Float]]:
        """Create a dictionary of bounds for each parameter.

        Parameters
        ----------
        parameters : Iterable[str]
            The parameters for which to create the bounds.

        bounds : np.ndarray
            The bounds for each parameter.
        """
        return {param: bounds[i, :] for i, param in enumerate(parameters)}

    def transform(self, target_space: TargetSpace) -> dict[str, NDArray[Float]]:
        """Transform the bounds of the target space.

        Parameters
        ----------
        target_space : TargetSpace
            TargetSpace this DomainTransformer operates on.

        Returns
        -------
        dict
            The new bounds of each parameter.
        """
        self._update(target_space)

        new_bounds = np.array([self.current_optimal - 0.5 * self.r, self.current_optimal + 0.5 * self.r]).T

        new_bounds = self._trim(new_bounds, self.original_bounds)
        self.bounds.append(new_bounds)
        return self._create_bounds(target_space.keys, new_bounds)
