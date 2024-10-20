from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dante.obj_functions import ObjectiveFunction


@dataclass
class Tracker:
    """A class for tracking optimization results and saving them periodically."""

    folder_name: str
    _counter: int = field(init=False, default=0)
    _results: list[float] = field(init=False, default_factory=list)
    _x_values: list[Optional[np.ndarray]] = field(init=False, default_factory=list)
    _current_best: float = field(init=False, default=float("inf"))
    _current_best_x: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self):
        """Initialize the tracker and create the folder after instance creation."""

        self._create_folder()

    def _create_folder(self) -> None:
        """Create a folder to store results."""
        try:
            os.mkdir(self.folder_name)
            print(f"Successfully created the directory {self.folder_name}")
        except OSError:
            print(f"Creation of the directory {self.folder_name} failed")

    def dump_trace(self) -> None:
        """Save the current results to a file."""
        np.save(
            f"{self.folder_name}/result.npy", np.array(self._results), allow_pickle=True
        )

    def track(
            self, result: float, x: Optional[np.ndarray] = None, save: bool = False
    ) -> None:
        """Track a new result and update the best if necessary.

        Args:
            result: The current optimization result.
            x: The current x value.
            save: Whether to save results immediately.
        """
        self._counter += 1
        if result < self._current_best:
            self._current_best = result
            self._current_best_x = x

        self._print_status()
        self._results.append(self._current_best)
        self._x_values.append(x)

        if save or self._counter % 20 == 0 or round(self._current_best, 5) == 0:
            self.dump_trace()

    def _print_status(self) -> None:
        """Print the current status of the optimization."""
        print("\n" + "=" * 10)
        print(f"#samples: {self._counter}, total samples: {len(self._results) + 1}")
        print("=" * 10)
        print(f"current best f(x): {self._current_best}")
        print(f"current best x: {np.around(self._current_best_x, decimals=4)}")


def generate_initial_samples(
        objective_function: ObjectiveFunction, sample_count: int = 200, apply_scaling: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate initial random samples for the given objective function.

    Args:
        objective_function (ObjectiveFunction): An instance of a class derived from ObjectiveFunction.
        sample_count (int): Number of samples to generate. Default is 200.
        apply_scaling (bool): Whether to apply scaling to the function output. Default is False.

    Returns:
        A tuple containing (input_samples, output_values)
            input_samples (np.ndarray): Array of input points.
            output_values (float): Function output value.
    """
    assert sample_count > 0, "sample_count must be positive"

    dimension_count = objective_function.dims
    lower_bounds, upper_bounds = objective_function.lb, objective_function.ub
    step_size = objective_function.turn

    # Generate random points within the function's bounds
    value_range = np.arange(lower_bounds[0], upper_bounds[0] + step_size, step_size).round(5)
    input_samples = np.random.choice(value_range, size=(sample_count, dimension_count))

    output_values = np.array([objective_function(x, apply_scaling=apply_scaling) for x in input_samples])

    print(f"\n{'=' * 20}")
    print(f"{sample_count} initial data points collection completed.")
    print(f"{'=' * 20}\n")

    return input_samples, output_values
