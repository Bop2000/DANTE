from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from dante.neural_surrogate import SurrogateModel
from dante.obj_functions import ObjectiveFunction
from dante.tree_exploration import TreeExploration
from dante.utils import generate_initial_samples


@dataclass
class DeepActiveLearning:
    func: ObjectiveFunction
    num_data_acquisition: int
    surrogate: SurrogateModel
    tree_explorer_args: Dict[str, Any] = field(default_factory=dict)
    num_init_samples: int = 200
    num_samples_per_acquisition: int = 20
    input_x: np.ndarray = None
    input_scaled_y: np.ndarray = None

    def __post_init__(self):
        assert self.num_data_acquisition > 0
        self.dims = self.func.dims
        self.input_x, self.input_scaled_y = generate_initial_samples(
            self.func, self.num_init_samples, apply_scaling=True
        )

    def run(self):
        for i in range(self.num_data_acquisition // self.num_samples_per_acquisition):
            model = self.surrogate(self.input_x, self.input_scaled_y, verbose=True)
            tree_explorer = TreeExploration(
                func=self.func,
                model=model,
                num_samples_per_acquisition=self.num_samples_per_acquisition,
                **self.tree_explorer_args,
            )
            top_x = tree_explorer.rollout(
                self.input_x,
                self.input_scaled_y,
                iteration=i,
            )
            top_y = np.array([self.func(x, apply_scaling=True) for x in top_x])
            self.input_x = np.concatenate((self.input_x, top_x), axis=0)
            self.input_scaled_y = np.concatenate((self.input_scaled_y, top_y))

            if np.isclose(self.input_scaled_y.min(), 0.0):
                break
