# DANTE: Deep Active learning with Neural-surrogate-guided Tree Exploration

## Introduction

DANTE is a deep active learning pipeline that combines deep neural surrogate models and a novel tree search explorationalgorithm to find superior solutions in high-dimensional complex problems characterized by limited data availability. 

For more details, please refer to our [paper](https://arxiv.org/abs/2404.04062).

<img src="assets/dante_flowchart.png" alt="DANTE Flowchart" width="600">

## Installation

DANTE requires `python>=3.10`. Installation of TensorFlow and Keras with CUDA support is strongly recommended.

To install DANTE, run:

```bash
pip install git+https://github.com/Bop2000/DANTE.git
```

Alternatively, you can clone the repository and install it locally:

```bash
git clone git@github.com:Bop2000/DANTE.git
cd DANTE
pip install -e .
```

## Running Tests

To run tests for DANTE, execute the following command in the project root directory:

```bash
python -m pytest -m "not slow"
```

## Sample Code

Here's a minimal example of how to use DANTE:

```python
from dante.deep_active_learning import DeepActiveLearning
from dante.neural_surrogate import AckleySurrogateModel
from dante.obj_functions import Ackley

# Initialise the Ackley objective function with 3 dimensions
obj_function = Ackley(dims=3)

# Create a surrogate model for the Ackley function, training for 5 epochs
surrogate = AckleySurrogateModel(input_dims=obj_function.dims, epochs=5)

# Set up the Deep Active Learning process
# We'll use 30 number of data acquisition in total and start with 10 initial samples
dal = DeepActiveLearning(
    func=obj_function,
    num_data_acquisition=30,
    surrogate=surrogate,
    num_init_samples=10,
)

# Begin the iterative Deep Active Learning process
dal.run()
```

## How to use DANTE to optimise your own function?

To incorporate your own function into DANTE, please encapsulate it within a class as demonstrated below. You can find several examples in the [obj_functions.py](dante/obj_functions.py) file for reference.

```python
@dataclass
class myFunction(ObjectiveFunction):
    dims: int = 10
    turn: float = 0.1
    name: str = "my_function"

    def __post_init__(self):
        self.lb = -5 * np.ones(self.dims) # Define the lower bounds for each dimension of the problem
        self.ub = 5 * np.ones(self.dims)  # Define the upper bounds for each dimension of the problem
        self.tracker = Tracker("my_function")  # Initialise a tracker to monitor the function's performance

    def scaled(self, y: float) -> float:  # Define a scaling function for better surrogate training
        return 100 / (y + 0.01)

    def __call__(self, x: np.ndarray, apply_scaling: bool = True) -> float:
        x = self._preprocess(x)
        y = some_function(x) # Define your function here
        return y if not apply_scaling else self.scaled(y)
```

## License

The source code is released under the MIT license, as presented in [here](LICENSE).