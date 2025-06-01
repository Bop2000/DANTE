"""Pure Python implementation of bayesian global optimization with gaussian processes."""

from __future__ import annotations

import importlib.metadata

from bayes_opt_dnn import acquisition
from bayes_opt_dnn.bayesian_optimization import BayesianOptimization
from bayes_opt_dnn.constraint import ConstraintModel
from bayes_opt_dnn.domain_reduction import SequentialDomainReductionTransformer
from bayes_opt_dnn.logger import ScreenLogger
from bayes_opt_dnn.target_space import TargetSpace

__version__ = importlib.metadata.version("bayesian-optimization")


__all__ = [
    "acquisition",
    "BayesianOptimization",
    "TargetSpace",
    "ConstraintModel",
    "ScreenLogger",
    "SequentialDomainReductionTransformer",
]
