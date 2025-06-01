import numpy as np
from dante.obj_functions import *
import os

# import argparse
# parser = argparse.ArgumentParser(description='Process inputs')
# parser.add_argument('--func', help='specify the test function')
# parser.add_argument('--dims', type=int, help='specify the problem dimensions')
# parser.add_argument('--samples', type=int, help='specify the number of samples to collect in the search')
# parser.add_argument('--num_initial_samples', type=int, help='specify the number of initial samples')
# parser.add_argument('--samples_per_acquisition', default=20, type=int, help='specify the number of samples to collect in each acquisition')
# parser.add_argument('--method', help='specify the method to search')
# parser.add_argument('--i', type=int, help='specify the repeat time')
# args = parser.parse_args()

class args:
    def __init__(self, func ='ackley', 
                  dims = 10, 
                  samples = 1000, 
                  num_initial_samples = 200,
                  samples_per_acquisition = 20,
                  method = 'DANTE',
                  i = 1
                  ):
        self.dims    = dims
        self.func    = func
        self.samples =  samples
        self.num_initial_samples =  num_initial_samples
        self.samples_per_acquisition =  samples_per_acquisition
        self.method  = method
        self.i  = i


args = args(func ='rastrigin',       # specify the test function
            dims = 20,            # specify the problem dimensions
            samples = 2000,     # specify the samples to collect in the search
            num_initial_samples = 200,
            samples_per_acquisition = 20,
            method = 'BO-DNN',  # specify the method to search, DANTE TurBO BO BO-DNN
            i = 1,
            )

# Define parameters
NUM_DIMENSIONS = args.dims
NUM_INITIAL_SAMPLES = args.num_initial_samples
NUM_ACQUISITIONS = int(args.samples/args.samples_per_acquisition)
SAMPLES_PER_ACQUISITION = args.samples_per_acquisition

# Initialise the objective function
if args.func == 'ackley':
    obj_function = Ackley(dims=NUM_DIMENSIONS,method=args.method + '-ninit' + str(NUM_INITIAL_SAMPLES),i=args.i)
elif args.func == 'rastrigin':
    obj_function = Rastrigin(dims=NUM_DIMENSIONS,method=args.method + '-ninit' + str(NUM_INITIAL_SAMPLES),i=args.i)
elif args.func == 'rosenbrock':
    obj_function = Rosenbrock(dims=NUM_DIMENSIONS,method=args.method + '-ninit' + str(NUM_INITIAL_SAMPLES),i=args.i)
elif args.func == 'levy':
    obj_function = Levy(dims=NUM_DIMENSIONS,method=args.method + '-ninit' + str(NUM_INITIAL_SAMPLES),i=args.i)
elif args.func == 'schwefel':
    obj_function = Schwefel(dims=NUM_DIMENSIONS,method=args.method + '-ninit' + str(NUM_INITIAL_SAMPLES),i=args.i)
elif args.func == 'michalewicz':
    obj_function = Michalewicz(dims=NUM_DIMENSIONS,method=args.method + '-ninit' + str(NUM_INITIAL_SAMPLES),i=args.i)
elif args.func == 'griewank':
    obj_function = Griewank(dims=NUM_DIMENSIONS,method=args.method + '-ninit' + str(NUM_INITIAL_SAMPLES),i=args.i)

else:
    print('function not defined')
    os._exit(1)


if args.method == 'BO-DNN':
    def obj_function2(**params):
        """
        Universal objective function example (Rosenbrock function)
        Processes parameters in x0, x1... order
        """
        # Extract and sort parameter values (ensuring x0 < x1 < ...)
        dim = len(params)
        x = np.array([params[f'x{i}'] for i in range(dim)])

        return -obj_function(x, apply_scaling=False)

    
    from bayes_opt_dnn import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {f'x{i}': (obj_function.lb[i], obj_function.ub[i]) for i in range(args.dims)}
    
    optimizer = BayesianOptimization(
        f=obj_function2,
        pbounds=pbounds,
        random_state=10,
        )
    
    optimizer.maximize(
        init_points=NUM_INITIAL_SAMPLES,
        n_iter=args.samples,
        )

