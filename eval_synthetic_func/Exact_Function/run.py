"""################################################################################
> # **Introduction**
> The notebook is divided into 4 major parts :

*   **Part I** : define the functions
*   **Part II** : optimization using the search algorithm

################################################################################

################################################################################
> # **Part - I**

*   Define the functions
*   Generate initial dataset
*   Set parameters

################################################################################
"""

############################### Import libraries ###############################

import argparse
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys 
sys.path.append("../..") 
from dots.functions_exact import *
from dots.dots_exact import *


############################### Set the objective ###############################

parser = argparse.ArgumentParser(description='Process inputs')
parser.add_argument('--func', help='specify the test function')
parser.add_argument('--dims', type=int, help='specify the problem dimensions')
parser.add_argument('--iterations', type=int, help='specify the iterations to collect in the search')
parser.add_argument('--method', help='specify the method to search')
args = parser.parse_args()
print("Test function:",args.func)
print("Problem dimensions:",args.dims)
print("Max number of samples:",args.iterations)
print("Optimization algorithm:",args.method)

############################### Set parameters ###############################

# Set the random seed for reproducibility
random.seed(42)



############################### Define the functions ###############################

f = None
iteration = 0
if args.func == 'ackley':
    assert args.dims > 0
    f = Ackley(dims =args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Ackley', f=f, iters = args.iterations)
elif args.func == 'rastrigin':
    assert args.dims > 0
    f = Rastrigin(dims =args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Rastrigin', f=f, iters = args.iterations)
elif args.func == 'rosenbrock':
    assert args.dims > 0
    f = Rosenbrock(dims =args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Rosenbrock', f=f, iters = args.iterations)
elif args.func == 'levy':
    f = Levy(dims = args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Levy', f=f, iters = args.iterations)
elif args.func == 'schwefel':
    f = Schwefel(dims = args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Schwefel', f=f, iters = args.iterations)
elif args.func == 'michalewicz':
    f = Michalewicz(dims = args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Michalewicz', f=f, iters = args.iterations)
elif args.func == 'griewank':
    f = Griewank(dims = args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Griewank', f=f, iters = args.iterations)

else:
    print('function not defined')
    os._exit(1)

assert args.dims > 0
assert f is not None
assert args.iterations > 0

lower = f.lb
upper = f.ub

bounds = []
for idx in range(0, len(f.lb) ):
    bounds.append( ( float(f.lb[idx]), float(f.ub[idx])) )


################################# End of Part I ################################




"""################################################################################
> # **Part - II**

*   optimization using the search algorithm

################################################################################
"""

################################ optimization using the search algorithm ###############################



if args.method == 'DOTS':
    aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn)
    init_X = np.random.choice(aaa,size=(args.dims))
    init_y = round(1000/(fx(init_X)+0.1),10)
    if args.func == 'ackley' or args.func == 'rastrigin' or args.func == 'schwefel':
        ratio = 0.01
    elif args.func == 'rosenbrock':
        ratio = 0.1
    elif args.func == 'griewank' or args.func == 'michalewicz':
        ratio = 0.1
    else:
        ratio = 0.01
        
    exp_weight = ratio * init_y
    
    board_ubt = opt_task(tup=tuple(init_X), value=init_y, terminal=False)
    tree_ubt = DOTS(exploration_weight=exp_weight, f=f, name=args.func)
    for i in range(args.iterations):
        tree_ubt.do_rollout(board_ubt)
        board_ubt = tree_ubt.choose(board_ubt)
        fy = fx(board_ubt.tup)
        tree_ubt.exploration_weight = ratio * round(1000/(fy+0.1),10)
        if args.func == 'michalewicz':
            print("iteration:", i, "current best f(x):", fx.tracker.curt_best - fx.dims + 0.3)
        if round(fy , 5) == 0:
            break
        
        if i % int(1e9/(args.dims**2)) == 0:
            tree_ubt = DOTS(exploration_weight=ratio * round(1000/(fy+0.1),10), f=f, name=args.func)



elif args.method == 'DOTS-Greedy':
    aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn)
    init_X = np.random.choice(aaa,size=(args.dims))
    init_y = round(fx(init_X),5)
    
    optimizer = DOTS_Greedy(f = f, dims = args.dims)
    for i in range(args.iterations):
        node_new, temp_Y = optimizer.choose(init_X)
        if temp_Y<init_y:
              init_X = np.array(node_new)
              init_y=np.array(temp_Y)
        fy = fx(node_new)
        if round(fy , 5) == 0:
            break

elif args.method == 'DOTS-eGreedy':
    aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn)
    init_X = np.random.choice(aaa,size=(args.dims))
    init_y = round(fx(init_X),5)
    
    optimizer = DOTS_eGreedy(f = f, dims = args.dims)
    for i in range(args.iterations):
        node_new, temp_Y = optimizer.choose(init_X)
        if temp_Y < init_y * (1 + 0.05 * np.random.random()):
              init_X = np.array(node_new)
              init_y=np.array(temp_Y)
        fy = fx(node_new)
        if round(fy , 5) == 0:
            break

elif args.method == 'Random':
    aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn)
    init_X = np.random.choice(aaa,size=(args.iterations,args.dims))
    for i in init_X:
        init_y = fx(i)


elif args.method == 'DualAnnealing':
    from scipy.optimize import dual_annealing
    res = dual_annealing(fx,                          # the function to minimize
                         bounds = bounds,            # the bounds on each dimension of x
                         maxiter=args.iterations,    # The maximum number of global search iterations. Default value is 1000.
                         initial_temp=5230.0,        # The initial temperature, use higher values to facilitates a wider search of the energy landscape, allowing dual_annealing to escape local minima that it is trapped in. Default value is 5230. Range is (0.01, 5.e4].
                         restart_temp_ratio=2e-05,   # During the annealing process, temperature is decreasing, when it reaches initial_temp * restart_temp_ratio, the reannealing process is triggered. Default value of the ratio is 2e-5. Range is (0, 1).
                         visit=2.62,                 # Parameter for visiting distribution. Default value is 2.62. Higher values give the visiting distribution a heavier tail, this makes the algorithm jump to a more distant region. The value range is (1, 3].
                         accept=-5.0,                # Parameter for acceptance distribution. It is used to control the probability of acceptance. The lower the acceptance parameter, the smaller the probability of acceptance. Default value is -5.0 with a range (-1e4, -5].
                         maxfun=args.iterations,     # Soft limit for the number of objective function calls. If the algorithm is in the middle of a local search, this number will be exceeded, the algorithm will stop just after the local search is done. Default value is 1e7.
                         seed=None,
                         no_local_search=False,
                         callback=None,
                         # x0=x_current                # Coordinates of a single N-D starting point.
                         )

elif args.method == 'DifferentialEvolution':
    from scipy.optimize import differential_evolution
    res = differential_evolution(fx,                        # the function to minimize
                                 bounds=bounds,            # the bounds on each dimension of x
                                 maxiter=args.iterations,  # The maximum number of generations over which the entire population is evolved. The maximum number of function evaluations (with no polishing) is: (maxiter + 1) * popsize * (N - N_equal)
                                 # x0=x_current,             # Coordinates of a single N-D starting point.
                                 popsize=15                # A multiplier for setting the total population size. The population has popsize * (N - N_equal) individuals. This keyword is overridden if an initial population is supplied via the init keyword. When using init='sobol' the population size is calculated as the next power of 2 after popsize * (N - N_equal).
                                 )

elif args.method == 'CMA-ES':
    import cma
    aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn)
    x_init = np.random.choice(aaa,size=(args.dims))
    # x_init = np.random.choice(101,size=(args.dims))/10-5
    while fx.counter < args.iterations:
        options = {'maxiter':args.iterations - fx.counter,
                   'bounds':[f.lb[0], f.ub[0]]
                   }
        res = cma.fmin(fx,                    # the function to minimize
                       x_init,               # Coordinates of a single N-D starting point.
                       1,
                       options,restart_from_best=True)
        print(fx.tracker.curt_best_x)
        x_init = np.array(fx.tracker.curt_best_x)



################################ Visualization ###############################

f = open(f"{fx.tracker.foldername}/result")
yourList = f.readlines()
yourList2=[]
max_len = 0
for i in yourList:
    i=i.strip('[')
    i=i.strip(']\n')
    # i=i.split(',')
    i = [item.strip() for item in i.split(',')]
    yourList2.append(i)
    if len(i) > max_len:
        max_len = len(i)
yourList3 = []
for i in yourList2:
    ii = np.array(i).astype(float)
    if len(ii) < max_len:
        ii = np.concatenate((ii, np.zeros(max_len-len(ii))))
    yourList3.append(ii)
results = np.array(yourList3)
print(results.shape)

plt.plot(results.reshape(-1))
plt.title(['ground truth progress'])

################################ End of Part II ################################

"""################################################################################

---------------------------------------------------------------------------- That's all folks ! ----------------------------------------------------------------------------


################################################################################
"""