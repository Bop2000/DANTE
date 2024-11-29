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

from obj_functions import *
from dante import *


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
if args.func == 'LunarLander':
    assert args.dims > 0
    f = LunarLander(dims =args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-LunarLander', f=f, iters = args.iterations)
else:
    print('function not defined')
    os._exit(1)



def fx2(x): # for methods who find minimum
    y = fx(x)
    return -y



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


if args.method == 'DANTE':
    aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn)
    init_X = np.random.choice(aaa,size=(args.dims))
    init_y = round(fx(init_X),10)

    ratio = 0.5
        
    exp_weight = ratio * init_y
    
    board_ubt = opt_task(tup=tuple(init_X), value=init_y, terminal=False)
    tree_ubt = DANTE(exploration_weight=exp_weight, f=fx, name=args.func)
    for i in range(args.iterations // args.dims):
        tree_ubt.do_rollout(board_ubt)
        board_ubt = tree_ubt.choose(board_ubt)
        fy = f(np.array(board_ubt.tup))
        tree_ubt.exploration_weight = ratio * round(fy,10)


else:
    print('Search algorithm not defined')
