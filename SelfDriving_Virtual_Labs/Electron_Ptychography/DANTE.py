
"""################################################################################
> # **Introduction**
> The notebook is divided into 4 major parts :

*   **Part I** : import dataset and visulization expert params
*   **Part II** : define DOTS algorithm
*   **Part III** : optimization using DOTS

################################################################################

################################################################################
> # **Part - I**

*   Import initial dataset
*   Visulization of reconstructed patterns using expert params
*   Set parameters

################################################################################
"""

############################### Import libraries ###############################

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from collections import namedtuple
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import py4DSTEM
print(py4DSTEM.__version__)


############################### Import Initial Dataset ###############################

dataset =  py4DSTEM.read("ptycho_Si-110_18nm.h5")
dataset.calibration






############################### Expert params ###############################

semiangle_cutoff      = 20
defocus               = 100
energy                = 200e3
max_iter              = 256
step_size             = 0.175
identical_slices_iter = 256
slice_thicknesses     = 30.794230884706234
num_slices            = 6


ms_ptycho_18nm = py4DSTEM.process.phase.MultislicePtychographicReconstruction(
    datacube=dataset,
    num_slices=num_slices,
    slice_thicknesses=slice_thicknesses,
    verbose=True,
    energy=energy,
    defocus=defocus,
    semiangle_cutoff=semiangle_cutoff,
    object_padding_px=(18,18),
    device='gpu',
).preprocess(
    plot_center_of_mass = False,
    plot_rotation=False,
)

ms_ptycho_18nm = ms_ptycho_18nm.reconstruct(
    reset=True,
    store_iterations=True,
    max_iter = max_iter,
    identical_slices_iter= identical_slices_iter,
    step_size=step_size ,
).visualize(
    iterations_grid = 'auto',
)
ms_ptycho_18nm._visualize_last_iteration(
    fig=None,
    cbar=True,
    plot_convergence=True,
    plot_probe=True,
    plot_fourier_probe=True,
    padding=0,
)

ms_ptycho_18nm.error
print('NMSE of Expert params:',ms_ptycho_18nm.error)

############################### Set Paramaters ###############################

weight_ratio = 0.5 # exploration weight = weight_ratio * max(score)

# 8 parameters to optimize, here are the ranges of these params
semiangle_cutoff = np.arange(1, 30.1, 0.1).round(1)
defocus          = np.arange(1, 201, 1).round(0)
energy           = np.arange(1e3, 300e3, 1000).round(0)
max_iter         = np.round(np.arange(1, 21, 1))
step_size        = np.arange(0.01,1,0.01).round(2)
identical_slices_iter = np.round(np.arange(1, 500, 1))
slice_thicknesses     = np.arange(1,50,0.1).round(1)
num_slices       = np.round(np.arange(1, 101, 1))

all_para = [semiangle_cutoff,
            defocus,
            energy,
            max_iter,
            step_size ,
            identical_slices_iter,
            slice_thicknesses,
            num_slices]



################################ reconstruction using py4stem ###############################


def oracle(x): # reconstruction and calculate the NMSE value using py4stem
    semiangle_cutoff = x[0]
    defocus          = x[1]
    energy           = x[2]
    max_iter         = x[3]
    step_size        = x[4]
    identical_slices_iter = x[5]
    slice_thicknesses     = x[6]
    num_slices       = x[7]

    ms_ptycho_18nm = py4DSTEM.process.phase.MultislicePtychographicReconstruction(
        datacube=dataset,
        num_slices=round(num_slices),
        slice_thicknesses=slice_thicknesses,
        verbose=True,
        energy=energy,
        defocus=defocus,
        semiangle_cutoff=semiangle_cutoff,
        object_padding_px=(18,18),
        device='gpu',
    ).preprocess(
        plot_center_of_mass = False,
        plot_rotation=False,
    )
    plt.close()
    ms_ptycho_18nm = ms_ptycho_18nm.reconstruct(
        reset=True,
        store_iterations=True,
        max_iter = round(max_iter),
        identical_slices_iter= round(identical_slices_iter),
        step_size=step_size,
    ).visualize(
        iterations_grid = 'auto',
    )

    plt.close()
    print(ms_ptycho_18nm.error)
    return ms_ptycho_18nm.error

def value_cal(x):
    value=oracle(x)
    return 1/value , value

def oracle_show(x): # reconstruction and visulization using py4stem
    semiangle_cutoff = x[0]
    defocus          = x[1]
    # rotation_degrees = x[2]
    energy           = x[2]
    max_iter         = x[3]
    step_size        = x[4]
    identical_slices_iter = x[5]
    slice_thicknesses     = x[6]
    num_slices       = x[7]

    ms_ptycho_18nm = py4DSTEM.process.phase.MultislicePtychographicReconstruction(
        datacube=dataset,
        num_slices=round(num_slices),
        slice_thicknesses=slice_thicknesses,
        verbose=True,
        energy=energy,
        defocus=defocus,
        semiangle_cutoff=semiangle_cutoff,
        object_padding_px=(18,18),
        device='gpu',
    ).preprocess(
        plot_center_of_mass = False,
        plot_rotation=False,
    )
    # plt.close()
    ms_ptycho_18nm = ms_ptycho_18nm.reconstruct(
        reset=True,
        store_iterations=True,
        max_iter = round(max_iter),
        identical_slices_iter= round(identical_slices_iter),
        step_size=step_size,
    ).visualize(
        iterations_grid = 'auto',
    )

    ms_ptycho_18nm._visualize_last_iteration(
        fig=None,
        cbar=True,
        plot_convergence=True,
        plot_probe=True,
        plot_fourier_probe=True,
        padding=0,
    )

    print(ms_ptycho_18nm.error)
    return ms_ptycho_18nm.error





############################### Generate 20 initial points ###############################

n_dim = len(all_para)
num_initial = 20 # number of initial points
para1 = np.random.choice(semiangle_cutoff,num_initial)
para2 = np.random.choice(defocus,num_initial)
para4 = np.random.choice(energy,num_initial)
para5 = np.random.choice(max_iter,num_initial)
para6 = np.random.choice(step_size,num_initial)
para7 = np.random.choice(identical_slices_iter,num_initial)
para8 = np.random.choice(slice_thicknesses,num_initial)
para9 = np.random.choice(num_slices,num_initial)


initial_points = np.concatenate(([para1],
                                  [para2],
                                  [para4],
                                   [para5],
                                  [para6],
                                  [para7],
                                  [para8],
                                  [para9]),axis=0).T
initial_values = []
for i in initial_points:
    print(i)
    initial_value = oracle(i)
    initial_values.append(initial_value)
initial_values = np.array(initial_values)
initial_point = initial_points[np.argmin(initial_values)].reshape(-1)
df = pd.DataFrame(np.concatenate((initial_points,initial_values.reshape(-1,1)),axis=1))
df.columns= ['semiangle_cutoff',
              'defocus',
              'energy',
               'max_iter',
              'step_size',
              'identical_slices_iter',
                'slice_thicknesses',
              'num_slices',
              'NMSE']
df.to_csv('initial.csv')

################################# End of Part I ################################

"""################################################################################
> # **Part - II**

*   Define the DOTS alghorithm

################################################################################
"""

################################# DOTS alghorithm ################################


class DOTS:
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.Qa = dict.fromkeys(list(range(20)), 0)
        self.Na = dict.fromkeys(list(range(20)), 1)
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node."
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            print('not seen before, randomly sampled!')
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
               return '-inf'  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward
        def evaluate(n):
            return n.value  # average reward
        log_N_vertex = math.log(self.N[node])
        def uct(n):
            "Upper confidence bound for trees"
            uct_value = n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n]+1))
            return uct_value

        media_node = max(self.children[node], key=uct)#self._uct_select(node)
        print(f'number of visit is {self.N[media_node]}')
        print(f'uct media node : {media_node}')
        print(f'uct of the node is{uct(media_node)} ')
        if uct(media_node) > uct(node):
            # print(f'number of visit is {self.N[media_node]}')
            # print(f'better uct media node : {media_node}')
            # print(f'uct of the node is{uct(media_node)} ')
            return media_node
        return node

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        count = 0
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
              return path
            unexplored = self.children[node] - self.children.keys()
            def evaluate(n):
              return n.value
            if count == 50:
              return max(path, key=evaluate)
            if unexplored:
              path.append(max(unexplored, key=evaluate))#
              return path
            node = self._uct_select(node)  # descend a layer deeper
            count+=1

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        action = [p for p in range(0, len(node.tup))]
        self.children[node] = node.find_children(action)

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        reward = node.reward()
        return reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
          self.N[node] += 1
          self.Q[node] += reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            uct_value = n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n]+1))
            return uct_value
        uct_node = max(self.children[node], key=uct)
        return uct_node

class Node(ABC):
    """
    A representation of a single board state.
    DOTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """
    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True

_OT = namedtuple("opt_task", "tup value terminal")

# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class opt_task(_OT, Node):

    def find_children(board,action):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        return {
            board.make_move(i) for i in action}

    def reward(board):
        return board.value

    def is_terminal(board):
        return board.terminal

    ############################ design the action space ############################

    def make_move(board, index):
        print(index)
        tup = list(board.tup)
        flip = np.random.randint(0,6)
        if flip ==0:
            indices = np.where(all_para[index]==tup[index])[0][0]
            try:
                tup[index] = all_para[index][indices+1]
            except:
                tup[index] = all_para[index][indices-1]

        elif flip ==1:
            indices = np.where(all_para[index]==tup[index])[0][0]
            try:
                tup[index] = all_para[index][indices-1]
            except:
                tup[index] = all_para[index][indices+1]

        elif flip ==2:
            n_flip = np.random.choice(np.arange(n_dim),int(n_dim/3),replace=False)
            for i in n_flip:
                tup[i] = np.random.choice(all_para[i],1)[0]

        elif flip ==3:
            n_flip = np.random.choice(np.arange(n_dim),int(n_dim/2),replace=False)
            for i in n_flip:
                tup[i] = np.random.choice(all_para[i],1)[0]

        else:
            tup[index] = np.random.choice(all_para[index],1)[0]

        tup[index] = round(tup[index],2)
        print(tup)
        value,ture_value = value_cal(tup)
        print('ptycho.error:',ture_value)
        is_terminal = False
        return opt_task(tuple(tup), value, is_terminal)

################################ End of Part II ################################

"""################################################################################
> # **Part - III**

*   Optimization using DOTS

################################################################################

Input description:
*   initial_point: initial node for DOTS

Output description:

*   all_input: newly sampled data
*   all_value: NMSE of sampled data
"""

################################ Optimization using DOTS ###############################

# Initialization
x_current = np.round(initial_point,5)
x = tuple(x_current)
values, cu_Y = value_cal(x)
exploration_weight = weight_ratio*values
board_ubt = opt_task(tup = x, value = values, terminal = False)
tree_ubt = DOTS(exploration_weight = exploration_weight)

# Optimization start
print('start...')
sign1 = 0 #Determine whether convergence
all_value=list([cu_Y])
all_input=list([x_current])
print('ptycho.error:',cu_Y)
for i in tqdm(range(0, 500, 1)):
   print('iteration:',i)
   tree_ubt.do_rollout(board_ubt)
   board_ubt = tree_ubt.choose(board_ubt)
   value_new = board_ubt.value
   temp_Y    = 1/value_new

   if temp_Y<cu_Y:
       cu_Y=float(temp_Y)
       x_current = np.array(board_ubt.tup)
       exploration_weight=weight_ratio*value_new
       tree_ubt = DOTS(exploration_weight=exploration_weight)
       print('current best ptycho.error:',cu_Y)

   all_value.append(cu_Y)
   all_input.append(x_current)

   df2 = pd.DataFrame(np.concatenate((np.array(all_input),np.array(all_value).reshape(-1,1)),axis=1))
   df2.columns= ['semiangle_cutoff',
                 'defocus',
                 'energy',
                  'max_iter',
                 'step_size',
                 'identical_slices_iter',
                 'slice_thicknesses',
                 'num_slices',
                 'NMSE']
   df2.to_csv('results-DOTS.csv')


   if round(cu_Y,10)==0:
        sign1=1
        break
print('final value:',cu_Y)
print('completed!')

################################ Final optimized params ###############################

max_input = all_input[np.argmin(all_value)]

print('Final optimized params')
print('semiangle_cutoff:',max_input[0])
print('defocus:',max_input[1])
print('energy:',max_input[2])
print('max_iter:',max_input[3])
print('step_size:',max_input[4])
print('identical_slices_iter:',max_input[5])
print('slice_thicknesses:',max_input[6])
print('num_slices:',max_input[7])

print('NMSE:',cu_Y)


################################ Visualization with final optimized params ###############################

oracle_show(max_input)

################################ End of Part III ################################

"""################################################################################

---------------------------------------------------------------------------- That's all folks ! ----------------------------------------------------------------------------


################################################################################
"""
