# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/", "height": 430} id="T1rifox3d3As" outputId="df3f70b6-0c00-448a-ba29-1b9e8ac3e736"
import numpy as np
import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time

import os
import random
from tqdm import tqdm

from abc import ABC, abstractmethod
from collections import defaultdict
import math

from collections import namedtuple

from get_sc import *
# -

# # Set the random seed for reproducibility
# random.seed()

pdb = '3p72.pdb'
seq_len = 11

seq_folder = './result/MCTS_peptide_' + pdb[:4] + '_' + time.strftime("%Y-%m-%d_%H-%M", time.localtime())
if not os.path.exists(seq_folder):
# If it doesn't exist, create it
    os.makedirs(seq_folder)

coefficients = {
        'TOTAL SASA': None,
        'NUMBER OF RESIDUES': None,
        'AVG RESIDUE ENERGY': None,
        'INTERFACE DELTA SASA': 0.3,
        'INTERFACE HYDROPHOBIC SASA': None,
        'INTERFACE POLAR SASA': None,
        'CROSS-INTERFACE ENERGY SUMS': None,
        'SEPARATED INTERFACE ENERGY DIFFERENCE': -0.3,
        'CROSS-INTERFACE ENERGY/INTERFACE DELTA SASA': None,
        'SEPARATED INTERFACE ENERGY/INTERFACE DELTA SASA': None,
        'DELTA UNSTAT HBONDS': None,
        'CROSS INTERFACE HBONDS': None,
        'HBOND ENERGY': None,
        'HBOND ENERGY/ SEPARATED INTERFACE ENERGY': None,
        'INTERFACE PACK STAT': None,
        'SHAPE COMPLEMENTARITY VALUE': 0.4
    }

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_nature_value(pdb_file):
    pdb_name = pdb_file.split('.')[0]
    def run_command_silently(command):
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    run_command_silently(f"InterfaceAnalyzer.default.linuxgccrelease -s {pdb_name}.pdb @pack_input_options.txt > {pdb_name}_log.txt")
    if os.path.exists("pack_input_score.sc"):
        os.rename("pack_input_score.sc", f"{pdb_name}_pack_input_score.sc")
    else:
        print("Expected file pack_input_score.sc does not exist.")
    run_command_silently(f"mv {pdb_name}_* ./{seq_folder}")
    return extract_values_from_rosetta_output(f'./{seq_folder}/{pdb_name}_log.txt')

nature_value = get_nature_value(pdb)
print(nature_value)
print('nature target: ', float(nature_value['SHAPE COMPLEMENTARITY VALUE']) * float(nature_value['INTERFACE DELTA SASA']) / 100)
print('TRY TO EXCEED IT!!!!!')



# +
class MCTS_ubt:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.Nstay = 0
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node not in self.children:
            print('not seen before, randomly sampled!')
            return node.find_random_child()
        
        print(f'number of visit is {self.N[node]}')
        log_N_vertex = math.log(self.N[node])
        def uct(n):
            "modified Upper confidence bound for trees"
            uct_value = n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n]+1))
            return uct_value
        if (self.Nstay + 1) % 4 == 0:
            action = [p for p in range(0, len(node.tup))]
            self.children[node] = self.children[node] | node.find_children(action)
        media_node = max(self.children[node], key=uct)#self._uct_select(node)
        rand_index = random.randint(0, len(list(self.children[node]))-1)
        node_rand = list(self.children[node])[rand_index]
        print(f'uct of the node is{uct(node)} ')
        if uct(media_node) > uct(node):
            print(f'better uct media node : {uct(media_node)}')
            print(f'better value media node : {media_node.value}')
            print('media_node: ', media_node)
            print('node_rand: ', node_rand)
            return media_node, media_node.value, node_rand, node_rand.value
        self.Nstay += 1
        print('node stays!', self.Nstay)
        print('node: ', node)
        print('node_rand: ', node_rand)
        return node, node.value, node_rand, node_rand.value

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
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
              return path
            unexplored = self.children[node] - self.children.keys()
            def evaluate(n):
              return n.value
            if unexplored:
              #n= unexplored.pop()
              path.append(max(unexplored, key=evaluate))#
              return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        action = [p for p in range(0, len(node.tup))]
        self.children[node] = node.find_children(action)

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        reward = node.value
        return reward


    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
          self.N[node] += 1
          self.Q[node] += reward
            #reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

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
        print(f'node with max uct is:{uct_node}')
        return uct_node

class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
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

_OT = namedtuple("opt_task", "tup value unsatH plddt terminal") # tup is the initial seq
class opt_task(_OT, Node):
    def find_children(board,action):
        if board.terminal:
            return set()
        all_tup=[]
        for index in action:
            tup = list(board.tup)
            flip = random.randint(0,6)
            if flip <= 0:
                while True:
                    mutation = random.randint(0,19)
                    if mutation != tup[index]:
                        break
                tup[index] = mutation
            elif flip <= 1:
                for i in range(int(seq_len/3)):
                    index_2 = random.randint(0, len(tup)-1)
                    tup[index_2] = random.randint(0,19)
            elif flip <= 5:
                p = np.maximum(1-np.array(eval(board.plddt)), 0) * 100
                p = softmax(p)
                index_2 = np.random.choice(seq_len, np.random.randint(0, seq_len) + 1, p=p, replace=False)
                for i in index_2:
                    tup[i] = random.randint(0,19)
            elif flip:
                while True:
                    new_tup = [random.randint(0, 19) for _ in range(seq_len)]
                    if new_tup != tup:
                        break
                tup = new_tup
            print(tup, int2aa(tup))
            all_tup.append(tup)
                
        all_value = []
        all_unsatH = []
        all_plddt = []
        for seq in all_tup:
            metrics, aux = get_value(seq_folder, seq, pdb)
            all_value.append(float(metrics['SHAPE COMPLEMENTARITY VALUE']) * float(metrics['INTERFACE DELTA SASA']) / 100)
            all_unsatH.append(metrics['unsatH_residues'])
            all_plddt.append(str(list(aux["all"]["plddt"][0][-seq_len:])))
        is_terminal=False
        return {opt_task(tuple(t), v, u, p, is_terminal) for t, v, u, p in zip(all_tup, all_value, all_unsatH, all_plddt)}

    def find_random_child(board):
        pass

    def find_uct_child(board, action):
        pass
        
    def reward(board):
        pass
        
    def is_terminal(board):
        return board.terminal


# + id="ZjqwsClse2jE"
def most_visit_node(tree_ubt, initial_X, top_n):
  N_visit = tree_ubt.N
  childrens = [i for i in tree_ubt.children]
  children_N = []
  X_top = []
  y_top = []
  for child in childrens:
    child_tup = np.array(child.tup)
    children_N.append(N_visit[child])
    X_top.append(child_tup)
    y_top.append(child.value)
  children_N = np.array(children_N)
  X_top = np.array(X_top)
  y_top = np.array(y_top)
  ind = np.argpartition(children_N, -top_n)[-top_n:]
  X_topN = X_top[ind]
  y_topN = y_top[ind]
  return X_topN, y_topN


def single_run(initial_X,top_n):
  metrics, aux = get_value(seq_folder, initial_X, pdb)
  values = float(metrics['SHAPE COMPLEMENTARITY VALUE']) * float(metrics['INTERFACE DELTA SASA']) / 100
  unsatH = metrics['unsatH_residues']
  plddt = list(aux["all"]["plddt"][0][-seq_len:])
  print(initial_X, values, unsatH, plddt)
  exp_weight = 0.4 * values
  if exp_weight < 2:
      exp_weight = 2
  board_uct = opt_task(tup=tuple(initial_X), value=values, unsatH=unsatH, plddt=str(plddt), terminal=False)
  rollout_round = 15
  tree_ubt = MCTS_ubt(exploration_weight=exp_weight)
  boards = []
  board_values = []

  for i in tqdm(range(0, rollout_round)):
      print(i)
      tree_ubt.do_rollout(board_uct)
      board_uct, board_uct_value, board_rand, board_rand_value = tree_ubt.choose(board_uct)
      boards.append(list(board_uct.tup))
      board_values.append(board_uct_value)
      boards.append(list(board_rand.tup))
      board_values.append(board_rand_value)
    
  new_x = []
  new_pred = []
  boards = np.array(boards)
        
  for i,j in zip(boards,board_values):
    temp_x = np.array(i)
    new_pred.append(j)
    new_x.append(temp_x)
  #print(new_pred)
  new_x= np.array(new_x)
  new_pred = np.array(new_pred)
  temp = np.concatenate((new_x, new_pred.reshape(-1, 1)), axis=1)
  temp = np.unique(temp, axis=0)
  new_x = temp[:, :-1]
  new_pred = temp[:, -1]

  ind = np.argpartition(new_pred, -top_n)[-top_n:]
  top_x =  new_x[ind]
  top_y = new_pred[ind]
  print('top x: ',top_x)
  print('top y: ', top_y)

  X_most_visit, y_topN=  most_visit_node(tree_ubt, initial_X, 1)
  X_next = np.concatenate([top_x, X_most_visit])
  y_next = np.concatenate([top_y, y_topN])
  y_next = np.array(y_next)
  return X_next, y_next


# + id="_Tf_t_PifSm5"
def run():
    x_current_top = [random.randint(0, 19) for _ in range(seq_len)]
    y_top=[]
    X_top=[]

    x, y_0 = single_run(x_current_top,2)
    y_top.append(y_0)
    X_top.append(x)

    for i in x:
        new_x, new_y = single_run(i, 2)
        X_top.append(new_x)
        y_top.append(new_y)

    return X_top,y_top


# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Wd538Uu0fjsC" outputId="40792bda-b49f-4466-bd57-1c83ac191db6"
X_top,y_top = run()
print(X_top)
print(y_top)
