import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import random


def create_new(tup, index, aaa, f):
    flip = random.randint(0,5)
    # if flip ==5:
    #   tup[index] += f.turn
    # elif flip ==6:
    #   tup[index] -= f.turn
    if flip ==0:
        for i in range(int(f.dims/5)):
          index_2 = random.randint(0, len(tup)-1)
          tup[index_2] = np.random.choice(aaa)
    elif flip ==1:
        for i in range(int(f.dims/10)):
          index_2 = random.randint(0, len(tup)-1)
          tup[index_2] = np.random.choice(aaa)
    elif flip ==2:
        for i in range(int(f.dims/20)):
          index_2 = random.randint(0, len(tup)-1)
          tup[index_2] = np.random.choice(aaa)
    elif flip ==3:
        for i in range(int(f.dims/50)):
          index_2 = random.randint(0, len(tup)-1)
          tup[index_2] = np.random.choice(aaa)
    elif flip ==4:
        for i in range(int(f.dims/3)):
          index_2 = random.randint(0, len(tup)-1)
          tup[index_2] = np.random.choice(aaa)
    else:
        tup[index] = np.random.choice(aaa)
    tup[index] = round(tup[index],5)
    if tup[index] > f.ub[0]:
        tup[index] = f.ub[0]
    if tup[index] < f.lb[0]:
        tup[index] = f.lb[0]
    return tup

def create_new_MCMC(tup, index, aaa, f):
    tup[index] = np.random.choice(aaa)
    tup[index] = round(tup[index],5)
    return tup


################################# DANTE alghorithm ################################

#######################################################
class DANTE:
    def __init__(self, exploration_weight=None, f = None, name = None):
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        
        print(self.exploration_weight)
        self.f = f
        self.name = name

    def choose(self, node):
        "Choose the best successor of node."
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

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

        media_node = max(self.children[node], key=uct)

        if uct(media_node) > uct(node):
            return media_node
        return node

    def do_rollout(self, node):
        """Make the tree one layer better. (Train for one iteration.)"""
        self._expand(node)
        self._backpropagate(path=node)

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        action = [p for p in range(0, len(node.tup))]
        self.children[node] = node.find_children(action, self.f)

    def _backpropagate(self, path):
        """Send the reward back up to the ancestors of the leaf"""
        self.N[path] += 1

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
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

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
        
    def find_children(board,action,f):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        return {
            board.make_move(i, f) for i in action}

    def is_terminal(board):
        return board.terminal

    def make_move(board, index, f):
        tup = list(board.tup)
        aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn).round(5)
        
        tup = create_new(tup, index, aaa, f)
            
        value = round(f(np.array(tup)),10)
        is_terminal = False
        return opt_task(tuple(tup), value, is_terminal)
















