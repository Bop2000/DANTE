import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import random

################################# DOTS alghorithm ################################

#######################################################
'DOTS'
#######################################################
class DOTS:
    def __init__(self, exploration_weight=None, f = None, name = None):
        self.Q = defaultdict(int)  # total reward of each node
        self.Qa = dict.fromkeys(list(range(20)), 0)
        self.Na = dict.fromkeys(list(range(20)), 1)
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

        action = [p for p in range(0, len(node.tup))]
        self.children[node] = node.find_children(action,self.f)  
        
        random_num=np.random.randint(0,10)
        if random_num<8:
            media_node = max(self.children[node], key=uct)
        else:
            rand_index = random.randint(0, len(list(self.children[node]))-1)
            media_node = list(self.children[node])[rand_index]

        if uct(media_node) * (1 + 0.05 * np.random.random()) > uct(node):
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
            if count == 5:
                return path
          
            if unexplored:
              path.append(max(unexplored, key=evaluate))#
              return path
            node = self._uct_select(node)  # descend a layer deeper
            count+=1

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        # if node in self.children:
        #     return  # already expanded
        action = [p for p in range(0, len(node.tup))]
        self.children[node] = node.find_children(action, self.f)

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        # reward = node.reward()
        reward = node.reward(self.f)
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
        
    def find_children(board,action,f):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        return {
            board.make_move(i, f) for i in action}

    def find_random_child(board):
        pass

    def find_uct_child(board, action):
        pass

    def reward(board,f):
        return board.value

    def is_terminal(board):
        return board.terminal

    def make_move(board, index, f):
        tup = list(board.tup)
        flip = random.randint(0,3)
        if flip ==0:
          tup[index] += f.turn
        elif flip ==1:
          tup[index] -= f.turn
        else:
            aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn)
            tup[index] = np.random.choice(aaa)
            # tup[index] = random.randint(-50, 50)/10
        tup[index] = round(tup[index],5)
        if tup[index] > f.ub[0]:
            tup[index] = f.ub[0]
        if tup[index] < f.lb[0]:
            tup[index] = f.lb[0]
        value = round(1000/(f(np.array(tup))+0.1),10)
        is_terminal = False
        return opt_task(tuple(tup), value, is_terminal)


#######################################################
'DOTS-Greedy'
#######################################################
class DOTS_Greedy:
    def __init__(self, f = None, dims = 20):
        self.f = f
        self.turn = f.turn
        self.dims = dims
        
    def choose(self, board):
        "Choose the best successor of node. (Choose a move in the game)"
        values = []
        nodes = []
        for index in range(self.dims):
            tup = np.array(board)
            flip = random.randint(0,3)
            if flip ==0:
              tup[index] += self.turn
            elif flip ==1:
              tup[index] -= self.turn
            else:
               aaa = np.arange(self.f.lb[0], self.f.ub[0] + self.turn, self.turn)
               tup[index] = np.random.choice(aaa)
            tup[index] = round(tup[index],5)
            if tup[index] > self.f.ub[0]:
                tup[index] = self.f.ub[0]
            if tup[index] < self.f.lb[0]:
                tup[index] = self.f.lb[0]
            value = round(self.f(np.array(tup)),5)
            values.append(value)
            nodes.append(tup)
        ind = np.argmin(values)
        node = nodes[ind]
        value = values[ind]
        return node, value



#######################################################
'DOTS-Episilon-Greedy'
#######################################################
class DOTS_eGreedy:
    def __init__(self, f = None, dims = 20):
        self.f = f
        self.turn = f.turn
        self.dims = dims
        
    def choose(self, board):
        "Choose the best successor of node. (Choose a move in the game)"
        values = []
        nodes = []
        for index in range(self.dims):
            tup = np.array(board)
            flip = random.randint(0,3)
            if flip ==0:
              tup[index] += self.turn
            elif flip ==1:
              tup[index] -= self.turn
            else:
               aaa = np.arange(self.f.lb[0], self.f.ub[0] + self.turn, self.turn)
               tup[index] = np.random.choice(aaa)
            tup[index] = round(tup[index],5)
            if tup[index] > self.f.ub[0]:
                tup[index] = self.f.ub[0]
            if tup[index] < self.f.lb[0]:
                tup[index] = self.f.lb[0]
            value = round(self.f(np.array(tup)),5)
            values.append(value)
            nodes.append(tup)
        #######epsilon greedy: 80% best and 20% random
        random_num=np.random.randint(0,10)
        if random_num<8:
            ind = np.argmin(values)
        else:
            ind=np.random.randint(0,len(values))
        node = nodes[ind]
        value = values[ind]
        return node, value





















