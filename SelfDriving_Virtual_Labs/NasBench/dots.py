import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import random
from dataset import * 
import copy

dataset = NasBench('./nasbench_dataset')
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
CANONICAL_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

################################# DOTS alghorithm ################################

class DOTS:
    def __init__(self, func, init_cells, init_X, init_y, config):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.func = func
        self.init_cells = init_cells
        self.init_X = init_X
        self.init_y = init_y
        self.config = config
        self.exploration_weight = self.config['exploration_weight']

    def choose(self, node):
        "Choose the best successor of node."
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            print('not seen before, randomly sampled!')
            return node.find_random_child()

        def evaluate(n):
            return n.value  # average reward
        log_N_vertex = math.log(self.N[node])
        def uct(n):
            "Upper confidence bound for trees"
            uct_value = n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n]+1))
            return uct_value

        action = [p for p in range(0, self.config['dim'])]
        self.children[node] = node.find_children(action,self.func)

        media_node = max(self.children[node], key=uct)
        node_rand = []
        # for i in range(len(list(self.children[node]))):
        ind=np.random.randint(0,len(list(self.children[node])),2) ##for computer memory consideration, choose only 2 random nodes in one rollout
        for i in ind:
              node_rand.append(list(self.children[node])[i])

        if uct(media_node) > uct(node):
            return media_node, node_rand
        return node, node_rand

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def data_process(self,X,boards):
        new_board = []
        boards_X = []
        for board in boards:
            boards_X.append(board)
        boards_X = np.array(boards_X)
        for i in range(boards_X.shape[0]):
          temp_x = np.array(boards_X[i, :])
          same = np.all(temp_x==X, axis=1)
          has_true = any(same)
          if has_true == False:
            new_x.append(temp_x)
            new_board.append(boards[i])
        new_x= np.array(new_x)
        print(f'unique number of boards: {len(new_x)}')
        return new_board

    def most_visit_node(self,top_n):
        N_visit = self.N
        childrens = [i for i in self.children]
        children_N = []
        for child in childrens:
            children_N.append(N_visit[child])
        children_N = np.array(children_N)
        ind = np.argpartition(children_N, -top_n)[-top_n:]
        board_topN = childrens[ind]
        return board_topN

    def single_rollout(self,X,rollout_round,board_uct,num_list=[5,1,1]):
        boards = []
        boards_rand = []
        for i in range(0, rollout_round):
            self.do_rollout(board_uct)
            board_uct,board_rand = self.choose(board_uct)
            boards.append(list(board_uct))
            boards_rand.append(list(board_rand))

        #visit nodes
        board_most_visit =  self.most_visit_node(num_list[1])

        #highest pred value nodes and random nodes
        new_board = self.data_process(X,boards)
        new_x = []
        for board in new_board:
            new_x.append(np.array(board.tup))
        new_x = np.array(new_x)
        try:
            new_pred = self.func.predict(np.array(new_x))
            new_pred = np.array(new_pred).reshape(len(new_x))
        except: # for nn
            new_pred = self.func.predict(np.array(new_x).reshape(len(new_x),-1,1))
            new_pred = np.array(new_pred).reshape(len(new_x))
        new_rands = self.data_process(X,boards_rand)
        top_n = num_list[0]
        ind = np.argsort(new_pred)[-top_n:]
        top_X =  new_x[ind]
        top_board = new_board[ind]
        board_rand2 = [new_rands[random.randint(0, len(new_rands)-1)] for i in range(num_list[2])]
        top_board = top_board.extend(board_rand2)
        top_board = top_board.extedn(board_most_visit)

        return top_board

    def run(self):
        if self.config['rollout_round'] % 100 < 80:
            UCT_low=False
        else:
            UCT_low=True

        #### make sure unique initial points
        ind = np.argsort(self.init_y)
        cell_current_top = self.init_cells[ind[-3:]]
        x_current_top = self.init_X[ind[-3:]]

        ### starting rollout
        X_top=[]
        cell_top = []
        for i in range(len(cell_current_top)):
            initial_X = x_current_top[i]
            values = max(self.init_y)
            exp_weight = self.config['exploration_weight'] * abs(values)
            if UCT_low ==True:
                try:
                    values = self.model.predict(np.array(initial_X).reshape(1,-1))
                except:
                    values = self.model.predict(np.array(initial_X).reshape(1,-1,1))
                values = float(np.array(values).reshape(1))
                exp_weight = self.config['exploration_weight']*0.5*values
            self.exploration_weight = exp_weight
            board_uct = opt_task(cell=cell_current_top[i],tup=tuple(initial_X), value=values, terminal=False)
            top_board = self.single_rollout(self.init_X,self.config['rollout_round'],board_uct)
            for board in top_board:
                top_cell = board.cell
                top_X = board.tup
                cell_top.append(top_cell)
                X_top.append(top_X)

        top_cell = cell_top[:20]
        top_X = np.vstack(X_top)
        top_X = top_X[:20]
        return top_cell, top_X

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
                return path
            if unexplored:
              path.append(max(unexplored, key=evaluate))#
              return path
            node = self._uct_select(node)  # descend a layer deeper
            count+=1

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        action = [p for p in range(0, self.config['dim'])]
        self.children[node] = node.find_children(action, self.func)

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        reward = node.reward(self.model)
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

_OT = namedtuple("opt_task", "cell tup value terminal")
class opt_task(_OT, Node):
    def find_children(board,action,model):
        mutation_rate = 1
        if board.terminal:
            return set()
        all_cell = []
        all_tup = []
        for index in action:
            flip = random.randint(0,3)
            new_matrix = copy.deepcopy(board.cell.original_matrix)
            new_ops = copy.deepcopy(board.cell.orginal_ops)
            if   flip ==0:
                while True:
                    for src in range(0, new_matrix.shape[0]-1):
                        for dst in range(src+1, new_matrix.shape[0]):
                            if random.random() < mutation_rate / new_matrix.shape[0]:
                                new_matrix[src, dst] = 1 - new_matrix[src, dst]
                    cell = ModelSpec(new_matrix, new_ops)
                    if dataset.is_valid(cell):
                        break
            elif flip ==1:
                while True:
                    for i in range(1, len(new_ops)-1):
                        if random.random() < mutation_rate / 3:
                            available = [o for o in CANONICAL_OPS if o != new_ops[i]]
                            new_ops[i] = random.choice(available)
                    cell = ModelSpec(new_matrix, new_ops)
                    if dataset.is_valid(cell):
                        break
            elif flip ==2:
                while True:
                    for src in range(0, new_matrix.shape[0]-1):
                        for dst in range(src+1, new_matrix.shape[0]):
                            if random.random() < mutation_rate / new_matrix.shape[0]:
                                new_matrix[src, dst] = 1 - new_matrix[src, dst]

                    for i in range(1, len(new_ops)-1):
                        if random.random() < mutation_rate / 3:
                            available = [o for o in CANONICAL_OPS if o != new_ops[i]]
                            new_ops[i] = random.choice(available)
                    cell = ModelSpec(new_matrix, new_ops)
                    if dataset.is_valid(cell):
                        break
            elif flip ==3:
                cell = dataset.random_spec()

            all_cell.append(cell)
            tup = np.array(cell.encoding())
            all_tup.append(tup)
        try:
            all_value = model.predict(np.array(all_tup))
        except:
            all_value = model.predict(np.array(all_tup).reshape(len(all_tup),40,1))
        is_terminal=False
        try:
            task = {opt_task(c, tuple(t), v, is_terminal) for c, t, v in  zip(all_cell,all_tup,all_value)}
        except:
            task = {opt_task(c, tuple(t), v[0], is_terminal) for c, t, v in  zip(all_cell, all_tup,all_value)}
        return  task

    def reward(board,model):
        try:
            values = model.predict(np.array(board.tup).reshape(1,-1))
        except:
            values = model.predict(np.array(board.tup).reshape(1,-1,1))
        values = float(np.array(values).reshape(1))
        return values
    def is_terminal(board):
        return board.terminal
