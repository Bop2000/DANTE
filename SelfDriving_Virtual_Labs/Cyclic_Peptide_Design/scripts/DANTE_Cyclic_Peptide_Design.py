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

# Set the random seed for reproducibility
# random.seed()

pdb = '4ib5.pdb' # Define the pdb file. Here take the 4ib5 as an example
seq_len = 13 # Define the length of cyclic peptide here.


seq_folder = './result/DANTE_peptide_' + pdb[:4] + '_' + time.strftime("%Y-%m-%d_%H-%M", time.localtime())

if not os.path.exists(seq_folder):
# If it doesn't exist, create it
    os.makedirs(seq_folder)

def get_native_value(pdb_file):
    # get the rosetta metrics of native complex
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

native_value = get_native_value(pdb)
print(native_value)
print('native target: ', float(native_value['SHAPE COMPLEMENTARITY VALUE']) * float(nature_value['INTERFACE DELTA SASA']) / 100)


class DANTE:
    def __init__(self, exploration_weight=1):
        self.N = defaultdict(int)  # total visit count for each node
        self.Nstay = 0
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"

        print(f'number of visit is {self.N[node]}')
        log_N_vertex = math.log(self.N[node])
        def uct(n):
            "modified Upper confidence bound for trees"
            uct_value = n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n]+1))
            return uct_value

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
        """Make the tree one layer better. (Train for one iteration.)"""
        self._expand(node)
        self._backpropagate(path=node)

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        action = [p for p in range(0, len(node.tup))]
        self.children[node] = node.find_children(action)

    def _backpropagate(self, path):
        """Send the reward back up to the ancestors of the leaf"""
        self.N[path] += 1

class Node(ABC):
    """
    A representation of a single board state.
    DANTE works by constructing a tree of these Nodes.
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

_OT = namedtuple("opt_task", "tup value terminal") # tup is the initial seq
class opt_task(_OT, Node):
    def find_children(board,action):
        if board.terminal:
            return set()
        all_tup=[]
        for index in action:
            tup = list(board.tup)
            flip = random.randint(0,5)
            if flip <= 3:
                while True:
                    mutation = random.randint(0,19)
                    if mutation != tup[index]:
                        break
                tup[index] = mutation
            elif flip:
                while True:
                    new_tup = [random.randint(0, 19) for _ in range(seq_len)]
                    if new_tup != tup:
                        break
                tup = new_tup
            print(tup, int2aa(tup))
            all_tup.append(tup)

        all_value = []
        for seq in all_tup:
            metrics = get_value(seq_folder, seq, pdb)
            all_value.append(float(metrics['interface_sc']) * float(metrics['interface_dSASA']) / 100)
        is_terminal=False
        return {opt_task(tuple(t), v, is_terminal) for t, v in zip(all_tup, all_value)}

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
    metrics = get_value(seq_folder, initial_X, pdb)
    values = float(metrics['SHAPE COMPLEMENTARITY VALUE']) * float(metrics['INTERFACE DELTA SASA']) / 100
    print(initial_X, values)
    exp_weight = 0.8 * values
    board_uct = opt_task(tup=tuple(initial_X), value=values, terminal=False)
    rollout_round = 15
    tree_ubt = DANTE(exploration_weight=exp_weight)
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


def run():
    x_current_top = [random.randint(0, 19) for _ in range(seq_len)] # random start sequence
    y_top=[]
    X_top=[]

    x, y_0 = single_run(x_current_top,2)
    y_top.append(y_0)
    X_top.append(x)

    for i in x:
        new_x, new_y = single_run(i, 2) # the new root node is the sequence with highest value in last round
        X_top.append(new_x)
        y_top.append(new_y)

    return X_top,y_top

X_top,y_top = run()
print(X_top)
print(y_top)
