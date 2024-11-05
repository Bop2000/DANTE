import numpy as np
from dataset import * 
from surrogate_model import *
from collections import namedtuple
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import random
import copy

dataset = NasBench('./nasbench_dataset')
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
CANONICAL_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

def mutate_spec(old_spec, mutation_rate=1.0):
  """Computes a valid mutated spec from the old_spec."""
  new_matrix = copy.deepcopy(old_spec.original_matrix)
  new_ops = copy.deepcopy(old_spec.original_ops)
  while True:

    # In expectation, V edges flipped (note that most end up being pruned).
    edge_mutation_prob = mutation_rate / new_matrix.shape[0]
    for src in range(0, new_matrix.shape[0] - 1):
      for dst in range(src + 1, new_matrix.shape[0]):
        if random.random() < edge_mutation_prob:
          new_matrix[src, dst] = 1 - new_matrix[src, dst]
          
    # In expectation, one op is resampled.
    op_mutation_prob = mutation_rate / 3
    for ind in range(1, len(new_ops) - 1):
      if random.random() < op_mutation_prob:
        available = [o for o in CANONICAL_OPS if o != new_ops[ind]]
        new_ops[ind] = random.choice(available)
    
    try:
        cell = ModelSpec(new_matrix, new_ops)
        if dataset.is_valid(cell):
            return cell
    except:
        continue

def get_spec(adj_indxs, op_indxs):
    """
    Construct a NASBench spec from adjacency matrix and op indicators
    """
    op_names = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    ops = ['input']
    ops.extend([op_names[i] for i in op_indxs])
    ops.append('output')
    iu = np.triu_indices(7, k=1)
    adj_matrix = np.zeros((7, 7), dtype=np.int32)
    adj_matrix[(iu[0][adj_indxs], iu[1][adj_indxs])] = 1
    try:
        spec = ModelSpec(adj_matrix, ops)
        if dataset.is_valid(spec):
            return spec
    except:
        return None

def get_indxs(adj_matrix, ops):
    """
    Extract adjacency indices and operation indices from the adjacency matrix and ops.
    """
    # Define the operation names in the same order as in the get_spec function
    op_names = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    op_indxs = [op_names.index(op) for op in ops[1:-1]]

    # Extract operation indices from the ops list
    x_op = np.zeros((3, 5), dtype=int)
    for i, op_idx in enumerate(op_indxs):
        x_op[op_idx, i] = 1

    # Extract adjacency indices from the adjacency matrix
    iu = np.triu_indices(7, k=1)  # Get the upper triangular indices excluding the diagonal
    adj_indxs = np.zeros(21)
    adj_indxs[np.where(adj_matrix[iu] == 1)[0]] = 1  # Get the indices where there's a connection (1)
    combined_array = np.concatenate([adj_indxs, x_op.flatten()])

    return combined_array


def evaluate_x(x, model):
    """
    Evaluate NASBench on the model defined by x.

    x is a 36-d array.
    The first 21 are for the adjacency matrix. Largest entries will have the
    corresponding element in the adjacency matrix set to 1, with as many 1s as
    possible within the NASBench model space.
    The last 15 are for the ops in each of the five NASBench model components.
    One-hot encoded for each of the 5 components, 3 options.
    """
    assert len(x) == 36
    x_adj = x[:21]
    x_op = x[-15:]
    x_ord = x_adj.argsort()[::-1]
    op_indxs = x_op.reshape(3, 5).argmax(axis=0).tolist()
    last_good = None
    for i in range(1, 22):
        model_spec = get_spec(x_ord[:i], op_indxs)
        if model_spec is not None:
            last_good = model_spec
            break
    if last_good is None:
        # Could not get a valid spec from this x. Return bad metric values.
        return None, 0.80
    acc = model.predict(np.array(last_good.encoding()).reshape(1,-1,1)).flatten()[0]
    return last_good, acc

def _evaluate_x(x, model):
    """
    Evaluate NASBench on the model defined by x.

    x is a 36-d array.
    The first 21 are for the adjacency matrix. Largest entries will have the
    corresponding element in the adjacency matrix set to 1, with as many 1s as
    possible within the NASBench model space.
    The last 15 are for the ops in each of the five NASBench model components.
    One-hot encoded for each of the 5 components, 3 options.
    """
    assert len(x) == 36
    x_adj = x[:21]
    x_op = x[-15:]
    x_ord = x_adj.argsort()[::-1]
    op_indxs = x_op.reshape(3, 5).argmax(axis=0).tolist()
    last_good = None
    for i in range(1, 22):
        model_spec = get_spec(x_ord[:i], op_indxs)
        if model_spec is not None:
            last_good = model_spec
            break
    if last_good is None:
        # Could not get a valid spec from this x. Return bad metric values.
        return 0.80
    acc = model.predict(np.array(last_good.encoding()).reshape(1,-1,1)).flatten()[0]
    return acc


class NASBenchRunner:
    """
    A runner for non-Ax methods.
    Assumes method MINIMIZES.
    """
    def __init__(self, model):
        # For tracking iterations
        self.fs = []
        self.model = model

    def f(self, x):
        test_acc = _evaluate_x(x, self.model)
        self.fs.append(test_acc)  # Store the true, not-negated value
        return -test_acc  # ASSUMES METHOD MINIMIZES

class Optimizer(object):
    def __init__(self, method, func, samples, init_cells, init_X, init_y, config):
        self.method = method
        self.func = func
        self.samples = samples
        self.init_cells = init_cells
        self.init_X = init_X
        self.init_y = init_y
        self.config = config
        self.result = self.init_y
        self.random_seed = self.config['random_seed']

    def run(self):
        if self.method == 'random':
            for i in range(self.samples):
                cell = dataset.random_spec()
                self.result.append(dataset.query(cell)[-1])

        if self.method == 'lamcts':
            from lamcts import MCTS
            for i in range(self.samples // self.config['samples_per_round']):
                print("")
                print("="*20)
                print(f"run iteration {i}!")
                print("="*20)
                print("")
                y = []
                r = NASBenchRunner(self.func)
                X = []
                fX = []
                for i in range(self.config['samples_per_round']):
                    agent = MCTS(
                            lb = np.array([0] * 36),              # the lower bound of each problem dimensions
                            ub = np.array([1] * 36),              # the upper bound of each problem dimensions
                            dims = 36,          # the problem dimensions
                            ninits = 40,      # the number of random samples used in initializations 
                            func = r.f,               # function object to be optimized
                            Cp = 0.1
                            )
                    agent.search(iterations=100)
                    xopt = agent.curt_best_sample
                    X.append(xopt)
                    fX.append(agent.curt_best_value)
                X = np.array(X)
                fX = np.array(fX)
                x_top = X
                cells = []
                X = []
                y = []
                for i in range(x_top.shape[0]):
                    cell, _ = evaluate_x(x_top[i, :], self.func)
                    cells.append(cell)
                    X.append(cell.encoding())
                    if cell is None:
                        y.append(0.80)
                    else:
                        y.append(dataset.fixed_statistics[cell.module_hash][1])
                self.init_cells.extend(cells)
                self.init_X = np.concatenate([self.init_X, X], axis=0)
                self.init_y = np.append(self.init_y, y)
                model = SurrogateModel(name=f'{self.method}/result/{self.random_seed}')
                self.func = model.fit(self.init_X, self.init_y, verbose=True)
                self.result = np.array(self.init_y)
                np.save(f'{self.method}/result/result_{self.random_seed}.npy', self.result)
            self.result = self.init_y


        if self.method == 'da':
            from scipy.optimize import dual_annealing
            for i in range(self.samples // self.config['samples_per_round']):
                print("")
                print("="*20)
                print(f"run iteration {i}!")
                print("="*20)
                print("")
                y = []
                r = NASBenchRunner(self.func)
                lw = [0] * 36
                up = [1] * 36
                X = []
                fX = []
                for i in range(self.config['samples_per_round']):
                    ret = dual_annealing(r.f, bounds=list(zip(lw, up)), maxiter=5, maxfun=300)
                    X.append(ret.x)
                    fX.append(ret.fun)
                X = np.array(X)
                fX = np.array(fX)
                sorted_indices = np.argsort(fX)
                top_indices = sorted_indices[:self.config['samples_per_round']]
                x_top = X[top_indices, :].reshape(self.config['samples_per_round'],36)
                cells = []
                X = []
                y = []
                for i in range(x_top.shape[0]):
                    cell, _ = evaluate_x(x_top[i, :], self.func)
                    cells.append(cell)
                    X.append(cell.encoding())
                    if cell is None:
                        y.append(0.80)
                    else:
                        y.append(dataset.fixed_statistics[cell.module_hash][1])
                self.init_cells.extend(cells)
                self.init_X = np.concatenate([self.init_X, X], axis=0)
                self.init_y = np.append(self.init_y, y)
                model = SurrogateModel(name=f'{self.method}/result/{self.random_seed}')
                self.func = model.fit(self.init_X, self.init_y, verbose=True)
                self.result = np.array(self.init_y)
                np.save(f'{self.method}/result/result_{self.random_seed}.npy', self.result)
            self.result = self.init_y

        if self.method == 'cma':
            import cma
            for i in range(self.samples // self.config['samples_per_round']):
                print("")
                print("="*20)
                print(f"run iteration {i}!")
                print("="*20)
                print("")
                y = []
                r = NASBenchRunner(self.func)
                options = {'bounds':[0, 1], 'maxfevals':300}
                ind = np.argsort(self.init_y)
                current_y = max(self.init_y)
                cell_top = [self.init_cells[ind[-self.config['samples_per_round']:][i]] for i in range(self.config['samples_per_round'])]
                X = []
                fX = []
                for i in range(self.config['samples_per_round']):
                    try:
                        x_current = get_indxs(cell_top[-i-1].original_matrix, cell_top[-i-1].original_ops)
                    except:
                        x_current = np.random.random(36)
                    xopt, es = cma.fmin2(r.f, x_current, (1-0)/4, options)
                    print(xopt, xopt.shape)
                    X.append(xopt)
                    fX.append(es.result[1])
                X = np.array(X)
                fX = np.array(fX)
                x_top = X
                cells = []
                X = []
                y = []
                for i in range(x_top.shape[0]):
                    cell, _ = evaluate_x(x_top[i, :], self.func)
                    cells.append(cell)
                    X.append(cell.encoding())
                    if cell is None:
                        y.append(0.80)
                    else:
                        y.append(dataset.fixed_statistics[cell.module_hash][1])
                self.init_cells.extend(cells)
                self.init_X = np.concatenate([self.init_X, X], axis=0)
                self.init_y = np.append(self.init_y, y)
                model = SurrogateModel(name=f'{self.method}/result/{self.random_seed}')
                self.func = model.fit(self.init_X, self.init_y, verbose=True)
                self.result = np.array(self.init_y)
                np.save(f'{self.method}/result/result_{self.random_seed}.npy', self.result)
            self.result = self.init_y

        if self.method == 'turbo':
            from turbo import TurboM
            for i in range(self.samples // self.config['samples_per_round']):
                print("")
                print("="*20)
                print(f"run iteration {i}!")
                print("="*20)
                print("")
                y = []
                r = NASBenchRunner(self.func)
                turbo_m = TurboM(
                    f=r.f,  # Handle to objective function
                    lb=np.zeros(36),
                    ub=np.ones(36),
                    n_init=50,  # Number of initial bounds from an Symmetric Latin hypercube design
                    max_evals=300,  # Maximum number of evaluations
                    n_trust_regions=5,  # Number of trust regions
                    batch_size=10,  # How large batch size TuRBO uses
                    verbose=True,  # Print information from each batch
                    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                    device="cpu",  # "cpu" or "cuda"
                    dtype="float64",  # float64 or float32
                )
                turbo_m.optimize()
                X = turbo_m.X  # Evaluated points
                fX = turbo_m.fX.flatten()  # Observed values
                sorted_indices = np.argsort(fX)
                top_indices = sorted_indices[:self.config['samples_per_round']]
                x_top = X[top_indices, :].reshape(self.config['samples_per_round'],36)
                cells = []
                X = []
                y = []
                for i in range(x_top.shape[0]):
                    cell, _ = evaluate_x(x_top[i, :], self.func)
                    cells.append(cell)
                    X.append(cell.encoding())
                    if cell is None:
                        y.append(0.80)
                    else:
                        y.append(dataset.fixed_statistics[cell.module_hash][1])
                self.init_cells.extend(cells)
                self.init_X = np.concatenate([self.init_X, X], axis=0)
                self.init_y = np.append(self.init_y, y)
                model = SurrogateModel(name=f'{self.method}/result/{self.random_seed}')
                self.func = model.fit(self.init_X, self.init_y, verbose=True)
                self.result = np.array(self.init_y)
                np.save(f'{self.method}/result/result_{self.random_seed}.npy', self.result)
            self.result = self.init_y
        
        if self.method == 'mcmc':
            for i in range(self.samples // self.config['samples_per_round']):
                T = self.config['T_init'] * (np.exp(np.log(0.5) / self.config['half_life']) ** i) 
                print("")
                print("="*20)
                print(f"run iteration {i}!")
                print("="*20)
                print("")
                ind = np.argsort(self.init_y)
                current_y = max(self.init_y)
                cell_current_top = [self.init_cells[ind[-3:][i]] for i in range(3)][-1]
                num = 0
                while num < 20:
                    new_cell = mutate_spec(cell_current_top)
                    new_X = np.array(new_cell.encoding())
                    new_y = self.func.predict(np.array(new_X).reshape(1,-1,1))
                    delta = new_y - current_y
                    if delta > 0 or np.random.uniform() < np.exp(-delta / T):
                        self.init_cells.append(new_cell)
                        self.init_X = np.concatenate([self.init_X, new_X.reshape(1, -1)], axis=0)
                        self.init_y = np.append(self.init_y, dataset.fixed_statistics[new_cell.module_hash][1])
                        num += 1
                model = SurrogateModel(name=f'{self.method}/result/{self.random_seed}')
                self.func = model.fit(self.init_X, self.init_y, verbose=True)
                self.result = np.array(self.init_y)
                np.save(f'{self.method}/result/result_{self.random_seed}.npy', self.result)
            self.result = self.init_y

        elif self.method == 'dots':
            for i in range(self.samples // self.config['samples_per_round']):
                print("")
                print("="*20)
                print(f"run iteration {i}!")
                print("="*20)
                print("")
                y = []
                dots = DOTS(self.func, self.init_cells, self.init_X, self.init_y, self.config)
                cells, X = dots.run()
                for cell in cells:
                    y.append(dataset.fixed_statistics[cell.module_hash][1])
                y = np.array(y)
                self.init_cells.extend(cells)
                self.init_X = np.concatenate([self.init_X, X], axis=0)
                self.init_y = np.append(self.init_y, y)
                model = SurrogateModel(name=f'{self.method}/result/{self.random_seed}')
                self.func = model.fit(self.init_X, self.init_y, verbose=True)
                self.result = np.array(self.init_y)
                np.save(f'{self.method}/result/result_{self.random_seed}.npy', self.result)
            self.result = self.init_y

        
        return self.result


#########################################################



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
        new_x = []
        boards_X = []
        for board in boards:
            boards_X.append(list(board.tup))
        boards_X = np.array(boards_X)
        for i in range(boards_X.shape[0]):
          temp_x = np.array(boards_X[i, :])
          same = np.all(temp_x==X, axis=1)
          has_true = any(same)
          if has_true == False:
            new_x.append(temp_x)
            new_board.append(boards[i])
        new_x = np.array(new_x)
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
        board_topN = [childrens[ind[i]] for i in range(top_n)]
        return board_topN

    def single_rollout(self,X,rollout_round,board_uct,num_list=[5,1,1]):
        boards = []
        boards_rand = []
        for i in range(0, rollout_round):
            self.do_rollout(board_uct)
            board_uct,board_rand = self.choose(board_uct)
            boards.append(board_uct)
            boards_rand.extend(board_rand)

        #visit nodes
        board_most_visit =  self.most_visit_node(num_list[1])

        #highest pred value nodes and random nodes
        _ = self.data_process(X,boards)
        top_n = num_list[0]
        if len(_) >= top_n:
            new_board = _
        else:
            new_board = boards
        new_x = []
        for board in new_board:
            new_x.append(np.array(board.tup))
        new_x = np.array(new_x)
        new_pred = self.func.predict(np.array(new_x).reshape(len(new_x),-1,1))
        new_pred = np.array(new_pred).reshape(len(new_x))
        new_rands = self.data_process(X,boards_rand)
        ind = np.argsort(new_pred)[-top_n:]
        top_X =  new_x[ind]
        if len(new_board) >= top_n:
            top_board = [new_board[ind[i]] for i in range(top_n)]
        else:
            top_board = new_board
        board_rand2 = [new_rands[random.randint(0, len(new_rands)-1)] for i in range(num_list[2])]
        top_board.extend(board_rand2)
        top_board.extend(board_most_visit)

        return top_board

    def run(self):
        if self.config['rollout_round'] % 100 < 80:
            UCT_low=False
        else:
            UCT_low=True

        #### make sure unique initial points
        ind = np.argsort(self.init_y)
        cell_current_top = [self.init_cells[ind[-4:][i]] for i in range(4)]
        x_current_top = self.init_X[ind[-4:]]

        ### starting rollout
        X_top=[]
        cell_top = []
        for i in range(len(cell_current_top)):
            initial_X = x_current_top[i]
            values = max(self.init_y)
            exp_weight = self.config['exploration_weight'] * abs(values)
            if UCT_low ==True:
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
        reward = node.reward(self.func)
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
            new_ops = copy.deepcopy(board.cell.original_ops)
            if   flip ==0:
                while True:
                    for src in range(0, new_matrix.shape[0]-1):
                        for dst in range(src+1, new_matrix.shape[0]):
                            if random.random() < mutation_rate / new_matrix.shape[0]:
                                new_matrix[src, dst] = 1 - new_matrix[src, dst]
                    try:
                        cell = ModelSpec(new_matrix, new_ops)
                        if dataset.is_valid(cell):
                            break
                    except:
                        continue
            elif flip ==1:
                while True:
                    for i in range(1, len(new_ops)-1):
                        if random.random() < mutation_rate / 3:
                            available = [o for o in CANONICAL_OPS if o != new_ops[i]]
                            new_ops[i] = random.choice(available)
                    try:
                        cell = ModelSpec(new_matrix, new_ops)
                        if dataset.is_valid(cell):
                            break
                    except:
                        continue
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
                    try:
                        cell = ModelSpec(new_matrix, new_ops)
                        if dataset.is_valid(cell):
                            break
                    except:
                        continue
            elif flip ==3:
                cell = dataset.random_spec()
            all_cell.append(cell)
            tup = np.array(cell.encoding())
            all_tup.append(tup)
        all_value = model.predict(np.array(all_tup).reshape(len(all_tup),-1,1))
        is_terminal=False
        try:
            task = {opt_task(c, tuple(t), v, is_terminal) for c, t, v in  zip(all_cell,all_tup,all_value)}
        except:
            task = {opt_task(c, tuple(t), v[0], is_terminal) for c, t, v in  zip(all_cell, all_tup,all_value)}
        return  task

    def reward(board,model):
        print(np.array(board.tup).reshape(1,-1,1).shape)
        values = model.predict(np.array(board.tup).reshape(1,-1,1))
        values = float(np.array(values).reshape(1))
        print(values)
        return values
    def is_terminal(board):
        return board.terminal