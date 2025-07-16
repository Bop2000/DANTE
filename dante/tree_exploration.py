import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Any, Set, Optional, Dict, List


import keras
import numpy as np

from dante.obj_functions import ObjectiveFunction, BuiltInSyntheticFunction


@dataclass
class TreeExploration:
    func: ObjectiveFunction = None
    model: keras.Model = None
    N: Dict[Any, int] = field(default_factory=lambda: defaultdict(int))
    children: Dict[Any, Set] = field(default_factory=dict)
    rollout_round: int = 200
    ratio: float = 0.02 # 0.02 is suit for 'rastrigin','ackley','griewank','schwefel'
    exploration_weight: float = 0.1
    num_list: List[int] = field(default_factory=lambda: [5, 1, 1])
    num_samples_per_acquisition: int = 20

    def choose(self, node):
        """Choose the best successor of node."""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            """Upper confidence bound for trees"""
            return n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n] + 1)
            )

        media_node = max(self.children[node], key=uct)
        node_rand = [
            list(self.children[node])[i].tup
            for i in np.random.randint(0, len(self.children[node]), 2)
        ]
        # print('uct of root:',uct(node),'value of root:',node.value)
        # print('uct of best leaf:',uct(media_node),'value of best leaf:',media_node.value)

        return (
            (media_node, node_rand)
            if uct(media_node) > uct(node)
            else (node, node_rand)
        )

    def do_rollout(self, node):
        """Make the tree one layer better. (Train for one iteration.)"""
        self._expand(node)
        self._backpropagate(path=node)

    @staticmethod
    def data_process(x: np.ndarray, boards: List[list]) -> np.ndarray:
        new_x = []
        boards = np.unique(np.array(boards), axis=0)
        new_x = [board for board in boards if not np.any(np.all(board == x, axis=1))]
        # print(f"Unique number of boards: {len(new_x)}")
        return np.array(new_x)

    def most_visit_node(self, x: np.ndarray, top_n: int) -> np.ndarray:
        """Find the most visited nodes."""
        N_visit = self.N
        childrens = [i for i in self.children]
        children_N = []
        X_top = []
        for child in childrens:
            child_tup = np.array(child.tup)
            same = np.all(child_tup == x, axis=1)
            has_true = any(same)
            if not has_true:
                children_N.append(N_visit[child])
                X_top.append(child_tup)
        children_N = np.array(children_N)
        X_top = np.array(X_top)
        ind = np.argpartition(children_N, -top_n)[-top_n:]
        X_topN = X_top[ind]
        return X_topN

    def single_rollout(self, X, board_uct, num_list: List[float]):
        """Perform a single rollout."""
        boards = []
        boards_rand = []
        for _ in range(0, self.rollout_round):
            self.do_rollout(board_uct)
            board_uct, board_rand = self.choose(board_uct)
            boards.append(list(board_uct.tup))
            boards_rand.append(list(board_rand))

        # visit nodes
        X_most_visit = self.most_visit_node(X, num_list[1])

        # highest pred value nodes and random nodes
        new_x = self.data_process(X, boards)
        try:
            new_pred = self.model.predict(
                np.array(new_x).reshape(len(new_x), -1, 1), verbose=False
            )
            new_pred = np.array(new_pred).reshape(len(new_x))
        except:
            pass
        boards_rand = np.vstack(boards_rand)
        new_rands = self.data_process(X, boards_rand)
        top_n = num_list[0]
        if len(new_x) >= top_n:
            ind = np.argsort(new_pred)[-top_n:]
            top_X = new_x[ind]
            X_rand2 = [
                new_rands[random.randint(0, len(new_rands) - 1)]
                for i in range(num_list[2])
            ]
        elif len(new_x) == 0:
            new_pred = self.model.predict(
                np.array(new_rands).reshape(len(new_rands), -1, 1), verbose=False
            ).reshape(-1)
            ind = np.argsort(new_pred)[-top_n:]
            top_X = new_rands[ind]
            X_rand2 = [
                new_rands[random.randint(0, len(new_rands) - 1)]
                for i in range(num_list[2])
            ]
        else:
            top_X = np.array(new_x)
            num_random = num_list[0] + num_list[2] - len(top_X)
            X_rand2 = [
                new_rands[random.randint(0, len(new_rands) - 1)]
                for i in range(num_random)
            ]
        try:
            top_X = np.concatenate([X_most_visit, top_X, X_rand2])
        except:
            top_X = np.concatenate([X_most_visit, top_X])

        return top_X

    def rollout(
        self,
        x: np.ndarray,
        y: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        """Perform rollout based on the function type."""
        if self.func.name == 'rosenbrock':
            self.ratio = 0.1

        if self.func.name in [
            'rastrigin',
            'levy',
        ]:
            return self._rollout_for_specific_functions(x, y)
        else:
            return self._rollout_for_other_functions(x, y, iteration)

    def _rollout_for_specific_functions(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        index_max = np.argmax(y)
        print(max(y))
        initial_x = x[index_max, :]
        values = float(
            self.model.predict(initial_x.reshape(1, -1, 1), verbose=False).reshape(1)
        )
        board_uct = OptTask(tup=tuple(initial_x), value=values, terminal=False)
        self.exploration_weight = self.ratio * abs(max(y))
        num_list = (
            [18, 2, 0]
            if self.func.name == 'rastrigin'
            else [15, 3, 2]
        )
        return self.single_rollout(x, board_uct, num_list=num_list)

    def _rollout_for_other_functions(
        self,
        x: np.ndarray,
        y: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        self.rollout_round = 100
        UCT_low = iteration % 100 >= 80
        x_current_top = self._get_unique_top_points(x, y)
        x_top = []
        for initial_X in x_current_top:
            values = float(
                self.model.predict(
                    initial_X.reshape(1, -1, 1), verbose=False
                ).reshape(1)
            )
            exp_weight = self.ratio * abs(max(y))
            if UCT_low:
                exp_weight = self.ratio * 0.5 * abs(max(y))
            self.exploration_weight = exp_weight
            board_uct = OptTask(tup=tuple(initial_X), value=values, terminal=False)
            x_top.append(self.single_rollout(x, board_uct, self.num_list))
        return np.vstack(x_top)[: self.num_samples_per_acquisition]

    def _get_unique_top_points(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        ind = np.argsort(y)
        x_current_top = X[ind[-3:]]
        x_current_top = np.unique(x_current_top, axis=0)
        i = -4
        while len(x_current_top) < 3:
            x_current_top = np.concatenate((x_current_top, X[ind[i]].reshape(1, -1)))
            x_current_top = np.unique(x_current_top, axis=0)
            i -= 1
        return x_current_top

    def _expand(self, node):
        """Update the `children` dict with the children of `node`"""
        action = list(range(len(node.tup)))
        self.children[node] = node.find_children(
            node, action, self.func, self.model
        )

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
        """All possible successors of this board state"""
        return set()

    @abstractmethod
    def is_terminal(self):
        """Returns True if the node has no children"""
        return True

    @abstractmethod
    def __hash__(self):
        """Nodes must be hashable"""
        return 123456789

    @abstractmethod
    def __eq__(self, node2):
        """Nodes must be comparable"""
        return True


_OT = namedtuple("OptTask", "tup value terminal")


class OptTask(_OT, Node):
    """Represents an optimization task node in the search tree."""

    @staticmethod
    def find_children(board, action, func, model):
        """Find all possible child nodes for the current board state."""
        if board.terminal:
            return set()

        all_tuples = OptTask._generate_child_tuples(board, action, func)
        all_values = model.predict(
            np.array(all_tuples).reshape(len(all_tuples), func.dims, 1), verbose=False
        )

        return {OptTask(tuple(t), v[0], False) for t, v in zip(all_tuples, all_values)}

    @staticmethod
    def _generate_child_tuples(board, action, func):
        """Generate child tuples based on the current board state and function parameters."""
        turn = func.turn
        possible_values = np.arange(
            func.lb[0], func.ub[0] + func.turn, func.turn
        ).round(5)
        all_tuples = []

        for index in action:
            tup = list(board.tup)
            OptTask._apply_random_modification(
                tup, index, turn, possible_values, func.dims
            )
            tup = OptTask._clip_to_bounds(np.array(tup), func.lb[0], func.ub[0])
            all_tuples.append(tup)

        return all_tuples

    @staticmethod
    def _apply_random_modification(tup, index, turn, possible_values, dims):
        """Apply a random modification to the tuple."""
        flip = random.randint(0, 5)
        if flip == 0:
            tup[index] += turn
        elif flip == 1:
            tup[index] -= turn
        elif flip in (2, 3):
            num_changes = int(dims / 5) if flip == 2 else int(dims / 10)
            for _ in range(num_changes):
                random_index = random.randint(0, len(tup) - 1)
                tup[random_index] = np.random.choice(possible_values)
        elif flip in (4, 5):
            tup[index] = np.random.choice(possible_values)

        tup[index] = round(tup[index], 5)

    @staticmethod
    def _clip_to_bounds(tup, lower_bound, upper_bound):
        """Clip the tuple values to the given bounds."""
        return np.clip(tup, lower_bound, upper_bound)

    def is_terminal(self):
        """Check if the current board state is terminal."""
        return self.terminal
