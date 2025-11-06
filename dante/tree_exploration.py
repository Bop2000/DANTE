import math
import os
import time
import random
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import keras
import numpy as np

from dante.obj_functions import ObjectiveFunction


@dataclass
class TreeExploration:
    func: ObjectiveFunction
    model: keras.Model
    exploration_weight: float = 0.1
    rollout_round: int = 200
    ratio: float = 0.02
    num_samples_per_acquisition: int = 20
    num_list: Tuple[int, int, int] = (5, 1, 1)
    # 新增：每轮用作根节点的数量（仅非特殊函数，如 rosenbrock 等）
    root_points: int = 3
    # 新增：是否在每个 100 轮迭代的后 20% 衰减探索权重
    attenuate_late: bool = True
    N: Dict["OptTask", int] = field(init=False)
    Q: Dict["OptTask", float] = field(init=False)
    children: Dict["OptTask", Set["OptTask"]] = field(init=False)
    unexpanded: Dict["OptTask", List["OptTask"]] = field(init=False)
    visited_nodes: Optional[Set[Tuple[float, ...]]] = None
    new_nodes: Set[Tuple[float, ...]] = field(init=False)

    def __post_init__(self) -> None:
        self._debug = os.getenv("DANTE_DEBUG_TIMING", "0") not in ("0", "", "false", "False")
        self.N = defaultdict(int)
        self.Q = defaultdict(float)
        self.children = {}
        self.unexpanded = {}
        if self.visited_nodes is None:
            self.visited_nodes = set()
        self._initial_visited = set(self.visited_nodes)
        self._local_seen: Set[Tuple[float, ...]] = set(self.visited_nodes)
        self.new_nodes = set()
        if self.func.name in {"rosenbrock", "schwefel"}:
            self.num_list = (15, 3, 2)

    def choose(self, node: "OptTask") -> Tuple["OptTask", List[Tuple[float, ...]]]:
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children or not self.children[node]:
            self._expand(node)
        if node not in self.children or not self.children[node]:
            raise RuntimeError("No children available for expansion.")

        log_n = math.log(self.N[node] + 1)

        def uct(n: "OptTask") -> float:
            return n.value + self.exploration_weight * math.sqrt(log_n / (self.N[n] + 1))

        candidates = list(self.children[node])
        best_child = max(candidates, key=uct)
        rand_size = min(2, len(candidates))
        rand_indices = np.random.choice(len(candidates), size=rand_size, replace=False)
        random_nodes = [candidates[idx].tup for idx in rand_indices]

        if uct(best_child) > uct(node):
            return best_child, random_nodes
        return node, random_nodes

    def do_rollout(self, node: "OptTask") -> None:
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def data_process(self, x: np.ndarray, boards: List[List[float]]) -> np.ndarray:
        if not boards:
            return np.empty((0, x.shape[1]), dtype=float)
        boards_array = np.unique(np.array(boards, dtype=float), axis=0)
        new_points: List[np.ndarray] = []
        for candidate in boards_array:
            if np.any(np.all(np.isclose(x, candidate, atol=1e-8), axis=1)):
                continue
            tup = tuple(np.round(candidate, 6))
            if tup in self._local_seen:
                continue
            new_points.append(candidate)
            # 延后记录：仅在最终候选确定后再写入 _local_seen/new_nodes
        return np.array(new_points, dtype=float) if new_points else np.empty((0, x.shape[1]), dtype=float)

    def most_visit_node(self, x: np.ndarray, top_n: int) -> np.ndarray:
        candidates: List[np.ndarray] = []
        visits: List[int] = []
        for child, visit in self.N.items():
            tup = np.array(child.tup, dtype=float)
            if np.any(np.all(np.isclose(x, tup, atol=1e-8), axis=1)):
                continue
            tup_key = tuple(np.round(tup, 6))
            if tup_key in self._local_seen:
                continue
            candidates.append(tup)
            visits.append(visit)
            # 延后记录到最终候选集合时再进行
        if not candidates or top_n <= 0:
            return np.empty((0, x.shape[1]), dtype=float)
        visits = np.array(visits)
        candidates = np.array(candidates)
        indices = np.argpartition(visits, -top_n)[-top_n:]
        return candidates[indices]

    def single_rollout(self, X: np.ndarray, board: "OptTask", num_list: Tuple[int, int, int]) -> np.ndarray:
        boards: List[List[float]] = []
        random_boards: List[List[Tuple[float, ...]]] = []
        current = board
        for step in range(self.rollout_round):
            t0 = time.perf_counter()
            self.do_rollout(current)
            t1 = time.perf_counter()
            current, random_nodes = self.choose(current)
            t2 = time.perf_counter()
            if self._debug and (t1 - t0 > 2.0 or t2 - t1 > 2.0):
                print(
                    f"[DEBUG] rollout step {step+1}/{self.rollout_round}: do_rollout={t1-t0:.2f}s, choose={t2-t1:.2f}s",
                    flush=True,
                )
            boards.append(list(current.tup))
            random_boards.append(random_nodes)
            if (step + 1) % max(1, self.rollout_round // 5) == 0:
                print(
                    f"[TreeExploration] rollout progress "
                    f"{step + 1}/{self.rollout_round} "
                    f"(current value={current.value:.5f})"
                )

        X_most_visit = self.most_visit_node(X, num_list[1])

        new_x = self.data_process(X, boards)
        try:
            new_pred = self.model.predict(new_x.reshape(len(new_x), -1, 1), verbose=False)
            new_pred = np.array(new_pred).reshape(len(new_x))
        except Exception:
            new_pred = np.array([])

        flat_random: List[List[float]] = [list(node) for nodes in random_boards for node in nodes]
        new_random = self.data_process(X, flat_random)

        top_n, _, extra = num_list
        if len(new_x) >= top_n and len(new_x) > 0:
            indices = np.argsort(new_pred)[-top_n:]
            top_x = new_x[indices]
            random_samples = [
                new_random[random.randint(0, len(new_random) - 1)]
                for _ in range(extra)
            ] if len(new_random) else []
            random_samples = np.array(random_samples, dtype=float) if random_samples else np.empty((0, X.shape[1]), dtype=float)
        elif len(new_x) == 0 and len(new_random) > 0:
            new_pred_random = self.model.predict(
                new_random.reshape(len(new_random), -1, 1), verbose=False
            ).reshape(-1)
            indices = np.argsort(new_pred_random)[-top_n:]
            top_x = new_random[indices]
            random_samples = [
                new_random[random.randint(0, len(new_random) - 1)]
                for _ in range(extra)
            ]
            random_samples = np.array(random_samples, dtype=float)
        else:
            top_x = new_x
            need = top_n + extra - len(top_x)
            random_samples = []
            if len(new_random) > 0 and need > 0:
                need = min(need, len(new_random))
                random_samples = new_random[np.random.choice(len(new_random), size=need, replace=False)]
            random_samples = np.array(random_samples, dtype=float) if len(random_samples) else np.empty((0, X.shape[1]), dtype=float)

        candidates = np.concatenate([X_most_visit, top_x, random_samples], axis=0)
        if candidates.size == 0:
            return np.empty((0, X.shape[1]), dtype=float)
        candidates = np.unique(candidates, axis=0)
        # 在此时才将最终候选写入“已见集合”，避免过早地抑制探索
        for row in candidates:
            self._record_node(tuple(np.round(row.reshape(-1), 6)))
        return candidates

    def rollout(self, x: np.ndarray, y: np.ndarray, iteration: int) -> np.ndarray:
        if self.func.name in {"rastrigin", "ackley", "levy"}:
            index_max = int(np.argmax(y))
            initial_x = x[index_max]
            predicted = float(
                self.model.predict(initial_x.reshape(1, -1, 1), verbose=False).reshape(1)
            )
            board = OptTask(tuple(initial_x), predicted, False)
            self.exploration_weight = self.ratio * abs(float(y.max()))
            num_list = (18, 2, 0) if self.func.name == "rastrigin" else (15, 3, 2)
            candidates = self.single_rollout(x, board, num_list)
        else:
            uct_low = self.attenuate_late and (iteration % 100 >= 80)
            top_points = self._get_unique_top_points(x, y, k=max(1, int(self.root_points)))
            collected: List[np.ndarray] = []
            for candidate in top_points:
                prediction = float(
                    self.model.predict(candidate.reshape(1, -1, 1), verbose=False).reshape(1)
                )
                exp_weight = self.ratio * abs(float(y.max()))
                if uct_low:
                    exp_weight *= 0.5
                self.exploration_weight = exp_weight
                board = OptTask(tuple(candidate), prediction, False)
                rollout_points = self.single_rollout(x, board, self.num_list)
                collected.append(rollout_points)
            candidates = np.vstack(collected) if collected else np.empty((0, x.shape[1]), dtype=float)

        if candidates.size == 0:
            return np.empty((0, x.shape[1]), dtype=float)
        unique_candidates = np.unique(candidates, axis=0)
        if len(unique_candidates) > self.num_samples_per_acquisition:
            indices = np.random.choice(
                len(unique_candidates), size=self.num_samples_per_acquisition, replace=False
            )
            unique_candidates = unique_candidates[indices]
        elif len(unique_candidates) < self.num_samples_per_acquisition:
            needed = self.num_samples_per_acquisition - len(unique_candidates)
            filler = self._sample_random_points(np.vstack([x, unique_candidates]), needed)
            if filler.size:
                unique_candidates = np.vstack([unique_candidates, filler])
        return unique_candidates

    def _get_unique_top_points(self, X: np.ndarray, y: np.ndarray, k: int = 3) -> np.ndarray:
        """从整体数据中选取按 y 排序最高的 k 个互不重复的点。

        若存在重复坐标，将继续向下取，直到满足 k 个或数据耗尽。
        """
        indices = np.argsort(y)
        want = max(1, int(k))
        # 先取最后 want 个（最高）
        current = np.unique(X[indices[-want:]], axis=0)
        offset = want + 1
        while len(current) < want and offset <= len(indices):
            candidate = X[indices[-offset]].reshape(1, -1)
            current = np.unique(np.concatenate([current, candidate], axis=0), axis=0)
            offset += 1
        return current

    def _select(self, node: "OptTask") -> List["OptTask"]:
        path: List["OptTask"] = []
        # 用坐标追踪选择路径，防止在已展开子图中循环不返
        seen_tups: Set[Tuple[float, ...]] = set()
        while True:
            path.append(node)
            seen_tups.add(tuple(np.round(np.array(node.tup, dtype=float), 6)))
            if node not in self.children or not self.children[node]:
                return path
            pending = self.unexpanded.get(node)
            if pending:
                while pending:
                    candidate = pending.pop()
                    if candidate not in self.children:
                        path.append(candidate)
                        return path
            # 正常 UCT 选择
            next_node = self._uct_select(node)
            # 检测环：若返回到已见坐标，尝试选择一个未见过的候选；若没有则提前返回当前路径
            cand_seen_key = tuple(np.round(np.array(next_node.tup, dtype=float), 6))
            if cand_seen_key in seen_tups:
                # 选择一个未出现过的子节点作为替代
                alt = None
                for cand in self.children[node]:
                    key = tuple(np.round(np.array(cand.tup, dtype=float), 6))
                    if key not in seen_tups:
                        alt = cand
                        break
                if alt is None:
                    # 无可选替代，返回当前路径作为叶子，交由上层进行扩展/回传
                    if self._debug:
                        print("[DEBUG] _select detected cycle; returning current path", flush=True)
                    return path
                next_node = alt
            node = next_node

    def _expand(self, node: "OptTask") -> None:
        if node in self.children:
            return
        action = list(range(len(node.tup)))
        t0 = time.perf_counter()
        children_set = OptTask.find_children(node, action, self.func, self.model)
        t1 = time.perf_counter()
        if self._debug and (t1 - t0 > 2.0):
            print(
                f"[DEBUG] _expand predict {len(children_set)} children took {t1 - t0:.2f}s",
                flush=True,
            )
        # 过滤掉与自身坐标相同的子节点，进一步降低自环概率
        children_set = {c for c in children_set if c.tup != node.tup}
        self.children[node] = children_set
        self.unexpanded[node] = [child for child in children_set if child not in self.children]

    def _simulate(self, node: "OptTask") -> float:
        return node.reward(self.model)

    def _backpropagate(self, path: List["OptTask"], reward: float) -> None:
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node: "OptTask") -> "OptTask":
        log_n = math.log(self.N[node] + 1)

        def uct(n: "OptTask") -> float:
            return n.value + self.exploration_weight * math.sqrt(log_n / (self.N[n] + 1))

        return max(self.children[node], key=uct)

    def _sample_random_points(self, existing: np.ndarray, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0, existing.shape[1]), dtype=float)

        lb, ub = self.func.lb[0], self.func.ub[0]
        step = self.func.turn
        grid = np.arange(lb, ub + step, step).round(5)

        samples: List[np.ndarray] = []
        attempts = 0
        max_attempts = max(100, count * 50)

        def is_duplicate(arr: np.ndarray, point: np.ndarray) -> bool:
            if arr.size == 0:
                return False
            arr_flat = arr.reshape(arr.shape[0], -1)
            point_flat = point.reshape(-1)
            return np.any(np.all(np.isclose(arr_flat, point_flat, atol=1e-8), axis=1))

        all_existing = existing.copy()
        while len(samples) < count and attempts < max_attempts:
            candidate = np.random.choice(grid, size=self.func.dims).astype(float)
            candidate = candidate.reshape(1, -1)
            tup_key = tuple(np.round(candidate.reshape(-1), 6))
            if tup_key in self._local_seen:
                attempts += 1
                continue
            if not is_duplicate(all_existing, candidate):
                samples.append(candidate.reshape(-1))
                all_existing = np.vstack([all_existing, candidate])
                self._record_node(tup_key)
            attempts += 1

        if not samples:
            return np.empty((0, existing.shape[1]), dtype=float)

        return np.array(samples, dtype=float)

    def _record_node(self, tup: Tuple[float, ...]) -> None:
        if tup not in self._local_seen:
            self._local_seen.add(tup)
        if tup not in self._initial_visited:
            self.new_nodes.add(tup)

    def get_generated_nodes(self) -> Set[Tuple[float, ...]]:
        return set(self.new_nodes)


_OptTaskBase = namedtuple("OptTaskBase", "tup value terminal")


class Node(ABC):
    """Abstract node definition used by the tree search."""

    @abstractmethod
    def find_children(self) -> Set["Node"]:
        """Return all possible successor nodes."""

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the node has no children."""

    @abstractmethod
    def reward(self, model: keras.Model) -> float:
        """Return the reward associated with this node."""

    @abstractmethod
    def __hash__(self) -> int:
        """Nodes must be hashable."""

    @abstractmethod
    def __eq__(self, other: "Node") -> bool:
        """Nodes must be comparable."""


class OptTask(_OptTaskBase, Node):
    """Represents an optimization task node in the search tree."""

    # 关键修复：以坐标 tup 作为节点唯一性判定，避免相同坐标被当作不同节点
    # 这可防止选择阶段在已完全展开的子图中产生环路而无法返回。
    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self.tup)

    def __eq__(self, other: "Node") -> bool:  # type: ignore[override]
        try:
            return isinstance(other, OptTask) and self.tup == other.tup
        except Exception:
            return False

    @staticmethod
    def _random_mutation(
        tup: List[float], index: int, turn: float, candidates: np.ndarray, dims: int
    ) -> None:
        flip = random.randint(0, 5)
        if flip == 0:
            tup[index] += turn
        elif flip == 1:
            tup[index] -= turn
        elif flip in (2, 3):
            count = int(dims / 5) if flip == 2 else int(dims / 10)
            for _ in range(max(1, count)):
                idx = random.randint(0, dims - 1)
                tup[idx] = float(np.random.choice(candidates))
        elif flip in (4, 5):
            tup[index] = float(np.random.choice(candidates))
        tup[index] = round(tup[index], 5)

    @classmethod
    def _clip(cls, tup: np.ndarray, lb: float, ub: float) -> np.ndarray:
        return np.clip(tup, lb, ub)

    @classmethod
    def _child_tuples(
        cls, board: "OptTask", action: List[int], func: ObjectiveFunction
    ) -> List[np.ndarray]:
        turn = func.turn
        values = np.arange(func.lb[0], func.ub[0] + func.turn, func.turn).round(5)
        children = []
        for index in action:
            candidate = list(board.tup)
            cls._random_mutation(candidate, index, turn, values, func.dims)
            candidate = cls._clip(np.array(candidate, dtype=float), func.lb[0], func.ub[0])
            children.append(candidate)
        return children

    @staticmethod
    def find_children(
        board: "OptTask", action: List[int], func: ObjectiveFunction, model: keras.Model
    ) -> Set["OptTask"]:
        if board.terminal:
            return set()
        tuples = OptTask._child_tuples(board, action, func)
        predictions = model.predict(
            np.array(tuples, dtype=float).reshape(len(tuples), func.dims, 1), verbose=False
        )
        return {OptTask(tuple(t), float(v[0]), False) for t, v in zip(tuples, predictions)}

    def reward(self, model: keras.Model) -> float:
        prediction = model.predict(
            np.array(self.tup, dtype=float).reshape(1, -1, 1), verbose=False
        )
        return float(np.asarray(prediction).reshape(-1)[0])

    def is_terminal(self) -> bool:
        return self.terminal
