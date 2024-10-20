import keras
import numpy as np
import pytest

from dante.neural_surrogate import AckleySurrogateModel
from dante.obj_functions import Ackley, ObjectiveFunction
from dante.tree_exploration import OptTask, TreeExploration
from dante.utils import generate_initial_samples


@pytest.fixture
def ackley_function():
    return Ackley(dims=3, turn=0.1)


@pytest.fixture
def ackley_surrogate_model(ackley_function: ObjectiveFunction) -> keras.Model:
    surrogate = AckleySurrogateModel(input_dims=ackley_function.dims, epochs=1)
    input_x, input_scaled_y = generate_initial_samples(
        ackley_function, sample_count=200, apply_scaling=True
    )
    return surrogate(input_x, input_scaled_y)


class TestOptTask:
    def test_find_children(self, ackley_function, ackley_surrogate_model):
        initial_board = OptTask(tup=(0, 0, 0), value=0, terminal=False)
        action = [0, 1, 2]
        children = OptTask.find_children(
            initial_board, action, ackley_function, ackley_surrogate_model
        )

        assert len(children) == 3
        for child in children:
            assert isinstance(child, OptTask)
            assert len(child.tup) == 3
            assert not child.terminal

    def test_reward(self, ackley_surrogate_model):
        board = OptTask(tup=(1, 1, 1), value=0, terminal=False)
        reward = OptTask.reward(board, ackley_surrogate_model)

        assert isinstance(reward, float)
        assert reward > 0  # Ackley function is always positive

    def test_is_terminal(self):
        terminal_board = OptTask(tup=(0, 0, 0), value=0, terminal=True)
        non_terminal_board = OptTask(tup=(0, 0, 0), value=0, terminal=False)

        assert terminal_board.is_terminal()
        assert not non_terminal_board.is_terminal()


@pytest.fixture
def tree_explorer(
        ackley_function: Ackley, ackley_surrogate_model: keras.Model
):
    return TreeExploration(
        func=ackley_function,
        model=ackley_surrogate_model,
        exploration_weight=0.1,
        rollout_round=50,
        ratio=0.1,
    )


class TestTreeExploration:
    def test_initialization(
            self,
            tree_explorer: TreeExploration,
            ackley_function: Ackley,
            ackley_surrogate_model: AckleySurrogateModel,
    ):
        assert tree_explorer.func == ackley_function
        assert tree_explorer.model == ackley_surrogate_model
        assert tree_explorer.exploration_weight == 0.1
        assert tree_explorer.rollout_round == 50
        assert tree_explorer.ratio == 0.1

    def test_do_rollout(self, tree_explorer: TreeExploration):
        initial_board = OptTask(tup=(0, 0, 0), value=0, terminal=False)
        tree_explorer.do_rollout(initial_board)

        assert initial_board in tree_explorer.N
        assert initial_board in tree_explorer.Q
        assert tree_explorer.N[initial_board] > 0
        assert tree_explorer.Q[initial_board] != 0

    def test_choose(self, tree_explorer: TreeExploration):
        initial_board = OptTask(tup=(0, 0, 0), value=10, terminal=False)
        tree_explorer.do_rollout(initial_board)
        result, random_nodes = tree_explorer.choose(initial_board)

        assert isinstance(result, OptTask)
        assert isinstance(random_nodes, list)
        assert len(random_nodes) == 2
        assert all(isinstance(node, tuple) for node in random_nodes)

    def test_data_process(self, tree_explorer: TreeExploration):
        x = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        boards = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
        result = tree_explorer.data_process(x, boards)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        assert np.array_equal(result, np.array([[3, 3, 3], [4, 4, 4]]))

    def test_most_visit_node(self, tree_explorer: TreeExploration):
        x = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        initial_board = OptTask(tup=(3, 3, 3), value=0, terminal=False)
        tree_explorer.N[initial_board] = 10
        tree_explorer.children[initial_board] = set()

        result = tree_explorer.most_visit_node(x, top_n=1)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        assert np.array_equal(result[0], np.array([3, 3, 3]))

    def test_single_rollout(self, tree_explorer: TreeExploration):
        x = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        initial_board = OptTask(tup=(0, 0, 0), value=0, terminal=False)
        result = tree_explorer.single_rollout(x, initial_board, [15, 1, 2])

        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 3
        assert (
                result.shape[0] <= 20
        )  # As per the implementation, it should return at most 20 points
