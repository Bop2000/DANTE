import os
import numpy as np
import pytest
from dante.obj_functions import (
    Ackley,
    Griewank,
    Levy,
    Michalewicz,
    ObjectiveFunction,
    Rastrigin,
    Rosenbrock,
    Schwefel,
)
from dante.utils import Tracker, generate_initial_samples


@pytest.fixture
def tracker():
    """Fixture to create a Tracker instance for testing."""
    return Tracker(folder_name="test_results")


class TestTracker:
    def test_tracker_initialization(self, tracker):
        """Test if the Tracker is initialized correctly."""
        assert tracker.folder_name == "test_results"
        assert tracker._counter == 0
        assert tracker._results == []
        assert tracker._x_values == []
        assert tracker._current_best == float("inf")
        assert tracker._current_best_x is None

    def test_create_folder(self, tracker, tmpdir):
        """Test if the folder is created successfully."""
        tracker.folder_name = str(tmpdir.join("test_folder"))
        tracker._create_folder()
        assert os.path.exists(tracker.folder_name)

    def test_dump_trace(self, tracker, tmpdir):
        """Test if dump_trace saves the results correctly."""
        tracker.folder_name = str(tmpdir.join("test_folder"))
        tracker._create_folder()
        tracker._results = [1.0, 0.5, 0.25]
        tracker.dump_trace()

        result_file = os.path.join(tracker.folder_name, "result.npy")
        assert os.path.exists(result_file)
        loaded_results = np.load(result_file)
        np.testing.assert_array_equal(loaded_results, np.array([1.0, 0.5, 0.25]))

    def test_track(self, tracker):
        """Test if track method updates the tracker correctly."""
        tracker.track(result=0.5, x=np.array([1, 2, 3]))
        assert tracker._counter == 1
        assert tracker._current_best == 0.5
        np.testing.assert_array_equal(tracker._current_best_x, np.array([1, 2, 3]))
        assert tracker._results == [0.5]
        assert len(tracker._x_values) == 1
        np.testing.assert_array_equal(tracker._x_values[0], np.array([1, 2, 3]))

    def test_track_update_best(self, tracker):
        """Test if track method updates the best result correctly."""
        tracker.track(result=0.5, x=np.array([1, 2, 3]))
        tracker.track(result=0.3, x=np.array([4, 5, 6]))
        assert tracker._current_best == 0.3
        np.testing.assert_array_equal(tracker._current_best_x, np.array([4, 5, 6]))
        assert tracker._results == [0.5, 0.3]

    def test_track_no_update(self, tracker):
        """Test if track method doesn't update when the result is worse."""
        tracker.track(result=0.5, x=np.array([1, 2, 3]))
        tracker.track(result=0.7, x=np.array([4, 5, 6]))
        assert tracker._current_best == 0.5
        np.testing.assert_array_equal(tracker._current_best_x, np.array([1, 2, 3]))
        assert tracker._results == [0.5, 0.5]


@pytest.mark.parametrize(
    "function_class, dims",
    [
        (Ackley, 3),
        (Rastrigin, 3),
        (Rosenbrock, 3),
        (Griewank, 3),
        (Michalewicz, 3),
        (Schwefel, 3),
        (Levy, 3),
    ],
)
def test_generate_initial_samples_shape(function_class: ObjectiveFunction, dims: int):
    """Test if the generated samples have the correct shape."""
    obj_func: ObjectiveFunction = function_class(dims=dims)  # type: ignore
    sample_count = 100

    input_samples, output_values = generate_initial_samples(obj_func, sample_count)

    assert input_samples.shape == (sample_count, dims)
    assert output_values.shape == (sample_count,)
