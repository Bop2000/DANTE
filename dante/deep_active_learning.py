from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
import time

import numpy as np
import matplotlib.pyplot as plt

from dante.neural_surrogate import SurrogateModel
from dante.obj_functions import ObjectiveFunction
from dante.tree_exploration import TreeExploration
from dante.utils import generate_initial_samples


def _plot_global_min_value_trend(
    iterations: list[int],
    min_values: list[float],
    output_path: Path,
) -> None:
    if not iterations or not min_values:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, min_values, marker="o", linestyle="-", color="#1f77b4")
    ax.set_title("Global Minimum Value Trend")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Current Minimum True Value Found")
    ax.grid(True)
    if iterations:
        f = max(1, len(iterations) // 20)
        xticks = [it for idx, it in enumerate(iterations, start=1) if idx % f == 0 or it == iterations[-1]]
        ax.set_xticks(xticks)

    global_min = float("inf")
    for iter_num, value in zip(iterations, min_values):
        if value < global_min:
            global_min = value
            label_txt = f"{value:.2f}" if abs(value) < 1e3 else f"{value:.2e}"
            ax.annotate(
                label_txt,
                (iter_num, value),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_best_samples_pearson(
    iterations: list[int],
    values: list[float],
    output_path: Path,
) -> None:
    if not iterations or not values:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        iterations,
        values,
        marker="o",
        linestyle="-",
        color="#ff00ff",
        linewidth=1.5,
        markersize=4,
        label="Pearson on Best Samples",
    )
    ax.set_title("Pearson Correlation on Best Samples")
    ax.set_xlabel("Active Learning Iteration")
    ax.set_ylabel("Pearson Correlation Coefficient")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    if iterations:
        f = max(1, len(iterations) // 20)
        xticks = [it for idx, it in enumerate(iterations, start=1) if idx % f == 0 or it == iterations[-1]]
        ax.set_xticks(xticks)
    ax.legend(loc="upper right")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@dataclass
class DeepActiveLearning:
    func: ObjectiveFunction
    num_data_acquisition: int
    surrogate: SurrogateModel
    tree_explorer_args: Dict[str, Any] = field(default_factory=dict)
    num_init_samples: int = 200
    num_samples_per_acquisition: int = 20
    input_x: np.ndarray = None
    input_scaled_y: np.ndarray = None
    input_raw_y: Optional[np.ndarray] = None

    def __post_init__(self):
        assert self.num_data_acquisition > 0
        self.dims = self.func.dims
        if self.input_x is None or self.input_scaled_y is None:
            self.input_x, self.input_scaled_y = generate_initial_samples(
                self.func, self.num_init_samples, apply_scaling=True
            )
        else:
            assert len(self.input_x) == len(
                self.input_scaled_y
            ), "input_x 与 input_scaled_y 的长度必须一致"
        self.input_x = np.asarray(self.input_x, dtype=np.float32)
        self.input_scaled_y = np.asarray(self.input_scaled_y, dtype=np.float32).reshape(-1)
        if self.input_raw_y is None:
            self.input_raw_y = np.array(
                [
                    self.func(x, apply_scaling=False, track=False)
                    for x in self.input_x
                ],
                dtype=np.float32,
            )
        else:
            assert len(self.input_x) == len(
                self.input_raw_y
            ), "input_raw_y 长度必须与 input_x 保持一致"
            self.input_raw_y = np.asarray(self.input_raw_y, dtype=np.float32).reshape(-1)
        self._visited_nodes: Set[Tuple[float, ...]] = set()

    def run(self):
        iteration_history: list[int] = []
        global_min_history: list[float] = []
        global_min_plot_path = Path(self.func.tracker.folder_name) / "global_min_value_trend.png"
        best_samples_iterations: list[int] = []
        best_samples_pearson_history: list[float] = []

        for i in range(self.num_data_acquisition // self.num_samples_per_acquisition):
            iteration_start = time.perf_counter()
            prev_best = getattr(self.func.tracker, "_current_best", float("inf"))
            train_start = time.perf_counter()
            model = self.surrogate(self.input_x, self.input_scaled_y, verbose=True)
            train_duration = time.perf_counter() - train_start
            viz_duration = 0.0

            viz_checkpoint = time.perf_counter()
            pearson_best = self._compute_best_samples_pearson(model)
            viz_duration += time.perf_counter() - viz_checkpoint
            if pearson_best is not None:
                iteration_idx = i + 1
                best_samples_iterations.append(iteration_idx)
                best_samples_pearson_history.append(pearson_best)
                pearson_path = (
                    Path(self.func.tracker.folder_name)
                    / "best_samples_pearson_trend.png"
                )
                viz_checkpoint = time.perf_counter()
                _plot_best_samples_pearson(
                    best_samples_iterations,
                    best_samples_pearson_history,
                    pearson_path,
                )
                viz_duration += time.perf_counter() - viz_checkpoint

            tree_explorer = TreeExploration(
                func=self.func,
                model=model,
                num_samples_per_acquisition=self.num_samples_per_acquisition,
                visited_nodes=self._visited_nodes,
                **self.tree_explorer_args,
            )
            rollout_start = time.perf_counter()
            top_x = tree_explorer.rollout(
                self.input_x,
                self.input_scaled_y,
                iteration=i,
            )
            rollout_duration = time.perf_counter() - rollout_start
            top_y = np.array([self.func(x, apply_scaling=True) for x in top_x])
            top_y_raw = np.array(
                [self.func(x, apply_scaling=False, track=False) for x in top_x]
            )
            self._log_iteration(i + 1, top_x, top_y_raw, top_y)
            self.input_x = np.concatenate((self.input_x, top_x), axis=0)
            self.input_scaled_y = np.concatenate((self.input_scaled_y, top_y))
            self.input_raw_y = np.concatenate((self.input_raw_y, top_y_raw))

            new_best = getattr(self.func.tracker, "_current_best", prev_best)
            generated_nodes = tree_explorer.get_generated_nodes()
            if new_best < prev_best - 1e-12:
                self._visited_nodes = set(generated_nodes)
            else:
                self._visited_nodes.update(generated_nodes)

            current_best = float(np.min(self.input_raw_y)) if self.input_raw_y.size else None
            if current_best is not None and np.isfinite(current_best):
                iteration_idx = i + 1
                iteration_history.append(iteration_idx)
                global_min_history.append(current_best)
                viz_checkpoint = time.perf_counter()
                _plot_global_min_value_trend(
                    iteration_history, global_min_history, global_min_plot_path
                )
                viz_duration += time.perf_counter() - viz_checkpoint

                # 若为 Rosenbrock/Ackley，真值达到 0 则提前结束
                _fname = getattr(self.func, "name", "").lower()
                if _fname in ("rosenbrock", "ackley", "schwefel") and np.isclose(current_best, 0.0, atol=1e-12):
                    print(f"[INFO] 触达 {_fname.capitalize()} 全局最优 (raw=0.0)，提前结束迭代。")
                    break

            total_duration = time.perf_counter() - iteration_start
            print(
                f"[Timing] Iteration {i + 1}: "
                f"train={train_duration:.2f}s, "
                f"rollout={rollout_duration:.2f}s, "
                f"viz={viz_duration:.2f}s, "
                f"total={total_duration:.2f}s"
            )
            if np.isclose(self.input_scaled_y.min(), 0.0):
                break

    def _log_iteration(
        self,
        iteration: int,
        top_x: np.ndarray,
        top_y_raw: np.ndarray,
        top_y_scaled: np.ndarray,
    ) -> None:
        print(
            f"[Iteration {iteration}] Selected {len(top_x)} candidates "
            "(raw and scaled objective values shown):"
        )
        for idx, (coords, raw_val, scaled_val) in enumerate(
            zip(top_x, top_y_raw, top_y_scaled), start=1
        ):
            coords_str = np.array2string(
                coords,
                precision=4,
                separator=", ",
                suppress_small=True,
            )
            print(
                f"  #{idx:02d} raw={raw_val:.6f}, scaled={scaled_val:.6f}, x={coords_str}"
            )
        best_val = getattr(self.func.tracker, "_current_best", None)
        best_x = getattr(self.func.tracker, "_current_best_x", None)
        if best_x is not None and best_val is not None and np.isfinite(best_val):
            best_coords = np.array2string(
                best_x,
                precision=4,
                separator=", ",
                suppress_small=True,
            )
            print(
                f"[Iteration {iteration}] Current best raw objective = {best_val:.6f}, "
                f"x = {best_coords}"
            )
        else:
            print(f"[Iteration {iteration}] Current best raw objective unavailable.")

    def _compute_best_samples_pearson(self, model: SurrogateModel) -> Optional[float]:
        if self.input_x is None or self.input_x.size == 0:
            return None
        if self.input_scaled_y is None or self.input_scaled_y.size == 0:
            return None
        if self.input_raw_y is None or self.input_raw_y.size == 0:
            return None

        try:
            predictions = model.predict(
                self.input_x.reshape(len(self.input_x), self.dims, 1), verbose=False
            ).reshape(-1)
        except Exception:
            return None

        true_scaled = self.input_scaled_y.reshape(-1)
        raw_values = self.input_raw_y.reshape(-1)

        top_k = min(60, len(raw_values))
        if top_k < 2:
            return 0.0

        best_indices = np.argsort(raw_values)[:top_k]
        subset_pred = predictions[best_indices]
        subset_true = true_scaled[best_indices]

        if subset_pred.size < 2 or subset_true.size < 2:
            return 0.0

        pearson = float(np.corrcoef(subset_pred, subset_true)[0, 1])
        if np.isnan(pearson):
            pearson = 0.0
        return pearson
