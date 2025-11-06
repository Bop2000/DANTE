#!/usr/bin/env python
"""
使用与 Rosenbrock-20d 相同的优化与可视化流程，对 data_raw/Ackley-20d 进行主动优化。

运行示例：
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    python -u experiments/ackley_20d_opt.py \
        --data-dir data_raw/Ackley-20d \
        --acquisitions 8000 \
        --samples-per-acquisition 20 \
        --epochs 400
"""

from __future__ import annotations

import argparse
import os
import sys
import signal
import faulthandler
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.manifold import TSNE

from dante.deep_active_learning import DeepActiveLearning
from dante.neural_surrogate import AckleySurrogateModel
from dante.obj_functions import Ackley


# Enable faulthandler and register SIGUSR1 to dump stacks without terminating.
try:
    if not faulthandler.is_enabled():
        faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
except Exception:
    pass


def _scale_values(func: Ackley, y_raw: np.ndarray) -> np.ndarray:
    """将 Ackley 原始目标值映射到最大化的缩放值。"""
    return np.array([func.scaled(float(val)) for val in y_raw], dtype=np.float32)


def _load_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取离线数据集。"""
    x_train = np.load(data_dir / "Ackley_x_train.npy").astype(np.float32)
    y_train = np.load(data_dir / "Ackley_y_train.npy").astype(np.float32)
    x_test = np.load(data_dir / "Ackley_x_test.npy").astype(np.float32)
    y_test = np.load(data_dir / "Ackley_y_test.npy").astype(np.float32)
    return x_train, y_train, x_test, y_test


@dataclass
class ExperimentConfig:
    data_dir: Path
    seed: int = 0
    init_size: int = 200
    acquisitions: int = 200
    samples_per_acquisition: int = 20
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 1e-3
    exploration_weight: float = 0.1
    rollout_round: int = 50
    ratio: float = 0.1
    verbose: bool = False
    turn: float = 0.1
    nte_roots: int = 3
    attenuate_late: bool = True


def run_experiment(cfg: ExperimentConfig) -> None:
    data_dir = cfg.data_dir

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在")

    x_train_raw, y_train_raw, x_test_raw, y_test_raw = _load_dataset(data_dir)

    ackley = Ackley(dims=x_train_raw.shape[1], turn=cfg.turn)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("ackley20") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    ackley.tracker.folder_name = str(run_dir)
    Path(ackley.tracker.folder_name).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 本次运行的可视化输出目录：{run_dir}")

    # 训练集与测试集都转换为 surrogate 需要的缩放目标（越大越好）
    y_train_scaled = _scale_values(ackley, y_train_raw)
    y_test_scaled = _scale_values(ackley, y_test_raw)

    init_x = x_train_raw
    init_y_scaled = y_train_scaled

    # 更新 tracker，使得初始最优值与坐标立即可用（使用真实目标值）
    init_y_raw = []
    for x_val in init_x:
        raw_val = ackley(x_val, apply_scaling=False, track=True)
        init_y_raw.append(raw_val)
    init_y_raw = np.array(init_y_raw, dtype=np.float32)
    initial_best_idx = int(np.argmin(init_y_raw))
    print(
        "[INFO] 初始数据最优解: "
        f"f(x) = {init_y_raw[initial_best_idx]:.6f}, "
        f"x = {np.array2string(init_x[initial_best_idx], precision=4, separator=', ', suppress_small=True)}"
    )

    surrogate = AckleySurrogateModel(
        input_dims=init_x.shape[1],
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
    )
    surrogate_plot_dir = run_dir / "surrogate_regression"
    surrogate.plot_dir = surrogate_plot_dir
    checkpoint_dir = Path.home() / ".cache" / "dante" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    surrogate.check_point_path = checkpoint_dir / "ackley20_best.keras"

    # save run configuration
    tree_args = {
        "exploration_weight": cfg.exploration_weight,
        "rollout_round": cfg.rollout_round,
        "ratio": cfg.ratio,
        "root_points": cfg.nte_roots,
        "attenuate_late": cfg.attenuate_late,
    }
    _write_run_config(run_dir, cfg, tree_args)

    dal = DeepActiveLearning(
        func=ackley,
        num_data_acquisition=cfg.acquisitions,
        surrogate=surrogate,
        tree_explorer_args=tree_args,
        num_samples_per_acquisition=cfg.samples_per_acquisition,
        num_init_samples=len(init_x),
        input_x=init_x,
        input_scaled_y=init_y_scaled,
        input_raw_y=init_y_raw,
    )

    print(
        f"[INFO] 启动优化：初始样本 {len(init_x)}（使用全部训练数据），"
        f"计划新增 {cfg.acquisitions} 条，每轮获取 {cfg.samples_per_acquisition} 条。"
    )
    dal.run()

    final_x = dal.input_x
    final_scaled = dal.input_scaled_y
    final_raw = np.array(
        [ackley(x, apply_scaling=False, track=False) for x in final_x],
        dtype=np.float32,
    )

    best_idx = int(np.argmin(final_raw))
    print(
        "[RESULT] 当前最优："
        f"f(x) = {final_raw[best_idx]:.6f}，"
        f"scale = {final_scaled[best_idx]:.6f}，"
        f"样本总数 {len(final_raw)}"
    )

    ackley.tracker.dump_trace()

    results = np.array(ackley.tracker._results, dtype=np.float64)
    if results.size:
        np.save(run_dir / "best_trace.npy", results)
        fig, ax = plt.subplots()
        ax.plot(results)
        ax.set_title("Best objective over iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best f(x)")
        fig.savefig(run_dir / "best_trace.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if final_x.size:
        try:
            initial_count = len(init_x)
            offset = np.ones_like(final_x[:1])
            tsne_input = final_x + offset if offset.size else final_x
            perplexity = max(1, min(30, tsne_input.shape[0] - 1))
            if tsne_input.shape[0] > 1 and perplexity > 0:
                embedding = TSNE(
                    n_components=2,
                    learning_rate="auto",
                    init="random",
                    perplexity=perplexity,
                ).fit_transform(tsne_input)
                fig, ax = plt.subplots()
                ax.scatter(
                    embedding[:initial_count, 0],
                    embedding[:initial_count, 1],
                    s=6,
                    c="tab:blue",
                    label="Initial data",
                )
                if embedding.shape[0] > initial_count:
                    ax.scatter(
                        embedding[initial_count:, 0],
                        embedding[initial_count:, 1],
                        s=6,
                        c="tab:red",
                        label="Acquired data",
                    )
                best_idx = int(np.argmin(final_raw))
                ax.scatter(
                    embedding[best_idx, 0],
                    embedding[best_idx, 1],
                    marker="*",
                    s=120,
                    c="gold",
                    label="Current best",
                )
                ax.set_title("t-SNE of sampled points")
                ax.legend(loc="best")
                fig.savefig(run_dir / "tsne_samples.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            print(f"[WARN] Failed to generate t-SNE visualization: {exc}")

    # 使用最终 surrogate 在离线测试集上评估表现
    model = surrogate.model
    if model is None:
        print("[WARN] surrogate 尚未训练成功，跳过测试集评估。")
        return

    y_test_pred_scaled = model.predict(
        x_test_raw.reshape(len(x_test_raw), init_x.shape[1], 1),
        verbose=int(cfg.verbose),
    ).reshape(-1)

    r2 = r2_score(y_test_scaled, y_test_pred_scaled)
    mae = mean_absolute_error(y_test_scaled, y_test_pred_scaled)
    print(f"[TEST] Scaled 目标: R2={r2:.4f}, MAE={mae:.4f}")

    # 将预测值反变换回原始目标，便于对比（Ackley: scaled=100/(y+0.01)）
    epsilon = 1e-8
    inv_pred_raw = 100 / np.clip(y_test_pred_scaled, epsilon, None) - 0.01
    print(
        "[TEST] 原始目标: MAE="
        f"{mean_absolute_error(y_test_raw, inv_pred_raw):.4f}"
    )


def _write_run_config(run_dir: Path, cfg: "ExperimentConfig", tree_args: dict) -> None:
    env_keys: List[str] = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "TF_FORCE_GPU_ALLOW_GROWTH",
        "PYTHONFAULTHANDLER",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYTHONUNBUFFERED",
        "DANTE_DEBUG_TIMING",
        "CUDA_VISIBLE_DEVICES",
    ]
    lines: List[str] = []
    lines.append("# DANTE run configuration\n")
    lines.append(f"script = {Path(__file__).name}\n")
    lines.append("[cli]")
    for k, v in vars(cfg).items():
        lines.append(f"{k} = {v}")
    lines.append("")
    lines.append("[tree_explorer_args]")
    for k, v in tree_args.items():
        lines.append(f"{k} = {v}")
    lines.append("")
    lines.append("[env]")
    for k in env_keys:
        lines.append(f"{k} = {os.getenv(k, '')}")
    try:
        (run_dir / "run_config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Failed to write run_config.txt: {exc}")


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="优化 Ackley-20d 数据集")
    parser.add_argument("--data-dir", type=Path, default=Path("data_raw/Ackley-20d"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-size", type=int, default=200)
    parser.add_argument("--acquisitions", type=int, default=200)
    parser.add_argument("--samples-per-acquisition", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--exploration-weight", type=float, default=0.1)
    parser.add_argument("--rollout-round", type=int, default=50)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--turn", type=float, default=0.1)
    parser.add_argument("--nte-roots", type=int, default=3, help="每轮用于 NTE 的根节点个数（默认 3）")
    parser.add_argument("--no-late-attenuation", action="store_true", help="关闭每 100 轮后 20% 的探索权重衰减")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    cfg_kwargs = vars(args)
    cfg_kwargs["attenuate_late"] = not args.no_late_attenuation
    del cfg_kwargs["no_late_attenuation"]
    return ExperimentConfig(**cfg_kwargs)


if __name__ == "__main__":
    config = parse_args()
    run_experiment(config)
