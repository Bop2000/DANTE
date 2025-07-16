import numpy as np
import lightgbm as lgb
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pyDOE import lhs

# 1. 数据准备与采样
SCHWEFEL_LOW, SCHWEFEL_HIGH = -500, 500
INPUT_DIM = 40
N_TRAIN = 1000
N_TEST = 200

# 优先从npy文件加载训练集和测试集，格式一致
TRAIN_PATH = os.path.join(os.path.dirname(__file__), '../data/Schwefel-40d-train.npy')
TEST_PATH = os.path.join(os.path.dirname(__file__), '../data/Schwefel-40d-test.npy')

if os.path.exists(TRAIN_PATH):
    train_data = np.load(TRAIN_PATH)
    X_train = train_data[:, :40]
    y_train = train_data[:, 40]
    print(f"成功加载训练集: {TRAIN_PATH}, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
else:
    raise FileNotFoundError(f"训练集文件未找到: {TRAIN_PATH}")

if os.path.exists(TEST_PATH):
    test_data = np.load(TEST_PATH)
    X_test = test_data[:, :40]
    y_test = test_data[:, 40]
    print(f"成功加载测试集: {TEST_PATH}, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")
else:
    raise FileNotFoundError(f"测试集文件未找到: {TEST_PATH}")

# 2. 不做归一化/标准化
# X_train, X_test 保持原始取值

# 3. 代理模型定义（LightGBM）
class LGBM_Surrogate:
    def __init__(self):
        self.model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, verbosity=1)

    def fit(self, x, y):
        print("\n--- 开始LightGBM代理模型训练 ---")
        self.model.fit(x, y)
        print("--- LightGBM代理模型训练结束 ---\n")

    def predict(self, x):
        return self.model.predict(x)

# 4. 训练并评估代理模型
model = LGBM_Surrogate()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'--- LightGBM代理模型在测试集上的表现 ---')
print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'R2: {r2:.4f}\n')

print('代理模型训练与评估结束，进入基于代理模型的采样与真实验证环节。')

# 采样与优化策略参数集中管理
sampling_cfg = {
    "N_SAMPLES_PER_START": 20,      # 每起点采样点数
    "N_STARTS_TOPK": 5,             # topK优点
    "N_STARTS_RANDOM": 10,          # 随机起点
    "mutation_ratio_start": 0.98,   # 初始扰动比例
    "mutation_ratio_end": 0.05,     # 末期扰动比例
    "turn_start": 1/10,             # turn初始比例
    "turn_end": 1/120,              # turn末期比例
}

np.random.seed(42)
N_SAMPLES_PER_START = sampling_cfg["N_SAMPLES_PER_START"]
N_STARTS_TOPK = sampling_cfg["N_STARTS_TOPK"]
N_STARTS_RANDOM = sampling_cfg["N_STARTS_RANDOM"]
mutation_ratio_start = sampling_cfg["mutation_ratio_start"]
mutation_ratio_end = sampling_cfg["mutation_ratio_end"]
turn_start = sampling_cfg["turn_start"]
turn_end = sampling_cfg["turn_end"]

TOTAL_SAMPLES = (N_STARTS_TOPK + N_STARTS_RANDOM) * N_SAMPLES_PER_START

# 采样起点：训练集前k优点
topk_idx = np.argsort(y_train)[:N_STARTS_TOPK]
start_points = [X_train[idx] for idx in topk_idx]

# 再加随机起点
rand_idx = np.random.choice(len(X_train), N_STARTS_RANDOM, replace=False)
for idx in rand_idx:
    start_points.append(X_train[idx])

print(f'采样起点数量: {len(start_points)}，每起点采样数: {N_SAMPLES_PER_START}')

# 自适应扰动参数
def get_mutation_ratio(i, total):
    # 退火式递减（从mutation_ratio_start到mutation_ratio_end）
    return mutation_ratio_start * (mutation_ratio_end/mutation_ratio_start) ** (i/(total-1))

def get_turn(i, total):
    base = (SCHWEFEL_HIGH - SCHWEFEL_LOW)
    turn_frac = turn_start * (turn_end/turn_start) ** (i/(total-1))
    return base * turn_frac

# 采样函数
def mutate_point(x_base, ratio, lb, ub, turn):
    x = x_base.copy()
    dims = len(x)
    n_mut = max(1, int(dims * ratio))
    idxs = np.random.choice(dims, n_mut, replace=False)
    for idx in idxs:
        delta = np.random.uniform(-turn, turn)
        x[idx] += delta
        x[idx] = np.clip(x[idx], lb[idx], ub[idx])
    return x

# 采样主循环
all_sampled_points = []
for start_idx, x_start in enumerate(start_points):
    for i in range(N_SAMPLES_PER_START):
        ratio = get_mutation_ratio(i, N_SAMPLES_PER_START)
        turn = get_turn(i, N_SAMPLES_PER_START)
        x_new = mutate_point(
            x_start,
            ratio,
            np.full(INPUT_DIM, SCHWEFEL_LOW),
            np.full(INPUT_DIM, SCHWEFEL_HIGH),
            turn
        )
        all_sampled_points.append(x_new)

all_sampled_points = np.array(all_sampled_points)
print(f'总采样点数: {len(all_sampled_points)}')

# 用代理模型预测
lgbm_model = model
y_pred = lgbm_model.predict(all_sampled_points)

def schwefel_40d(x):
    x = np.asarray(x)
    assert x.shape[-1] == 40, 'Input must be 40-dimensional.'
    return 418.9829 * 40 - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# 用真实Schwefel函数评估
real_y = np.array([schwefel_40d(x) for x in all_sampled_points])

print('\n--- 采样点、代理模型预测值与真实值对比 ---')
for i, (x, y_hat, y_true) in enumerate(zip(all_sampled_points, y_pred, real_y)):
    print(f'采样点{i+1}:', np.round(x, 2))
    print(f'  代理模型预测: {y_hat:.2f}  真实值: {y_true:.2f}  误差: {abs(y_hat-y_true):.2f}')

print('\n采样与真实验证流程结束。')

# 找到真实值最优的采样点
best_sample_idx = np.argmin(real_y)
best_sample_x = all_sampled_points[best_sample_idx]
best_sample_y = real_y[best_sample_idx]

print('\n==============================')
print('最优采样点编号:', best_sample_idx + 1)
print('最优采样点坐标:', np.round(best_sample_x, 4))
print('最优采样点真实值:', best_sample_y)
print('==============================\n')

print('''\n【采样策略说明】\n- 多起点采样：训练集前k优点+随机起点，提升多样性，避免陷入局部最优。\n- turn和mutation_ratio自适应递减，实现先全局探索，后局部精细搜索。\n- 采样点全局筛选最优。\n''')

# ========== 新增：保存Y值最低的50个点 ==========
# 按真实Y值从低到高排序，取前50个点
sorted_indices = np.argsort(real_y)  # 升序
X_top50 = all_sampled_points[sorted_indices[:50]]
Y_top50 = real_y[sorted_indices[:50]]
np.save('top50_points.npy', X_top50)
print(f"已将Y值最低的50个采样点（shape={X_top50.shape}）保存到top50_points.npy")
print(f"对应的Y真实值为：{Y_top50}")
