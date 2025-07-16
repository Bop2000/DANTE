import numpy as np
import lightgbm as lgb
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dante.tree_exploration import TreeExploration
from dante.obj_functions import Schwefel
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names*")
warnings.filterwarnings("ignore")
import os
os.environ["LIGHTGBM_VERBOSE"] = "0"

# 1. 数据准备
SCHWEFEL_LOW, SCHWEFEL_HIGH = -500, 500
INPUT_DIM = 40
N_TRAIN = 1000
N_TEST = 200

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

# 2. 代理模型定义（LightGBM）
class LGBM_Surrogate:
    def __init__(self):
        self.model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, verbosity=-1)

    def fit(self, x, y):
        print("\n--- 开始LightGBM代理模型训练 ---")
        self.model.fit(x, y)
        print("--- LightGBM代理模型训练结束 ---\n")

    def predict(self, x, verbose=False):
        x = np.asarray(x)
        if x.ndim == 3:
            x = x.squeeze(-1)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        y_pred = self.model.predict(x)
        return np.array(y_pred).reshape(-1)

# 3. Schwefel目标函数
schwefel_func = Schwefel(dims=INPUT_DIM, turn=1)

# 4. 主动学习参数
N_ACQUISITIONS = 10  # 主动采样轮数，可根据需要调整
SAMPLES_PER_ACQ = 20  # 每轮采样点数

# 5. 主动学习主循环
for acq_iter in range(N_ACQUISITIONS):
    print(f"\n========== 主动采样第{acq_iter+1}轮 ==========")
    # 训练代理模型
    model = LGBM_Surrogate()
    model.fit(X_train, y_train)
    # 评估代理模型
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'--- LightGBM代理模型在测试集上的表现 ---')
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R2: {r2:.4f}\n')

    # 用TreeExploration采样新点
    explorer = TreeExploration(func=schwefel_func, model=model, num_samples_per_acquisition=SAMPLES_PER_ACQ)
    new_x = explorer.rollout(X_train, y_train, iteration=acq_iter)
    if new_x.ndim == 1:
        new_x = new_x.reshape(1, -1)
    print(f"采样新点 shape: {new_x.shape}")

    # 用真实Schwefel函数评估新点
    new_y = np.array([schwefel_func(x, apply_scaling=False, track=False) for x in new_x])
    print(f"新采样点真实y值: {new_y}")

    # 数据集扩充
    X_train = np.concatenate((X_train, new_x), axis=0)
    y_train = np.concatenate((y_train, new_y))
    print(f"当前训练集规模: {X_train.shape}")

# 6. 采样点保存与评估
print('\n采样与真实验证流程结束。')

# 找到真实值最优的采样点
best_sample_idx = np.argmin(y_train)
best_sample_x = X_train[best_sample_idx]
best_sample_y = y_train[best_sample_idx]

print('\n==============================')
print('最优采样点编号:', best_sample_idx + 1)
print('最优采样点坐标:', np.round(best_sample_x, 4))
print('最优采样点真实值:', best_sample_y)
print('==============================\n')

# ========== 保存Y值最低的50个点 ==========
sorted_indices = np.argsort(y_train)  # 升序
X_top50 = X_train[sorted_indices[:50]]
Y_top50 = y_train[sorted_indices[:50]]
np.save('top50_points.npy', X_top50)
print(f"已将Y值最低的50个采样点（shape={X_top50.shape}）保存到top50_points.npy")
print(f"对应的Y真实值为：{Y_top50}")
