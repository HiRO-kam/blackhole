import numpy as np
import matplotlib.pyplot as plt

# geodesic.py で定義した関数を読み込む
from geodesic import rk4_step, Error_evaluation


# --- 計算条件 ---
M = 1.0
b = 5.2
r_start = 100.0
phi_start = 0.0

h_list = [0.1, 0.05, 0.025, 0.0125]
lambda_end = 100.0

epsilon_list = []


# --- 刻み幅を変えながら計算 ---
for h in h_list:

    # 毎回同じ初期条件から開始する
    vr_start = -np.sqrt(
        1.0 - (b**2 / r_start**2)
        * (1.0 - 2.0 * M / r_start)
    )

    state = np.array([r_start, vr_start, phi_start])

    # 同じ積分区間を保つ
    steps = round(lambda_end / h)

    error_history = []

    for _ in range(steps):
        error = Error_evaluation(state, b, M)
        error_history.append(error)

        state = rk4_step(state, h, b, M)

    # 最終ステップの誤差も含める
    error_history.append(Error_evaluation(state, b, M))

    epsilon = max(error_history)
    epsilon_list.append(epsilon)

    print(f"h={h:.4f}, epsilon={epsilon:.6e}")


# --- 実験的な収束次数を計算 ---
print("\nConvergence order:")

for i in range(len(h_list) - 1):
    p = np.log(
        epsilon_list[i] / epsilon_list[i + 1]
    ) / np.log(h_list[i] / h_list[i + 1])

    print(
        f"h={h_list[i]} -> {h_list[i+1]}, "
        f"p={p:.4f}"
    )


# --- グラフ描画 ---
plt.figure(figsize=(7, 5))

h_ref = np.array(h_list)
epsilon_ref = epsilon_list[0] * (h_ref / h_list[0])**4

plt.loglog(h_ref, epsilon_ref, "--", label="Reference: O(h^4)")
plt.loglog(
    h_list,
    epsilon_list,
    "o-",
    label="Conservation error"
)

plt.xlabel("Step size h")
plt.ylabel("Maximum |Q - 1|")
plt.title("RK4 Convergence Test")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()
