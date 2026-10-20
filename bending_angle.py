import numpy as np
import matplotlib.pyplot as plt

from geodesic import rk4_step

def calculate_bending_angle(b, M, r_start, r_end, h):
    # 初期条件
    vr_start = -np.sqrt(
        1.0 - (b**2 / r_start**2)
        * (1.0 - 2.0 * M / r_start)
    )
    state = np.array([r_start, vr_start, 0.0])

    max_steps = 1000000

    for _ in range(max_steps):
        r, vr, phi = state

        # 捕獲された場合
        if r <= 2.0 * M:
            return np.nan

        # 散乱して遠方に到達した場合
        if r >= r_end and vr > 0:
            alpha = (
                phi - np.pi
                + np.arcsin(b / r_start)
                + np.arcsin(b / r)
            )
            return alpha

        # RK4で次の状態へ
        state = rk4_step(state, h, b, M)

    # 最大ステップ数に達した場合
    raise RuntimeError(
        f"Integration did not finish for b={b}"
    )

M = 1.0
r_start = 500.0
r_end = 500.0
h = 0.01

alpha_list = []

bc = 3 * np.sqrt(3) * M

delta_list = np.array([
    1e-2,
    5e-3,
    1e-3,
    5e-4,
    1e-4,
    5e-5,
    1e-5
])

b_list = bc * (1 + delta_list)

for b in b_list:
    alpha = calculate_bending_angle(b, M, r_start, r_end, h)
    print(f"b={b:.9f}, alpha={alpha:.8f} rad")
    alpha_list.append(alpha)

b_arr = np.array(b_list)
alpha_arr = np.array(alpha_list)

#X = -np.log(b_arr / bc - 1)
X = -np.log(delta_list)

# 最小二乗法の計算
def reg1dim(x, y):
    n = len(x)
    a = (np.dot(x, y) - y.sum() * x.sum() / n) / ((x**2).sum() - (x.sum()**2) / n)
    b = (y.sum() - a * x.sum()) / n
    return a, b

# 臨界値近傍の4点で最小二乗法
m, C_fit = reg1dim(X[3:], alpha_arr[3:])

# 全7点の数値結果
plt.scatter(X, alpha_arr, label="Numerical results")

# 近傍4点から得られた回帰直線
x_line = np.linspace(X[3], X[-1], 100)

plt.plot(
    x_line,
    m * x_line + C_fit,
    color="r",
    label=f"y = {m:.6f}x + {C_fit:.6f}"
)

plt.xlabel("-log(delta)")
plt.ylabel("Bending angle (rad)")
plt.legend()
plt.grid()
plt.show()

print(f"slope = {m:.6f}")
print(f"intercept = {C_fit:.6f}")

C_eff = alpha_arr + np.log(delta_list)
C_theory = np.log(216 * (7 - 4 * np.sqrt(3))) - np.pi

plt.figure(figsize=(8, 6))
plt.semilogx(delta_list, C_eff, "o-", label="Numerical results")
plt.axhline(C_theory, color="black", linestyle="--",
            label="Theoretical limit")

plt.xlabel("delta")
plt.ylabel("C_eff (rad)")
plt.legend()
plt.grid(True)
plt.show()