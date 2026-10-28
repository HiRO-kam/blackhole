import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

from pathlib import Path
from PIL import Image

plt.rcParams["font.family"] = "MS Gothic"   # 文字化けするなら "Yu Gothic" や "Meiryo" でも可
plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止

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
plt.figure(figsize=(8, 6))

h_ref = np.array(h_list)
epsilon_ref = epsilon_list[0] * (h_ref / h_list[0])**4

plt.loglog(h_ref, epsilon_ref, "--", label=r"基準線：$O(h^{4})$")
plt.loglog(h_list, epsilon_list, "o-", label="保存量の誤差",)

plt.xlabel("刻み幅 $h$", fontsize=13)
plt.ylabel(r"最大値 $|Q-1|$", fontsize=13)
plt.title("4次ルンゲ＝クッタ法の収束確認", fontsize=15, pad=12, y=-0.2)

plt.minorticks_on()

# 横軸の主目盛を h_list に固定
h_sorted = sorted(h_list)

plt.xticks(
    h_sorted,
    [f"{h:g}" for h in h_sorted],
    fontsize=11
)

# 各主目盛の間に補助目盛を3本ずつ配置（対数間隔）
minor_ticks = []

for h1, h2 in zip(h_sorted[:-1], h_sorted[1:]):
    minor_ticks.extend(np.geomspace(h1, h2, 5)[1:-1])

ax = plt.gca()
ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
ax.xaxis.set_minor_formatter(NullFormatter())

# 目盛を内向きに
plt.tick_params(
    axis="both",
    which="both",
    direction="in",
    top=True,
    right=True
)

plt.xlim(0.01, 0.12)
# 凡例とグリッド
plt.legend(
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    fontsize=11
)
plt.grid(True, which="both", alpha=0.3)

plt.subplots_adjust(
    left=0.13,
    right=0.96,
    bottom=0.19,
    top=0.94
)

output = Path(
r"C:\Users\sun8s\OneDrive\Desktop\latex\blackhole_figure\convergence_test.png"
)

plt.savefig(output, dpi=100)

with Image.open(output) as img:
    print("保存先:", output)
    print("画像サイズ:", img.size)
plt.show()
