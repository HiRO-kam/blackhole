import numpy as np
import matplotlib.pyplot as plt

# --- シミュレーションの実行設定 ---
M = 1.0
b_list = [1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.2, 6.0, 7.0, 8.0, 9.0, 10.0]
r_start = 100.0
phi_start = 0.0
dlam = 0.01
steps = 80000
error_data = {}

#  導関数（右辺の関数）を定義する
def derivatives(state, b, M):
    r, vr, phi = state
    
    dr_dlam = vr
    dvr_dlam = (b**2 / r**3) * (1.0 - 3.0 * M / r)
    dphi_dlam = b / r**2
    
    return np.array([dr_dlam, dvr_dlam, dphi_dlam])

def Error_evaluation(state, b, M):
    r, vr, phi = state

    Q = vr**2 + (b**2 / r**2) * (1.0 - 2.0 * M / r)
    
    return abs(Q-1)    

#  4次のルンゲ・クッタ法の1ステップ分を実行する関数
def rk4_step(state, dlam, b, M):
    k1 = derivatives(state, b, M)
    k2 = derivatives(state + 0.5 * dlam * k1, b, M)
    k3 = derivatives(state + 0.5 * dlam * k2, b, M)
    k4 = derivatives(state + dlam * k3, b, M)
    
    return state + (dlam / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def calculate_trajectory(b, M, r_start, dlam, steps, r_end=150.0):

    vr_start = -np.sqrt(
        1.0 - (b**2 / r_start**2)
        * (1.0 - 2.0 * M / r_start)
    )

    state = np.array([r_start, vr_start, 0.0])

    r_history = []
    phi_history = []
    error_history = []

    for _ in range(steps):
        r, vr, phi = state

        r_history.append(r)
        phi_history.append(phi)
        error_history.append(Error_evaluation(state, b, M))

        if r <= 2.0 * M:
            break

        if r >= r_end and vr > 0:
            break

        state = rk4_step(state, dlam, b, M)

    return (
        np.array(r_history),
        np.array(phi_history),
        np.array(error_history)
    )


def main():
    plt.figure(figsize=(8, 8))
    error_data = {}

    for b in b_list:
        r_arr, phi_arr, error_arr = calculate_trajectory(
            b, M, r_start, dlam, steps
        )

        x = r_arr * np.cos(phi_arr)
        y = r_arr * np.sin(phi_arr)

        epsilon_cons = np.max(error_arr)

        error_data[b] = (r_arr, error_arr)

        plt.plot(x, y, label=f"Light ray (b={b})")
        print(f"b={b}, epsilon_cons={epsilon_cons:.6e}")

    # ブラックホールの描画
    bh = plt.Circle((0, 0), 2.0 * M, color='black', zorder=5, label='Event Horizon (r=2M)')
    photon_sphere = plt.Circle((0, 0), 3.0 * M, color='gray', fill=False, linestyle='--', zorder=4, label='Photon Sphere (r=3M)')

    plt.gca().add_patch(bh)
    plt.gca().add_patch(photon_sphere)
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.gca().set_aspect('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='upper right')
    plt.title("Schwarzschild Light Geodesic (Custom RK4)")
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(8, 6))

    for b, (r_values, errors) in error_data.items():
        plt.plot(r_values / M, errors, label=f"b={b}")

    plt.axvline(2*M, color='black', linestyle='--', label='Event Horizon')
    plt.xlim(2, 10)
    plt.yscale('log')
    plt.xlabel('r / M')
    plt.ylabel('|Q - 1|')
    plt.legend()
    plt.title("Error Evaluation of Light Geodesic (Custom RK4)")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()