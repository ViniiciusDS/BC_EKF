import numpy as np
import matplotlib.pyplot as plt
from bc_ekf import run_bc_ekf

# ============================
# 1. Parâmetros gerais
# ============================
T = 0.05
t_final = 50

anchors = np.array([
    [0,0,1],
    [0,5,1],
    [5,0,1]
]).T
num_anchors = anchors.shape[1]

l = 0.65 / 2
z_c = 0.5

v_true = 0.3
w_true = np.deg2rad(7.5)

sigma_v = 0.02
sigma_w = 0.05
sigma_uwb = np.sqrt(0.0025)

N_runs = 50  # Número de execuções Monte Carlo

# Para armazenar os RMSEs
rmse_pos_list = []
rmse_heading_list = []

print(f"Iniciando Monte Carlo com {N_runs} execuções...\n")

# ============================
# 2. Loop Monte Carlo
# ============================
for run in range(N_runs):
    # Chama a função única que faz tudo
    rmse_pos, rmse_heading, t, x_hist_true, x_hist_est = run_bc_ekf(
        T,
        t_final,
        anchors,
        v_true,
        w_true,
        l,
        z_c,
        sigma_v,
        sigma_w,
        sigma_uwb
    )

    # Guarda os resultados
    rmse_pos_list.append(rmse_pos)
    rmse_heading_list.append(rmse_heading)

    print(f"Execução {run+1}/{N_runs} - RMSE posição: {rmse_pos:.4f} m - RMSE heading: {rmse_heading:.4f} graus")

# ============================
# 3. Estatísticas finais
# ============================
mean_rmse_pos = np.mean(rmse_pos_list)
std_rmse_pos = np.std(rmse_pos_list)
mean_rmse_heading = np.mean(rmse_heading_list)
std_rmse_heading = np.std(rmse_heading_list)

print("\n==========================")
print("Monte Carlo concluído.")
print(f"RMSE posição média: {mean_rmse_pos:.4f} ± {std_rmse_pos:.4f} m")
print(f"RMSE heading média: {mean_rmse_heading:.4f} ± {std_rmse_heading:.4f} graus")

# ============================
# 4. Boxplots dos RMSEs
# ============================
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.boxplot(rmse_pos_list)
plt.title("RMSE Posição (m)")
plt.ylabel("Erro (m)")
plt.grid(True)

plt.subplot(1,2,2)
plt.boxplot(rmse_heading_list)
plt.title("RMSE Heading (graus)")
plt.ylabel("Erro (graus)")
plt.grid(True)

plt.tight_layout()
plt.show()
