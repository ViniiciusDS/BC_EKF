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

# Para armazenar erros ao longo do tempo
pos_errors_all_runs = []
heading_errors_all_runs = []

# Para armazenar trajetórias
trajectories_true = []
trajectories_est = []

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

    ## Guarda os resultados
    # Calcula erros ponto a ponto
    error = x_hist_true - x_hist_est
    error[2,:] = np.arctan2(np.sin(error[2,:]), np.cos(error[2,:]))
    pos_error = np.linalg.norm(error[0:2,:], axis=0)
    heading_error_deg = np.abs(error[2,:])*(180/np.pi)

    # Guarda RMSE
    rmse_pos_list.append(rmse_pos)
    rmse_heading_list.append(rmse_heading)

    # Guarda erros ao longo do tempo
    pos_errors_all_runs.append(pos_error)
    heading_errors_all_runs.append(heading_error_deg)

    # Guarda trajetórias
    trajectories_true.append(x_hist_true.copy())
    trajectories_est.append(x_hist_est.copy())

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

# ============================
# 5. Erro ao longo do tempo
# ============================
pos_errors_all_runs = np.array(pos_errors_all_runs)        # shape: (N_runs, len(t))
heading_errors_all_runs = np.array(heading_errors_all_runs)

# Média e desvio padrão ao longo do tempo
mean_pos_error = np.mean(pos_errors_all_runs, axis=0)
std_pos_error = np.std(pos_errors_all_runs, axis=0)

mean_heading_error = np.mean(heading_errors_all_runs, axis=0)
std_heading_error = np.std(heading_errors_all_runs, axis=0)

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t, mean_pos_error, label="Média")
plt.fill_between(t, mean_pos_error - std_pos_error, mean_pos_error + std_pos_error, alpha=0.3)
plt.title("Erro de Posição ao Longo do Tempo")
plt.ylabel("Erro (m)")
plt.grid()
plt.legend()

plt.subplot(2,1,2)
plt.plot(t, mean_heading_error, label="Média")
plt.fill_between(t, mean_heading_error - std_heading_error, mean_heading_error + std_heading_error, alpha=0.3)
plt.title("Erro de Orientação ao Longo do Tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("Erro (graus)")
plt.grid()
plt.legend()

plt.tight_layout()

# ============================
# 6. Trajetórias aleatórias
# ============================
np.random.seed(42)  # Para reprodutibilidade
idx_examples = np.random.choice(N_runs, size=5, replace=False)

plt.figure(figsize=(12,10))

for i, idx in enumerate(idx_examples, 1):
    true_traj = trajectories_true[idx]
    est_traj = trajectories_est[idx]

    plt.subplot(3,2,i)
    plt.plot(true_traj[0,:], true_traj[1,:], 'k-', label="Real")
    plt.plot(est_traj[0,:], est_traj[1,:], 'b--', label="Estimado")
    plt.scatter(anchors[0,:], anchors[1,:], c='r', marker='*', s=80, label="Âncoras")
    plt.title(f"Execução {idx+1}")
    plt.axis("equal")
    plt.grid()
    if i==1:
        plt.legend()

plt.tight_layout()

plt.show()

