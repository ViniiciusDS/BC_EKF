# sim_artigo.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import simulate_run
from bc_ekf import run_bc_ekf_from_data
from scenarios import anchors_tectrol
import config
import os


os.makedirs("resultados", exist_ok=True)

# ======================
# Configuração geral
# ======================
N_MONTE_CARLO = 50
T = 0.05
t_final = 50
z_c = 0.5
sigma_v = 0.02
sigma_w = 0.05

# Ranges de teste
ANCHOR_COUNTS = [3, 4, 5, 6, 7]        # quantidade de âncoras
BASELINES = [0.3, 0.5, 0.65, 0.85]     # baseline (m)
SIGMAS_UWB = [0.05, 0.1, 0.2]          # ruído UWB (m)

# ======================
# Função para executar um experimento Monte Carlo
# ======================
def run_experiment(anchors, baseline, sigma_uwb):
    rmse_pos_list = []
    rmse_heading_list = []
    for _ in range(N_MONTE_CARLO):
        # Simula a trajetória realista com ruído
        t, x_true, v_noisy, w_noisy, z_hist = simulate_run(
            T, t_final, anchors, 0.3, np.deg2rad(7.5), baseline, z_c, sigma_v, sigma_w, sigma_uwb
        )
        # Executa EKF com os dados simulados
        x_est = run_bc_ekf_from_data(
            T, anchors, np.vstack((v_noisy, w_noisy)), z_hist, baseline, z_c, sigma_uwb
        )
        # Calcula RMSE
        error = x_true - x_est
        error[2,:] = np.arctan2(np.sin(error[2,:]), np.cos(error[2,:]))
        pos_error = np.linalg.norm(error[0:2,:], axis=0)
        heading_error_deg = np.abs(error[2,:])*(180/np.pi)
        rmse_pos = np.sqrt(np.mean(pos_error**2))
        rmse_heading = np.sqrt(np.mean(heading_error_deg**2))
        rmse_pos_list.append(rmse_pos)
        rmse_heading_list.append(rmse_heading)
    return np.mean(rmse_pos_list), np.std(rmse_pos_list), np.mean(rmse_heading_list), np.std(rmse_heading_list)

# ======================
# Rodar todos os experimentos
# ======================
results = []
for n_anchors in ANCHOR_COUNTS:
    for baseline in BASELINES:
        for sigma_uwb in SIGMAS_UWB:
            anchors = anchors_tectrol[:, :n_anchors]  # seleciona subconjunto de âncoras
            mean_pos, std_pos, mean_heading, std_heading = run_experiment(anchors, baseline/2, sigma_uwb)
            results.append({
                "anchors": n_anchors,
                "baseline": baseline,
                "sigma_uwb": sigma_uwb,
                "rmse_pos_mean": mean_pos,
                "rmse_pos_std": std_pos,
                "rmse_heading_mean": mean_heading,
                "rmse_heading_std": std_heading
            })
            print(f"Anchors={n_anchors}, Baseline={baseline}, Sigma={sigma_uwb} => "
                  f"RMSE Pos={mean_pos:.3f}m, RMSE Heading={mean_heading:.3f}°")

# ======================
# Exportar resultados
# ======================
df = pd.DataFrame(results)
df.to_csv("resultados/resultados_artigo.csv", sep=";", index=False)
print("\nResultados salvos em resultados_artigo.csv")

# ======================
# Heatmaps para visualização
# ======================
def plot_heatmap(df, x_param, y_param, value_param, title, cmap="viridis"):
    pivot_table = df.pivot_table(index=y_param, columns=x_param, values=value_param)
    plt.figure(figsize=(8,6))
    plt.imshow(pivot_table, cmap=cmap, aspect='auto', origin='lower')
    plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
    plt.yticks(range(len(pivot_table.index)), pivot_table.index)
    plt.colorbar(label=value_param)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Heatmaps: RMSE posição
for sigma in SIGMAS_UWB:
    subset = df[df["sigma_uwb"] == sigma]
    plot_heatmap(subset, "anchors", "baseline", "rmse_pos_mean",
                 f"RMSE Posição (m) - Ruído={sigma}m")

# Heatmaps: RMSE heading
for sigma in SIGMAS_UWB:
    subset = df[df["sigma_uwb"] == sigma]
    plot_heatmap(subset, "anchors", "baseline", "rmse_heading_mean",
                 f"RMSE Heading (°) - Ruído={sigma}m")

def plot_bar_comparisons(results_csv_path):
    """
    Lê o CSV de resultados e plota gráficos de barras comparando RMSE médio por variável.
    """
    import pandas as pd
    df = pd.read_csv(results_csv_path, sep=";")

    # Gráfico 1: Impacto da quantidade de âncoras
    anchors_group = df.groupby("anchors")["rmse_pos_mean"].mean()
    plt.figure(figsize=(8,5))
    anchors_group.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Impacto da Quantidade de Âncoras no RMSE de Posição")
    plt.xlabel("Quantidade de Âncoras")
    plt.ylabel("RMSE de Posição (m)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("resultados/bar_anchors_vs_rmse.png", dpi=300)
    plt.show()

    # Gráfico 2: Impacto do baseline
    baseline_group = df.groupby("baseline")["rmse_pos_mean"].mean()
    plt.figure(figsize=(8,5))
    baseline_group.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title("Impacto do Baseline no RMSE de Posição")
    plt.xlabel("Baseline (m)")
    plt.ylabel("RMSE de Posição (m)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("resultados/bar_baseline_vs_rmse.png", dpi=300)
    plt.show()

    # Gráfico 3: Impacto do ruído UWB
    noise_group = df.groupby("sigma_uwb")["rmse_pos_mean"].mean()
    plt.figure(figsize=(8,5))
    noise_group.plot(kind='bar', color='salmon', edgecolor='black')
    plt.title("Impacto do Ruído UWB no RMSE de Posição")
    plt.xlabel("Desvio Padrão do Ruído (m)")
    plt.ylabel("RMSE de Posição (m)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("resultados/bar_noise_vs_rmse.png", dpi=300)
    plt.show()

# Chamar após salvar o CSV
plot_bar_comparisons("resultados/resultados_artigo.csv")