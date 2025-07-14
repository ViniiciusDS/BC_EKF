# montecarlo_runner.py
import numpy as np
from bc_ekf import run_bc_ekf

def run_monte_carlo(
    T,
    t_final,
    anchors,
    v_true,
    w_true,
    l,
    z_c,
    sigma_v,
    sigma_w,
    sigma_uwb,
    N_runs
):
    """
    Executa múltiplas simulações (Monte Carlo) do EKF e coleta estatísticas.
    
    Args:
        T (float): Intervalo de tempo (s).
        t_final (float): Tempo total da simulação (s).
        anchors (ndarray): Matriz 3xN de âncoras.
        v_true (float): Velocidade linear verdadeira.
        w_true (float): Velocidade angular verdadeira.
        l (float): Metade do baseline.
        z_c (float): Altura das tags.
        sigma_v (float): Ruído na velocidade linear.
        sigma_w (float): Ruído na velocidade angular.
        sigma_uwb (float): Ruído nas medições UWB.
        N_runs (int): Número de execuções Monte Carlo.
    
    Returns:
        Tuple contendo arrays de RMSEs, erros ao longo do tempo e trajetórias.
    """
    rmse_pos_list = []
    rmse_heading_list = []
    pos_errors_all_runs = []
    heading_errors_all_runs = []
    trajectories_true = []
    trajectories_est = []

    for run in range(N_runs):
        # Executa uma simulação EKF
        rmse_pos, rmse_heading, t, x_hist_true, x_hist_est = run_bc_ekf(
            T, t_final, anchors, v_true, w_true, l, z_c,
            sigma_v, sigma_w, sigma_uwb
        )

        # Calcula erros ponto a ponto
        error = x_hist_true - x_hist_est
        error[2, :] = np.arctan2(np.sin(error[2, :]), np.cos(error[2, :]))
        pos_error = np.linalg.norm(error[0:2, :], axis=0)
        heading_error_deg = np.abs(error[2, :]) * (180 / np.pi)

        # Armazena resultados
        rmse_pos_list.append(rmse_pos)
        rmse_heading_list.append(rmse_heading)
        pos_errors_all_runs.append(pos_error)
        heading_errors_all_runs.append(heading_error_deg)
        trajectories_true.append(x_hist_true)
        trajectories_est.append(x_hist_est)

    return (
        np.array(rmse_pos_list),
        np.array(rmse_heading_list),
        np.array(pos_errors_all_runs),
        np.array(heading_errors_all_runs),
        trajectories_true,
        trajectories_est
    )
