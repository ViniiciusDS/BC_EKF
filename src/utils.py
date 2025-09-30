# utils.py
# ruido, I/O, helpers
import os, csv, time, json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import src.config as config
from datetime import datetime
from typing import Optional

def save_data(filename, data, headers, precision=5):
    """
    Salva dados em CSV com separador ; e precisão configurável.
    """
    df = pd.DataFrame(data, columns=headers)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(lambda x: round(x, precision))
    df.to_csv(filename, index=False, sep=";")

def plot_trajectory(map_size, waypoints, path):
    """
    Plota um mapa com waypoints e trajetória percorrida.
    """
    plt.figure(figsize=(6,6))
    plt.xlim(0, map_size[0])
    plt.ylim(0, map_size[1])

    # Waypoints
    wp_x, wp_y = zip(*waypoints)
    plt.plot(wp_x, wp_y, "ro--", label="Waypoints")

    # Trajetória
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    plt.plot(path_x, path_y, "b-", label="Trajetória")

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Simulação de Trajetória")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def simulate_run(T, t_final, anchors, v_true, w_true, l, z_c, sigma_v, sigma_w, sigma_uwb):
    """
    Simula a trajetória real e gera medições ruidosas (versão simplificada).
    """
    t = np.arange(0, t_final+T, T)
    num_anchors = anchors.shape[1]

    # Trajetória real
    x_hist_true = np.zeros((3, len(t)))
    x_hist_true[:,0] = [2.5, 0, 0]
    for k in range(1, len(t)):
        theta = x_hist_true[2,k-1]
        x_hist_true[0,k] = x_hist_true[0,k-1] + v_true*T*np.cos(theta + w_true*T/2)
        x_hist_true[1,k] = x_hist_true[1,k-1] + v_true*T*np.sin(theta + w_true*T/2)
        x_hist_true[2,k] = np.arctan2(np.sin(x_hist_true[2,k-1] + w_true*T), np.cos(x_hist_true[2,k-1] + w_true*T))

    # Odometria ruidosa
    v_noisy = v_true + sigma_v * np.random.randn(len(t))
    w_noisy = w_true + sigma_w * np.random.randn(len(t))

    # Medições UWB
    z_hist = np.zeros((2*num_anchors, len(t)))
    for k in range(len(t)):
        theta = x_hist_true[2,k]
        xt, yt = x_hist_true[0,k], x_hist_true[1,k]
        pf = np.array([xt + l*np.cos(theta), yt + l*np.sin(theta), z_c])
        pr = np.array([xt - l*np.cos(theta), yt - l*np.sin(theta), z_c])
        for i in range(num_anchors):
            dist_f = apply_uwb_errors(np.linalg.norm(pf - anchors[:,i]), sigma_uwb)
            dist_r = apply_uwb_errors(np.linalg.norm(pr - anchors[:,i]), sigma_uwb)
            z_hist[2*i,k] = dist_f
            z_hist[2*i+1,k] = dist_r

    return t, x_hist_true, v_noisy, w_noisy, z_hist

def generate_ground_truth(t, v_true, w_true):
    """
    Gera a trajetória real do robô diferencial ao longo do tempo.

    Args:
        t (ndarray): Vetor de tempo.
        v_true (float): Velocidade linear constante.
        w_true (float): Velocidade angular constante.

    Returns:
        x_hist (ndarray): Matriz 3 x N com [x, y, theta] ao longo do tempo.
    """
    x_hist = np.zeros((3, len(t)))
    x_hist[:, 0] = [2.5, 0, 0]  # Estado inicial

    for k in range(1, len(t)):
        theta_prev = x_hist[2, k-1]
        x_hist[0, k] = x_hist[0, k-1] + v_true * (t[k] - t[k-1]) * np.cos(theta_prev + w_true*(t[k] - t[k-1])/2)
        x_hist[1, k] = x_hist[1, k-1] + v_true * (t[k] - t[k-1]) * np.sin(theta_prev + w_true*(t[k] - t[k-1])/2)
        x_hist[2, k] = np.arctan2(
            np.sin(x_hist[2, k-1] + w_true*(t[k] - t[k-1])),
            np.cos(x_hist[2, k-1] + w_true*(t[k] - t[k-1]))
        )
    return x_hist


def generate_noisy_odometry(v_true, w_true, t, sigma_v, sigma_w):
    """
    Gera odometria ruidosa ao longo do tempo.

    Args:
        v_true (float): Velocidade linear constante.
        w_true (float): Velocidade angular constante.
        t (ndarray): Vetor de tempo.
        sigma_v (float): Desvio padrão do ruído linear.
        sigma_w (float): Desvio padrão do ruído angular.

    Returns:
        Tuple (v_noisy, w_noisy): Arrays de velocidades ruidosas.
    """
    v_noisy = v_true + sigma_v * np.random.randn(len(t))
    w_noisy = w_true + sigma_w * np.random.randn(len(t))
    return v_noisy, w_noisy


def generate_uwb_measurements(x_hist, anchors, l, z_c, sigma_uwb):
    """
    Simula medições UWB ruidosas a partir da trajetória real.

    Args:
        x_hist (ndarray): Matriz 3 x N com [x, y, theta] ao longo do tempo.
        anchors (ndarray): Matriz 3 x num_anchors das posições das âncoras.
        l (float): Metade do baseline.
        z_c (float): Altura das tags.
        sigma_uwb (float): Desvio padrão do ruído UWB.

    Returns:
        z_hist (ndarray): Medições simuladas (2*num_anchors x N).
    """
    num_anchors = anchors.shape[1]
    z_hist = np.zeros((2 * num_anchors, x_hist.shape[1]))

    for k in range(x_hist.shape[1]):
        theta = x_hist[2, k]
        xt, yt = x_hist[0, k], x_hist[1, k]
        pf = np.array([xt + l*np.cos(theta), yt + l*np.sin(theta), z_c])
        pr = np.array([xt - l*np.cos(theta), yt - l*np.sin(theta), z_c])

        for i in range(num_anchors):
            dist_f = np.linalg.norm(pf - anchors[:, i]) + sigma_uwb * np.random.randn()
            dist_r = np.linalg.norm(pr - anchors[:, i]) + sigma_uwb * np.random.randn()
            z_hist[2*i, k] = dist_f
            z_hist[2*i+1, k] = dist_r

    return z_hist

def generate_uwb_single_measurement(x_state, anchors, l, z_c, sigma_uwb):
    """
    Gera medições UWB para um único estado do robô (não histórico).
    Args:
        x_state: lista ou array [x, y, theta].
        anchors: matriz 3xN com posições das âncoras.
        l: metade do baseline.
        z_c: altura das tags.
        sigma_uwb: desvio padrão do ruído.
    Returns:
        z_k: vetor de medições UWB (2*num_anchors, ).
    """
    if anchors is None:
        return np.array([])
    num_anchors = anchors.shape[1]
    if num_anchors == 0:
        return np.array([])
    
    xk, yk, th = x_state
    pf = np.array([xk + l*np.cos(th), yk + l*np.sin(th), z_c])
    pr = np.array([xk - l*np.cos(th), yk - l*np.sin(th), z_c])
    
    num_anchors = anchors.shape[1]
    z_k = np.zeros(2 * num_anchors)

    for i in range(num_anchors):
        a = anchors[:, i]
        z_k[2*i]     = np.linalg.norm(pf - a) + sigma_uwb*np.random.randn()
        z_k[2*i + 1] = np.linalg.norm(pr - a) + sigma_uwb*np.random.randn()
    return z_k

def apply_uwb_errors(base_distance, sigma_uwb):
    """Aplica viés e desalinhamento às medições UWB."""
    # Erro de viés aleatório
    bias = 0.0
    if config.UWB_BIAS_ENABLED and np.random.rand() < config.UWB_BIAS_PROBABILITY:
        bias = np.random.choice([-1, 1]) * config.UWB_BIAS_VALUE

    # Ruído de desalinhamento
    noise_factor = 1.0
    if config.UWB_MISALIGNMENT_ENABLED and np.random.rand() < config.UWB_MISALIGNMENT_PROBABILITY:
        noise_factor = config.UWB_MISALIGNMENT_FACTOR

    return base_distance + bias + noise_factor * sigma_uwb * np.random.randn()

def _makedirs_silent(path: str):
    os.makedirs(path, exist_ok=True)

class RunLogger:
    """
    Logger simples de execução:
      - meta.json: guarda metadados da simulação (config, âncoras, rota etc.)
      - data.csv: guarda amostras por passo (tempo, estados, comandos, erros)
    """
    def __init__(self,
                 out_dir: str,
                 run_name: Optional[str] = None,
                 meta: Optional[dict] = None,
                 flush_every_n: int = 200):

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{ts}"
        self.root = os.path.join(out_dir, self.run_name)
        _makedirs_silent(self.root)

        # salva metadados
        self.meta_path = os.path.join(self.root, "meta.json")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta or {}, f, ensure_ascii=False, indent=2)

        # prepara CSV
        self.csv_path = os.path.join(self.root, "data.csv")
        self._fh = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh, delimiter=";")

        header = [
            "t_sec",
            # true
            "x_true", "y_true", "theta_true",
            # pred
            "x_pred", "y_pred", "theta_pred",
            # est
            "x_est", "y_est", "theta_est",
            # comandos / entradas
            "v_cmd", "w_cmd", "v_meas", "w_meas",
            # erros
            "pos_err_m", "heading_err_deg"
        ]
        self._writer.writerow(header)
        self._count = 0
        self._t0 = time.time()
        self._flush_every_n = max(1, int(flush_every_n))

    @property
    def out_path(self) -> str:
        return self.root

    def log_step(self,
                 true_state,      # [x,y,theta]
                 pred_state,      # [x,y,theta] (opcional: pode ser None)
                 est_state,       # [x,y,theta]
                 v_cmd, w_cmd,
                 v_meas, w_meas,
                 pos_err, heading_err_deg):
        t_sec = time.time() - self._t0
        x_pred, y_pred, th_pred = (pred_state if pred_state is not None
                                   else (float("nan"),) * 3)

        row = [
            f"{t_sec:.5f}",
            f"{true_state[0]:.6f}", f"{true_state[1]:.6f}", f"{true_state[2]:.6f}",
            f"{x_pred:.6f}", f"{y_pred:.6f}", f"{th_pred:.6f}",
            f"{est_state[0]:.6f}", f"{est_state[1]:.6f}", f"{est_state[2]:.6f}",
            f"{v_cmd:.6f}", f"{w_cmd:.6f}", f"{v_meas:.6f}", f"{w_meas:.6f}",
            f"{pos_err:.6f}", f"{heading_err_deg:.6f}",
        ]
        self._writer.writerow(row)
        self._count += 1

        if self._count % self._flush_every_n == 0:
            self._fh.flush()

    def close(self):
        try:
            self._fh.flush()
        finally:
            self._fh.close()



#   Noise functions
rng = np.random.default_rng()

def add_gaussian_noise(value, std_dev):
    """
    Retorna 'value' com ruído gaussiano de desvio padrão 'std_dev'.
    """
    return value + rng.normal(0, std_dev)
