# utils.py
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
            dist_f = np.linalg.norm(pf - anchors[:,i]) + sigma_uwb*np.random.randn()
            dist_r = np.linalg.norm(pr - anchors[:,i]) + sigma_uwb*np.random.randn()
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