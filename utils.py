# utils.py
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def save_data(filename, data, headers, precision=5):
    """
    Salva dados em CSV com precisão fixa e separador ;
    """
    # Cria DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    # Formata cada coluna numérica
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(lambda x: round(x, precision))
    
    # Salva com ponto decimal e separador ;
    df.to_csv(filename, index=False, sep=";")
def plot_trajectory(map_size, waypoints, path):
    plt.figure(figsize=(6,6))
    plt.xlim(0, map_size[0])
    plt.ylim(0, map_size[1])
    # Desenhar waypoints
    wp_x, wp_y = zip(*waypoints)
    plt.plot(wp_x, wp_y, "ro--", label="Waypoints")
    # Caminho percorrido
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    plt.plot(path_x, path_y, "b-", label="Trajetória Percorrida")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.title("Simulação de Trajetória")
    plt.show()

def simulate_run(T, t_final, anchors, v_true, w_true, l, z_c, sigma_v, sigma_w, sigma_uwb):
    """
    Simula a trajetória real e gera medições ruidosas.
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
        x_hist_true[2,k] = x_hist_true[2,k-1] + w_true*T
        x_hist_true[2,k] = np.arctan2(np.sin(x_hist_true[2,k]), np.cos(x_hist_true[2,k]))

    # Odometria ruidosa
    v_noisy = v_true + sigma_v*np.random.randn(len(t))
    w_noisy = w_true + sigma_w*np.random.randn(len(t))

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



