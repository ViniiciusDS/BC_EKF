# bc_ekf.py
import numpy as np

def run_bc_ekf(
    T,
    t_final,
    anchors,
    v_true=0.3,
    w_true=np.deg2rad(7.5),
    l=0.65/2,
    z_c=0.5,
    sigma_v=0.02,
    sigma_w=0.05,
    sigma_uwb=np.sqrt(0.0025)
):
    """
    Executa o EKF em uma simulação com velocidades constantes.
    """
    t = np.arange(0, t_final + T, T)
    num_anchors = anchors.shape[1]
    update_ratio = int((1/T) / 5)  # Correção UWB a cada 0.2s

    # Trajetória real
    x_hist_true = np.zeros((3, len(t)))
    x_hist_true[:,0] = [2.5, 0, 0]

    for k in range(1, len(t)):
        theta = x_hist_true[2,k-1]
        x_hist_true[0,k] = x_hist_true[0,k-1] + v_true*T*np.cos(theta + w_true*T/2)
        x_hist_true[1,k] = x_hist_true[1,k-1] + v_true*T*np.sin(theta + w_true*T/2)
        x_hist_true[2,k] = np.arctan2(np.sin(x_hist_true[2,k-1] + w_true*T), np.cos(x_hist_true[2,k-1] + w_true*T))

    # Odometria ruidosa
    input_noisy = np.vstack([
        v_true + sigma_v * np.random.randn(len(t)),
        w_true + sigma_w * np.random.randn(len(t))
    ])

    # Medidas UWB
    z_hist = _generate_uwb_measurements(x_hist_true, anchors, l, z_c, sigma_uwb)

    # Estimativa inicial e covariância
    x_est = np.array([2.5,0,0])
    P = np.diag([0.1,0.1,0.1])
    Q = np.diag([1e-4]*3)
    R = np.diag([sigma_uwb**2]*(2*num_anchors))
    x_hist_est = np.zeros((3, len(t)))
    x_hist_est[:,0] = x_est

    # Loop EKF
    for k in range(1, len(t)):
        v_k = input_noisy[0,k]
        w_k = input_noisy[1,k]
        x_pred, A_k = _predict_state(x_est, v_k, w_k, T)
        P_pred = A_k @ P @ A_k.T + Q

        if k % update_ratio == 0:
            h_pred, H_k = _measurement_model(x_pred, anchors, l, z_c)
            K_k = P_pred @ H_k.T @ np.linalg.inv(H_k @ P_pred @ H_k.T + R)
            z_k = z_hist[:,k]
            x_est = x_pred + K_k @ (z_k - h_pred)
            x_est[2] = np.arctan2(np.sin(x_est[2]), np.cos(x_est[2]))
            P = (np.eye(3) - K_k @ H_k) @ P_pred
        else:
            x_est = x_pred
            P = P_pred

        x_hist_est[:,k] = x_est

    rmse_pos, rmse_heading = _compute_rmse(x_hist_true, x_hist_est)

    return rmse_pos, rmse_heading, t, x_hist_true, x_hist_est


def run_bc_ekf_custom_commands(
    T,
    t_final,
    anchors,
    v_commands,
    w_commands,
    l=0.65/2,
    z_c=0.5,
    sigma_v=0.02,
    sigma_w=0.05,
    sigma_uwb=np.sqrt(0.0025)
):
    """
    Executa EKF com comandos de velocidade variáveis ao longo do tempo.
    """
    t = np.arange(0, t_final, T)
    num_anchors = anchors.shape[1]
    update_ratio = int((1/T)/5)

    # Trajetória real
    x_hist_true = np.zeros((3, len(t)))
    x_hist_true[:,0] = [2.5,0,0]
    for k in range(1, len(t)):
        v_k = v_commands[k]
        w_k = w_commands[k]
        theta = x_hist_true[2,k-1]
        x_hist_true[0,k] = x_hist_true[0,k-1] + v_k*T*np.cos(theta + w_k*T/2)
        x_hist_true[1,k] = x_hist_true[1,k-1] + v_k*T*np.sin(theta + w_k*T/2)
        x_hist_true[2,k] = np.arctan2(np.sin(x_hist_true[2,k-1]+w_k*T), np.cos(x_hist_true[2,k-1]+w_k*T))

    # Odometria ruidosa
    input_noisy = np.vstack([
        v_commands + sigma_v*np.random.randn(len(t)),
        w_commands + sigma_w*np.random.randn(len(t))
    ])

    # Medidas UWB
    z_hist = _generate_uwb_measurements(x_hist_true, anchors, l, z_c, sigma_uwb)

    # Inicialização
    x_est = np.array([2.5,0,0])
    P = np.diag([0.1,0.1,0.1])
    Q = np.diag([1e-4]*3)
    R = np.diag([sigma_uwb**2]*(2*num_anchors))
    x_hist_est = np.zeros((3, len(t)))
    x_hist_est[:,0] = x_est

    for k in range(1, len(t)):
        v_k = input_noisy[0,k]
        w_k = input_noisy[1,k]
        x_pred, A_k = _predict_state(x_est, v_k, w_k, T)
        P_pred = A_k @ P @ A_k.T + Q

        if k % update_ratio == 0:
            h_pred, H_k = _measurement_model(x_pred, anchors, l, z_c)
            K_k = P_pred @ H_k.T @ np.linalg.inv(H_k @ P_pred @ H_k.T + R)
            z_k = z_hist[:,k]
            x_est = x_pred + K_k @ (z_k - h_pred)
            x_est[2] = np.arctan2(np.sin(x_est[2]), np.cos(x_est[2]))
            P = (np.eye(3) - K_k @ H_k) @ P_pred
        else:
            x_est = x_pred
            P = P_pred

        x_hist_est[:,k] = x_est

    rmse_pos, rmse_heading = _compute_rmse(x_hist_true, x_hist_est)

    return rmse_pos, rmse_heading, t, x_hist_true, x_hist_est


# ======================
# Funções auxiliares
# ======================
def _predict_state(x_est, v, w, T):
    """
    Prediz o próximo estado do robô com base nas entradas de velocidade e no modelo de movimento diferencial.
    
    Args:
        x_est (ndarray): Estado estimado atual [x, y, theta].
        v (float): Velocidade linear.
        w (float): Velocidade angular.
        T (float): Intervalo de tempo (s).
        
    Returns:
        x_pred (ndarray): Estado previsto [x, y, theta].
        A_k (ndarray): Jacobiano da função de transição de estado.
    """
    theta = x_est[2]
    dx = v * T * np.cos(theta + w * T / 2)
    dy = v * T * np.sin(theta + w * T / 2)
    dtheta = w * T

    x_pred = x_est + np.array([dx, dy, dtheta])
    x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))  # normaliza ângulo

    A_k = np.array([
        [1, 0, -v * T * np.sin(theta + w * T / 2)],
        [0, 1,  v * T * np.cos(theta + w * T / 2)],
        [0, 0, 1]
    ])
    return x_pred, A_k


def _measurement_model(x_pred, anchors, l, z_c):
    """
    Calcula as distâncias esperadas das tags (frontal e traseira) às âncoras
    e o Jacobiano da função de medição.
    
    Args:
        x_pred (ndarray): Estado previsto [x, y, theta].
        anchors (ndarray): Matriz 3xN das posições das âncoras.
        l (float): Metade do baseline do robô.
        z_c (float): Altura fixa das tags.
        
    Returns:
        h_pred (ndarray): Vetor de medições esperadas.
        H_k (ndarray): Jacobiano da função de medição.
    """
    num_anchors = anchors.shape[1]
    xp, yp, theta_p = x_pred

    pf = np.array([xp + l*np.cos(theta_p), yp + l*np.sin(theta_p), z_c])
    pr = np.array([xp - l*np.cos(theta_p), yp - l*np.sin(theta_p), z_c])

    h_pred = np.zeros(2 * num_anchors)
    H_k = np.zeros((2 * num_anchors, 3))

    for i in range(num_anchors):
        # Distâncias previstas
        D_f = np.linalg.norm(pf - anchors[:, i])
        D_r = np.linalg.norm(pr - anchors[:, i])

        # Componentes parciais para derivada em relação ao ângulo
        C_f = -(pf[0] - anchors[0, i]) * l * np.sin(theta_p) + (pf[1] - anchors[1, i]) * l * np.cos(theta_p)
        C_r = (pr[0] - anchors[0, i]) * l * np.sin(theta_p) - (pr[1] - anchors[1, i]) * l * np.cos(theta_p)

        # Vetor de medições
        h_pred[2*i] = D_f
        h_pred[2*i + 1] = D_r

        # Jacobiano
        H_k[2*i, :] = [
            (pf[0] - anchors[0, i]) / D_f,
            (pf[1] - anchors[1, i]) / D_f,
            C_f / D_f
        ]
        H_k[2*i + 1, :] = [
            (pr[0] - anchors[0, i]) / D_r,
            (pr[1] - anchors[1, i]) / D_r,
            C_r / D_r
        ]

    return h_pred, H_k


def _generate_uwb_measurements(x_hist, anchors, l, z_c, sigma_uwb):
    """
    Simula medições UWB (distâncias) com ruído gaussiano ao longo da trajetória.
    
    Args:
        x_hist (ndarray): Trajetória real [3 x N].
        anchors (ndarray): Matriz 3xN das âncoras.
        l (float): Metade do baseline.
        z_c (float): Altura das tags.
        sigma_uwb (float): Desvio padrão do ruído UWB.
        
    Returns:
        z_hist (ndarray): Matriz 2*num_anchors x N com medições ruidosas.
    """
    num_anchors = anchors.shape[1]
    z_hist = np.zeros((2 * num_anchors, x_hist.shape[1]))

    for k in range(x_hist.shape[1]):
        theta = x_hist[2, k]
        xt, yt = x_hist[0, k], x_hist[1, k]
        pf = [xt + l*np.cos(theta), yt + l*np.sin(theta), z_c]
        pr = [xt - l*np.cos(theta), yt - l*np.sin(theta), z_c]
        for i in range(num_anchors):
            dist_f = np.linalg.norm(np.array(pf) - anchors[:, i]) + sigma_uwb * np.random.randn()
            dist_r = np.linalg.norm(np.array(pr) - anchors[:, i]) + sigma_uwb * np.random.randn()
            z_hist[2*i, k] = dist_f
            z_hist[2*i + 1, k] = dist_r
    return z_hist


def _compute_rmse(x_true, x_est):
    """
    Calcula RMSE de posição e orientação ao longo da trajetória.
    
    Args:
        x_true (ndarray): Trajetória real.
        x_est (ndarray): Trajetória estimada.
        
    Returns:
        rmse_pos (float): RMSE da posição Euclidiana.
        rmse_heading (float): RMSE do heading (graus).
    """
    error = x_true - x_est
    error[2, :] = np.arctan2(np.sin(error[2, :]), np.cos(error[2, :]))
    pos_error = np.linalg.norm(error[0:2, :], axis=0)
    heading_error_deg = np.abs(error[2, :]) * (180 / np.pi)
    rmse_pos = np.sqrt(np.mean(pos_error ** 2))
    rmse_heading = np.sqrt(np.mean(heading_error_deg ** 2))
    return rmse_pos, rmse_heading