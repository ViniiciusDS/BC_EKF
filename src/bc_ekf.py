# bc_ekf.py
# EKF (passo unico + wrapper) para robô diferencial com UWB
import numpy as np
import src.utils as utils
from typing import Optional

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
    sigma_uwb=np.sqrt(0.0025),
    x0=None,
):
    """
    Executa o EKF em uma simulação com velocidades constantes.
    """
    t = np.arange(0, t_final + T, T)
    num_anchors = anchors.shape[1]
    update_ratio = int((1/T) / 5)  # Correção UWB a cada 0.2s

    # Trajetória real
    x_hist_true = np.zeros((3, len(t)))
    if x0 is None:
        x_hist_true[:, 0] = [2.5, 0.0, 0.0]
    else:
        x_hist_true[:, 0] = np.asarray(x0, dtype=float).reshape(3,)

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
    x_est = x_hist_true[:, 0].copy()
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
    sigma_uwb=np.sqrt(0.0025),
    x0=None,
):
    """
    Executa EKF com comandos de velocidade variáveis ao longo do tempo.
    """
    t = np.arange(0, t_final, T)
    num_anchors = anchors.shape[1]
    update_ratio = int((1/T)/5)

    # Trajetória real
    x_hist_true = np.zeros((3, len(t)))
    if x0 is None:
        x_hist_true[:, 0] = [2.5, 0.0, 0.0]
    else:
        x_hist_true[:, 0] = np.asarray(x0, dtype=float).reshape(3,)

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
    x_est = x_hist_true[:, 0].copy()
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

def run_bc_ekf_from_data(
    T,
    anchors,
    odometry_noisy,
    z_hist,
    l,
    z_c,
    sigma_uwb,
    x0=None,
    update_hz=5.0,
):
    """
    Roda o BC-EKF recebendo diretamente odometria e medições UWB.

    Compatível com datasets simulados e reais.
    Para datasets reais com T maior que 0,2 s, a correção UWB passa a ocorrer
    em toda amostra, evitando update_ratio = 0.
    """
    T = float(T)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"T inválido no BC-EKF: {T}")

    anchors = np.asarray(anchors, dtype=float)
    odometry_noisy = np.asarray(odometry_noisy, dtype=float)
    z_hist = np.asarray(z_hist, dtype=float)

    num_steps = int(odometry_noisy.shape[1])
    num_anchors = int(anchors.shape[1])

    if z_hist.shape[1] != num_steps:
        raise ValueError(
            f"z_hist possui {z_hist.shape[1]} passos, mas odometria possui {num_steps}"
        )

    if z_hist.shape[0] != 2 * num_anchors:
        raise ValueError(
            f"z_hist possui {z_hist.shape[0]} linhas, esperado {2 * num_anchors}"
        )

    # Inicialização
    if x0 is None:
        x_est = np.array([2.5, 0.0, 0.0], dtype=float)
    else:
        x_est = np.asarray(x0, dtype=float).reshape(3,)

    P = np.diag([0.1, 0.1, 0.1])
    Q = np.diag([1e-4] * 3)
    R = np.diag([float(sigma_uwb) ** 2] * (2 * num_anchors))

    x_hist_est = np.zeros((3, num_steps), dtype=float)
    x_hist_est[:, 0] = x_est

    # Correção UWB a update_hz, mas nunca deixa update_ratio virar zero.
    # Se T for maior que o período desejado, corrige em toda amostra.
    desired_update_period = 1.0 / float(update_hz)
    update_ratio = max(1, int(round(desired_update_period / T)))

    print(
        "[BC_EKF_FROM_DATA]",
        "T=", T,
        "num_steps=", num_steps,
        "num_anchors=", num_anchors,
        "update_ratio=", update_ratio,
        "x0=", x_est,
    )

    # Loop EKF
    for k in range(1, num_steps):
        v_k = odometry_noisy[0, k]
        w_k = odometry_noisy[1, k]

        x_pred, A_k = _predict_state(x_est, v_k, w_k, T)
        P_pred = A_k @ P @ A_k.T + Q

        if k % update_ratio == 0:
            h_pred, H_k = _measurement_model(x_pred, anchors, l, z_c)

            z_k = z_hist[:, k]
            if np.all(np.isfinite(z_k)):
                S = H_k @ P_pred @ H_k.T + R
                K_k = P_pred @ H_k.T @ np.linalg.inv(S)

                x_est = x_pred + K_k @ (z_k - h_pred)
                x_est[2] = np.arctan2(np.sin(x_est[2]), np.cos(x_est[2]))
                P = (np.eye(3) - K_k @ H_k) @ P_pred
            else:
                x_est = x_pred
                P = P_pred
        else:
            x_est = x_pred
            P = P_pred

        x_hist_est[:, k] = x_est

    return x_hist_est


def run_bc_ekf_step(
        x_est: np.ndarray,
        P: np.ndarray, 
        u_k: np.ndarray, 
        z_k: np.ndarray, 
        anchors: np.ndarray, 
        l: float, 
        z_c: float, 
        Q: np.ndarray, 
        R: np.ndarray, 
        dt: Optional[float] = None, 
        debug: bool = False
        ):
    """
    Executa UM passo do BC-EKF (predição +, opcionalmente, correção).
    - Aceita z_k vazio e ignora a correção se não houver medições.
    - dt explicito para coerencia com o modelo discreto.
    - Quando debug=True, retorna também um dicionário de diagnósticos com:
      {'innov': y, 'S': S, 'K': K, 'h': h, 'H': H}

    Args:
    x_est: estado estimado corrente (shape (3,))
    P: matriz de covariância atual (3x3)
    u_k: comando de controle [v, w]
    z_k: vetor de medições UWB (2*N anchors) ou vazio
    anchors: matriz 3xN com posições das âncoras
    l: metade do baseline entre as tags
    z_c: altura das tags
    Q: covariância do processo (3x3)
    R: covariância das medições (2N x 2N ou vazia)
    dt: passo de tempo
    debug: se True, retorna também dicionário de debug

    Retornos:
      - Se debug=False (padrão): (x_next, P_next)
      - Se debug=True: (x_next, P_next, diag_dict)
    """
    if dt is None:
        raise ValueError("run_bc_ekf_step: dt é None; passe dt=... (time step)")
    v, w = u_k

    # --- Predição discreta com dt ---
    theta = x_est[2]
    dx   = v * dt * np.cos(theta + w*dt/2.0)
    dy   = v * dt * np.sin(theta + w*dt/2.0)
    dth  = w * dt

    x_pred = x_est + np.array([dx, dy, dth])
    x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))

    # Jacobiano do processo (discreto) w.r.t. estado
    A = np.array([
        [1, 0, -v * dt * np.sin(theta + w*dt/2.0)],
        [0, 1,  v * dt * np.cos(theta + w*dt/2.0)],
        [0, 0, 1]
    ])
    P_pred = A @ P @ A.T + Q

    # --- Correção: checagens rápidas ---
    # Sem medições ou R inválido -> sai só com predição
    if z_k is None or len(z_k) == 0 or R is None or R.size == 0:
        if debug:
            return x_pred, P_pred, {
                "innov": None, "S": None, "K": None, "h": None, "H": None,
                "x_pred": x_pred.copy(), "P_pred": P_pred.copy(), "dt": dt
            }
        return x_pred, P_pred

    num_anchors = anchors.shape[1]
    if num_anchors == 0:
        if debug:
            return x_pred, P_pred, {
                "innov": None, "S": None, "K": None, "h": None, "H": None,
                "x_pred": x_pred.copy(), "P_pred": P_pred.copy(), "dt": dt
            }
        return x_pred, P_pred

    # --- Modelo de medição (h, H) em x_pred ---
    xp, yp, th = x_pred
    pf = np.array([xp + l*np.cos(th), yp + l*np.sin(th), z_c])
    pr = np.array([xp - l*np.cos(th), yp - l*np.sin(th), z_c])

    h = np.zeros(2*num_anchors)
    H = np.zeros((2*num_anchors, 3))

    for i in range(num_anchors):
        a = anchors[:, i]
        Df = np.linalg.norm(pf - a)
        Dr = np.linalg.norm(pr - a)

        # Evitar divisões por zero
        if Df < 1e-9 or Dr < 1e-9:
            if debug:
                return x_pred, P_pred, {
                    "innov": None, "S": None, "K": None, "h": None, "H": None,
                    "x_pred": x_pred.copy(), "P_pred": P_pred.copy(), "dt": dt,
                    "warn": "Distância tag-âncora ~ 0"
                }
            return x_pred, P_pred

        h[2*i]     = Df
        h[2*i + 1] = Dr

        Cf = - (pf[0]-a[0]) * l*np.sin(th) + (pf[1]-a[1]) * l*np.cos(th)
        Cr =   (pr[0]-a[0]) * l*np.sin(th) - (pr[1]-a[1]) * l*np.cos(th)

        H[2*i, :]     = [(pf[0]-a[0]) / Df, (pf[1]-a[1]) / Df, Cf / Df]
        H[2*i + 1, :] = [(pr[0]-a[0]) / Dr, (pr[1]-a[1]) / Dr, Cr / Dr]

    # --- Atualização de Kalman robusta ---
    # Checagem de dimensões
    m = 2 * num_anchors
    if z_k.shape[0] != m or R.shape != (m, m):
        raise ValueError(f"z_k/R mismatch: z_k={z_k.shape}, R={R.shape}, expected m={m}")
    
    S = H @ P_pred @ H.T + R
    # regularização leve em S (caso mal-condicionado)
    eps = 1e-9
    S = S + eps * np.eye(S.shape[0])

    y = z_k - h

    # Use solve ao invés de inv
    try:
        assert S.shape[0] == S.shape[1] == H.shape[0], f"Dimensão inconsistente: S={S.shape}, H={H.shape}, P_pred={P_pred.shape}"
        # S: (m,m), m = 2*num_anchors
        # Queremos K = P_pred H^T S^{-1}
        # Resolva:  S^T X^T = (P_pred H^T)^T  ->  X = P_pred H^T S^{-1}
        PHt = P_pred @ H.T                 # (3,m)
        # resolve S^T X^T = PHt^T  -> X = PHt @ S^{-1}
        K = np.linalg.solve(S.T, PHt.T).T  # (3,m)  dimensões batem sempre
    except np.linalg.LinAlgError:
        # fallback numérico
        Sinv = np.linalg.pinv(S)
        K = (P_pred @ H.T) @ Sinv

    x_upd = x_pred + K @ y
    x_upd[2] = np.arctan2(np.sin(x_upd[2]), np.cos(x_upd[2]))
    P_upd = (np.eye(3) - K @ H) @ P_pred

    if debug:
        return x_upd, P_upd, {
            "innov": y, "S": S, "K": K, "h": h, "H": H,
            "x_pred": x_pred.copy(), "P_pred": P_pred.copy(), "dt": dt
        }
    return x_upd, P_upd

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
            dist_f = utils.apply_uwb_errors(np.linalg.norm(pf - anchors[:,i]), sigma_uwb)
            dist_r = utils.apply_uwb_errors(np.linalg.norm(pr - anchors[:,i]), sigma_uwb)
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