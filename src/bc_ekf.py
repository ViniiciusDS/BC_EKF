# bc_ekf.py
# EKF (passo unico + wrapper) para robô diferencial com UWB
import numpy as np
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
    tag_offsets_xyz=None,
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
    z_hist = _generate_uwb_measurements(x_hist_true, anchors, l, z_c, sigma_uwb, tag_offsets_xyz=tag_offsets_xyz)

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
            h_pred, H_k = _measurement_model(x_pred, anchors, l, z_c, tag_offsets_xyz=tag_offsets_xyz)
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
    tag_offsets_xyz=None,
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
    z_hist = _generate_uwb_measurements(x_hist_true, anchors, l, z_c, sigma_uwb, tag_offsets_xyz=tag_offsets_xyz)

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
            h_pred, H_k = _measurement_model(x_pred, anchors, l, z_c, tag_offsets_xyz=tag_offsets_xyz)
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
    tag_offsets_xyz=None,
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
            h_pred, H_k = _measurement_model(x_pred, anchors, l, z_c, tag_offsets_xyz=tag_offsets_xyz)

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
        debug: bool = False,
        tag_offsets_xyz=None,
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

    # --- Modelo de medição compartilhado com batch/simulador ---
    # A ordem dos ranges permanece A0_canal0, A0_canal1, A1_canal0,
    # A1_canal1... independente da montagem física das tags.
    try:
        h, H = _measurement_model(
            x_pred, anchors, l, z_c, tag_offsets_xyz=tag_offsets_xyz
        )
    except ValueError as exc:
        if "Distância tag-âncora" not in str(exc):
            raise
        if debug:
            return x_pred, P_pred, {
                "innov": None, "S": None, "K": None, "h": None, "H": None,
                "x_pred": x_pred.copy(), "P_pred": P_pred.copy(), "dt": dt,
                "warn": str(exc),
            }
        return x_pred, P_pred

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


def _resolved_tag_offsets(l, z_c, tag_offsets_xyz=None):
    """Matriz 2x3 em referencial robô (x frente, y esquerda, z cima).

    Ausência da opção reproduz o modelo histórico: frente/trás
    com distância 2*l e altura z_c. Não reordena os canais do log.
    """
    if tag_offsets_xyz is None:
        offsets = np.array([[l, 0., z_c], [-l, 0., z_c]], dtype=float)
    else:
        offsets = np.asarray(tag_offsets_xyz, dtype=float)
    if offsets.shape != (2, 3) or not np.isfinite(offsets).all():
        raise ValueError("tag_offsets_xyz exige duas coordenadas XYZ finitas")
    return offsets


def _measurement_model(x_pred, anchors, l, z_c, tag_offsets_xyz=None):
    """Ranges e Jacobiano para tags XYZ arbitrárias, com canais fixos.

    Para cada tag j, offset local (a,b,c) vira no mundo:
    xt=x+cos(theta)*a-sin(theta)*b
    yt=y+sin(theta)*a+cos(theta)*b
    zt=c
    h_ij=norm([xt,yt,zt] - anchor_i)
    d(xt)/dtheta=-sin(theta)*a-cos(theta)*b
    d(yt)/dtheta= cos(theta)*a-sin(theta)*b

    Ordem h/H: [âncora0_canal0, âncora0_canal1, ...].
    """
    anchors=np.asarray(anchors,dtype=float)
    if anchors.ndim!=2 or anchors.shape[0]!=3 or not np.isfinite(anchors).all():
        raise ValueError("anchors deve ser matriz 3xN finita")
    p=np.asarray(x_pred,dtype=float).reshape(3)
    offsets=_resolved_tag_offsets(l,z_c,tag_offsets_xyz)
    ct,st=np.cos(p[2]),np.sin(p[2])
    h=np.empty(2*anchors.shape[1],dtype=float)
    H=np.empty((2*anchors.shape[1],3),dtype=float)
    for i in range(anchors.shape[1]):
        for j,(ox,oy,oz) in enumerate(offsets):
            tx=p[0]+ct*ox-st*oy
            ty=p[1]+st*ox+ct*oy
            dtx=-st*ox-ct*oy
            dty= ct*ox-st*oy
            delta=np.array([tx-anchors[0,i],ty-anchors[1,i],oz-anchors[2,i]])
            dist=float(np.linalg.norm(delta))
            if dist<1e-9:
                raise ValueError("Distância tag-âncora ~ 0")
            k=2*i+j
            h[k]=dist
            H[k,0]=delta[0]/dist
            H[k,1]=delta[1]/dist
            H[k,2]=(delta[0]*dtx+delta[1]*dty)/dist
    return h,H


def _generate_uwb_measurements(
    x_hist, anchors, l, z_c, sigma_uwb, tag_offsets_xyz=None
):
    """Gera 2N ranges por passo usando a MESMA geometria do filtro.

    Importa o ruído sintético do projeto somente quando usado, para
    permitir testes NumPy-only do BC-EKF com dados reais no CI.
    """
    import src.utils as utils
    num_anchors=anchors.shape[1]
    z_hist=np.empty((2*num_anchors,x_hist.shape[1]),dtype=float)
    for k in range(x_hist.shape[1]):
        expected,_=_measurement_model(
            x_hist[:,k],anchors,l,z_c,tag_offsets_xyz=tag_offsets_xyz
        )
        for j,dist in enumerate(expected):
            z_hist[j,k]=utils.apply_uwb_errors(dist,sigma_uwb)
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