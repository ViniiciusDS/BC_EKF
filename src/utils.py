# utils.py
# ruido, I/O, helpers
import os, csv, time, json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
try:
    import src.config as config  # projeto como pacote (layout src/)
except Exception:
    import config  # fallback quando rodar como scripts soltos
from .uwb_channel import uwb_range_measure
from .environment import Environment
import multiprocessing as mp
from multiprocessing import Process, Queue
import queue
import time

_plot_backend_started = False

def save_data(filename, data, headers, precision=None):
    """Salva dados em CSV com separador ';' e precisão configurável."""
    if precision is None:
        precision = getattr(config, "CSV_PRECISION", 5)
    df = pd.DataFrame(data, columns=headers)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(lambda x: round(x, precision))
    df.to_csv(filename, index=False, sep=";")

def plot_trajectory(map_size, waypoints, path):
    """Plota mapa com waypoints e trajetória (seguro para listas vazias)."""
    plt.figure(figsize=(6,6))
    plt.xlim(0, map_size[0]); plt.ylim(0, map_size[1])

    if waypoints and len(waypoints) > 0:
        wp_x, wp_y = zip(*waypoints)
        plt.plot(wp_x, wp_y, "ro--", label="Waypoints")

    if path and len(path) > 0:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, "b-", label="Trajetória")

    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title("Simulação de Trajetória"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

def simulate_run(T, t_final, anchors, v_true, w_true, l, z_c, sigma_v, sigma_w, sigma_uwb):
    """Simula trajetória e gera medições ruidosas (robusto a âncoras vazias)."""
    t = np.arange(0, t_final + T, T)
    num_anchors = 0 if anchors is None else (anchors.shape[1] if anchors.size else 0)

    # trajetória real
    x_hist_true = np.zeros((3, len(t)))
    x_hist_true[:, 0] = [2.5, 0, 0]
    for k in range(1, len(t)):
        theta = x_hist_true[2, k-1]
        x_hist_true[0, k] = x_hist_true[0, k-1] + v_true*T*np.cos(theta + w_true*T/2)
        x_hist_true[1, k] = x_hist_true[1, k-1] + v_true*T*np.sin(theta + w_true*T/2)
        x_hist_true[2, k] = np.arctan2(np.sin(x_hist_true[2, k-1] + w_true*T),
                                       np.cos(x_hist_true[2, k-1] + w_true*T))

    v_noisy = v_true + sigma_v*np.random.randn(len(t))
    w_noisy = w_true + sigma_w*np.random.randn(len(t))

    if num_anchors == 0:
        z_hist = np.empty((0, len(t)))
        return t, x_hist_true, v_noisy, w_noisy, z_hist

    # medições UWB com o mesmo modelo de erros da função “apply_uwb_errors”
    z_hist = np.zeros((2*num_anchors, len(t)))
    for k in range(len(t)):
        theta = x_hist_true[2, k]; xt, yt = x_hist_true[0, k], x_hist_true[1, k]
        pf = np.array([xt + l*np.cos(theta), yt + l*np.sin(theta), z_c])
        pr = np.array([xt - l*np.cos(theta), yt - l*np.sin(theta), z_c])
        for i in range(num_anchors):
            dist_f = apply_uwb_errors(np.linalg.norm(pf - anchors[:, i]), sigma_uwb)
            dist_r = apply_uwb_errors(np.linalg.norm(pr - anchors[:, i]), sigma_uwb)
            z_hist[2*i, k] = dist_f
            z_hist[2*i + 1, k] = dist_r

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
    Medições UWB históricas usando o mesmo modelo de erro (bias + desalinh.).
        Args:
        x_hist (ndarray): Matriz 3 x N com [x, y, theta] ao longo do tempo.
        anchors (ndarray): Matriz 3 x num_anchors das posições das âncoras.
        l (float): Metade do baseline.
        z_c (float): Altura das tags.
        sigma_uwb (float): Desvio padrão do ruído UWB.

    Returns:
        z_hist (ndarray): Medições simuladas (2*num_anchors x N).
    """
    num_anchors = 0 if anchors is None else anchors.shape[1]
    if num_anchors == 0:
        return np.empty((0, x_hist.shape[1]))

    z_hist = np.zeros((2 * num_anchors, x_hist.shape[1]))
    for k in range(x_hist.shape[1]):
        theta = x_hist[2, k]; xt, yt = x_hist[0, k], x_hist[1, k]
        pf = np.array([xt + l*np.cos(theta), yt + l*np.sin(theta), z_c])
        pr = np.array([xt - l*np.cos(theta), yt - l*np.sin(theta), z_c])
        for i in range(num_anchors):
            z_hist[2*i, k]     = apply_uwb_errors(np.linalg.norm(pf - anchors[:, i]), sigma_uwb)
            z_hist[2*i + 1, k] = apply_uwb_errors(np.linalg.norm(pr - anchors[:, i]), sigma_uwb)
    return z_hist

def generate_uwb_single_measurement(
    x_state,
    anchors,
    l,
    z_c,
    sigma_uwb,
    env: Environment | None = None,
    channel_params: dict | None = None,
    return_meta: bool = False,
    rng: np.random.Generator | None = None
):
    """
    Medição UWB instantânea (retorna vetor vazio se não houver âncoras).
    Args:
        x_state: lista ou array [x, y, theta].
        anchors: matriz 3xN com posições das âncoras.
        l: metade do baseline.
        z_c: altura das tags.
        sigma_uwb: desvio padrão do ruído.
    Returns:
      - se return_meta=False: z_k (np.ndarray shape (2*N,))
      - se return_meta=True : (z_k, meta_list)  [meta_list: len==2*N]
    """
    if anchors is None or anchors.shape[1] == 0:
        return (np.empty((0,)), []) if return_meta else np.empty((0,))
    
    rng = rng or np.random.default_rng()

    xk, yk, th = x_state
    pf = np.array([xk + l*np.cos(th), yk + l*np.sin(th), z_c])
    pr = np.array([xk - l*np.cos(th), yk - l*np.sin(th), z_c])
    num_anchors = anchors.shape[1]

    z_k = np.zeros(2 * num_anchors)
    meta_list = []

    for i in range(num_anchors):
        a = anchors[:, i]

        if env is None:
            # --- comportamento antigo ---
            dist_f = np.linalg.norm(pf - a) + sigma_uwb * np.random.randn()
            dist_r = np.linalg.norm(pr - a) + sigma_uwb * np.random.randn()
            z_k[2*i]     = dist_f
            z_k[2*i + 1] = dist_r
            if return_meta:
                meta_list += [{'mode':'LOS','used':'direct'}, {'mode':'LOS','used':'direct'}]
        else:
            # --- canal com ambiente (camada 1) ---
            zf, mf = uwb_range_measure(pf[:2], a[:2], sigma_uwb, env, rng, channel_params)
            zr, mr = uwb_range_measure(pr[:2], a[:2], sigma_uwb, env, rng, channel_params)

            # segurança: degrade para LOS se vier NaN (dropout desabilitado por padrão)
            if not np.isfinite(zf):
                zf = np.linalg.norm(pf - a) + sigma_uwb * rng.normal()
                mf = {'mode':'fallback','used':'degraded'}
            if not np.isfinite(zr):
                zr = np.linalg.norm(pr - a) + sigma_uwb * rng.normal()
                mr = {'mode':'fallback','used':'degraded'}

            z_k[2*i]     = zf
            z_k[2*i + 1] = zr
            if return_meta:
                meta_list += [mf, mr]

    return (z_k, meta_list) if return_meta else z_k

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

def _segments_intersect(p1, p2, q1, q2):
    """Teste robusto de interseção de segmentos 2D."""
    def orient(a,b,c):
        return np.cross(b-a, c-a)
    p1 = np.array(p1[:2], float); p2 = np.array(p2[:2], float)
    q1 = np.array(q1[:2], float); q2 = np.array(q2[:2], float)

    o1 = orient(p1, p2, q1); o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1); o4 = orient(q1, q2, p2)

    if (o1 == 0 and np.allclose(q1, p1)) or (o2 == 0 and np.allclose(q2, p1)):
        return True
    return (o1*o2 < 0) and (o3*o4 < 0)


def _ray_blocked_by_walls(p_src3, p_dst3, walls):
    """Retorna True se o segmento src→dst cruza alguma parede."""
    if not walls:
        return False
    a = (p_src3[0], p_src3[1]); b = (p_dst3[0], p_dst3[1])
    for (w1, w2) in walls:
        if _segments_intersect(np.array(a), np.array(b), np.array(w1), np.array(w2)):
            return True
    return False

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

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # não suprime exceções


#   Noise functions
rng = np.random.default_rng()

def add_gaussian_noise(value, std_dev):
    """
    Retorna 'value' com ruído gaussiano de desvio padrão 'std_dev'.
    """
    return value + rng.normal(0, std_dev)

def set_random_seed(seed: int):
    """Define semente global para reproduzibilidade."""
    np.random.seed(seed)



#   Gráficos em tempo real com multiprocessing

def _plotting_process(q: Queue):
    """
    Processo separado que mantém a janela do Matplotlib e atualiza gráficos
    com dados recebidos pela fila q. Envie None para encerrar.
    """
    import matplotlib
    # backend com janela
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.canvas.manager.set_window_title("Erros BC-EKF — tempo real")

    line1, = ax1.plot([], [], label="Erro Pos (m)")
    line2, = ax2.plot([], [], label="Erro Heading (°)")

    ax1.set_ylabel("Erro [m]")
    ax1.grid(True); ax1.legend(loc="upper right")
    ax2.set_xlabel("Tempo [s]")
    ax2.set_ylabel("Erro [°]")
    ax2.grid(True); ax2.legend(loc="upper right")

    # garante que NÃO fica “sempre no topo”
    try:
        win = fig.canvas.manager.window
        win.attributes("-topmost", False)
    except Exception:
        pass

    # mostra a janela UMA vez, sem travar processo
    plt.show(block=False)

    running = True
    while running and plt.fignum_exists(fig.number):
        try:
            msg = q.get(timeout=0.1)
        except queue.Empty:
            msg = None

        # None = pedido para encerrar
        if msg is None:
            break

        t_hist, pos_hist, head_hist = msg
        if not t_hist:
            continue

        # atualiza dados
        line1.set_data(t_hist, pos_hist)
        line2.set_data(t_hist, head_hist)

        ax1.set_xlim(0, max(t_hist))
        ax2.set_xlim(0, max(t_hist))
        ax1.set_ylim(0, max(1e-3, max(pos_hist) * 1.1))
        ax2.set_ylim(0, max(1e-3, max(head_hist) * 1.1))

        # redesenha sem POPUP
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        time.sleep(0.05)  # ~20 Hz de atualização máx.

    plt.close("all")


def start_plot_process(state: dict):
    """
    Garante que existe um processo de gráficos rodando.
    Uso:
      state = {"plot_proc": None, "plot_q": None}
      start_plot_process(state)
    """
    if state.get("plot_proc") is not None and state["plot_proc"].is_alive():
        return  # já está rodando

    q = mp.Queue()
    p = mp.Process(target=_plotting_process, args=(q,), daemon=True)
    p.start()
    state["plot_proc"] = p
    state["plot_q"] = q


def stop_plot_process(state: dict):
    """Encerra o processo de gráficos (se existir)."""
    proc = state.get("plot_proc")
    q = state.get("plot_q")
    try:
        if q is not None:
            # envia sentinela para o worker encerrar
            q.put_nowait(None)
    except Exception:
        pass
    if proc is not None and proc.is_alive():
        proc.join(timeout=1.0)
    state["plot_proc"] = None
    state["plot_q"] = None


def push_plot_data(state: dict, t_vec, pos_err_vec, head_err_vec):
    """
    Envia (cópias) dos dados atuais para o processo de gráficos, se ativo.
    """
    q = state.get("plot_q")
    if q is None:
        return
    try:
        # manda cópias simples (listas) para não ter problema com numpy
        q.put_nowait((list(t_vec), list(pos_err_vec), list(head_err_vec)))
    except Exception:
        # se a fila estiver cheia ou processo morto
        pass

# Paredes/obstáculos
def point_segment_distance(p, p0, p1):
    """
    Distância mínima entre um ponto p e o segmento [p0, p1] (todos np.array de shape (2,)).
    """
    p  = np.asarray(p, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    v = p1 - p0
    w = p  - p0
    denom = np.dot(v, v)
    if denom <= 1e-12:
        return np.linalg.norm(w)  # segmento degenerado
    t = np.dot(w, v) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = p0 + t * v
    return np.linalg.norm(p - proj)