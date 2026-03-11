# main_realtime.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
from src.simulator import Simulator
from src.scenarios import anchors_tectrol
import src.config as config
from src.trajectory import Trajectory
from src.utils import RunLogger
import os

# ======================
# Configurações
# ======================
DT = 0.05
BASELINE = 0.65
Z_C = 0.5
Q = np.diag([1e-4, 1e-4, 1e-4])
R = np.eye(2 * anchors_tectrol.shape[1]) * 0.0025

# Seleção de rota
ROUTE_NAME = "square"  # "square" | "circle" | "figure_eight"
if ROUTE_NAME == "square":
    trajectory = Trajectory.square(size=10, start=(0,0))
elif ROUTE_NAME == "circle":
    trajectory = Trajectory.circle(radius=8, points=72, center=(5,5))
elif ROUTE_NAME == "figure_eight":
    trajectory = Trajectory.figure_eight(radius=4, points=72, center=(5,5))
else:
    raise ValueError("Tipo de rota inválido!")
waypoints = np.array(trajectory.waypoints)

# Inicializa simulador
sim = Simulator(
    anchors=anchors_tectrol,
    baseline=BASELINE,
    z_c=Z_C,
    Q=Q,
    R=R,
    dt=DT,
    config=config
)

# ======================
# Controle simples de waypoint
# ======================
def waypoint_controller(current_pos, waypoints, current_idx, v_max=0.25, w_max=0.5, threshold=0.3):
    """
    Controlador de waypoint simples (proporcional).
    """
    if current_idx >= len(waypoints):
        return 0.0, 0.0, current_idx  # Chegou ao fim

    target = waypoints[current_idx]
    dx = target[0] - current_pos[0]
    dy = target[1] - current_pos[1]
    dist = np.hypot(dx, dy)

    # Ângulo desejado
    target_theta = np.arctan2(dy, dx)
    angle_error = np.arctan2(np.sin(target_theta - current_pos[2]), np.cos(target_theta - current_pos[2]))

    # Velocidade linear e angular
    v = v_max * np.clip(dist, 0, 1)
    w = 1.5 * angle_error  # Ganho simples

    # Saturação
    v = np.clip(v, -v_max, v_max)
    w = np.clip(w, -w_max, w_max)

    # Avança para o próximo waypoint se estiver próximo
    if dist < threshold:
        current_idx += 1

    return v, w, current_idx

# ======================
# Setup do logger (condicional)
# ======================
logger = None
if getattr(config, "LOGGING_ENABLED", False):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    meta = {
        "DT": DT,
        "BASELINE": BASELINE,
        "Z_C": Z_C,
        "anchors_count": int(anchors_tectrol.shape[1]),
        "route": ROUTE_NAME,
    }
    logger = RunLogger(
        out_dir=config.LOG_DIR,
        meta=meta,
        flush_every_n=getattr(config, "LOG_FLUSH_EVERY_N", 200),
    )
    print(f"[LOG] Registrando nesta pasta: {logger.out_path}")

# ======================
# Gráficos
# ======================
plt.ion()
fig = plt.figure(figsize=(12, 6))

# (1) Trajetórias
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax1.set_title("Simulação BC-EKF em Tempo Real")
ax1.set_xlim(-1, 30); ax1.set_ylim(-1, 30)
ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
ax1.grid(True)
ax1.plot(anchors_tectrol[0], anchors_tectrol[1], 'r*', markersize=12, label='Âncoras')
ax1.plot(waypoints[:, 0], waypoints[:, 1], 'k--', label='Rota Planejada')
traj_real_line, = ax1.plot([], [], 'k-', label='Trajetória Real')
traj_pred_line, = ax1.plot([], [], 'b--', label='Predição (EKF)')
traj_est_line,  = ax1.plot([], [], 'r--', label='Estimativa Corrigida')
ax1.legend()

# (2) Erro de posição
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2.set_title("Erro de Posição"); ax2.set_xlim(0, 500); ax2.set_ylim(0, 2)
ax2.set_xlabel("Iterações"); ax2.set_ylabel("Erro (m)")
ax2.grid(True)
error_pos_line, = ax2.plot([], [], 'g-', label='Erro de Posição')
pos_text = ax2.text(0.05, 0.9, "", transform=ax2.transAxes, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
ax2.legend()

# (3) Erro de orientação
ax3 = plt.subplot2grid((2, 2), (1, 1))
ax3.set_title("Erro de Orientação"); ax3.set_xlim(0, 500); ax3.set_ylim(0, 15)
ax3.set_xlabel("Iterações"); ax3.set_ylabel("Erro (°)")
ax3.grid(True)
error_heading_line, = ax3.plot([], [], 'm-', label='Erro de Orientação')
heading_text = ax3.text(0.05, 0.9, "", transform=ax3.transAxes, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7))
ax3.legend()

# Slider de velocidade
ax_slider = plt.axes([0.25, 0.01, 0.5, 0.03])
speed_slider = Slider(ax_slider, 'Velocidade x', 0.2, 3.0, valinit=1.0)

plt.tight_layout()

# ======================
# Loop principal
# ======================
print("Rodando simulação... Pressione Ctrl+C para encerrar.")
pos_errors, heading_errors = [], []
current_idx = 0

try:
    while True:
        # controle para seguir a rota
        v_cmd, w_cmd, current_idx = waypoint_controller(sim.x_est, waypoints, current_idx)
        v_cmd *= speed_slider.val
        w_cmd *= speed_slider.val

        # passo de simulação (gera v_noisy / w_noisy lá dentro)
        sim.step(v_cmd, w_cmd)

        # logs de trajetória
        true_traj, est_traj = sim.get_logs()
        pred_traj = np.array(sim.history_pred)

        # atualiza plot de trajetórias
        traj_real_line.set_data(true_traj[:, 0], true_traj[:, 1])
        if len(pred_traj) > 0:
            traj_pred_line.set_data(pred_traj[:, 0], pred_traj[:, 1])
        traj_est_line.set_data(est_traj[:, 0], est_traj[:, 1])

        # calcula erros atuais
        if len(true_traj) > 0:
            pos_err = np.linalg.norm(true_traj[-1, 0:2] - est_traj[-1, 0:2])
            heading_err = abs((true_traj[-1, 2] - est_traj[-1, 2]) * 180 / np.pi)
            pos_errors.append(pos_err); heading_errors.append(heading_err)

            error_pos_line.set_data(range(len(pos_errors)), pos_errors)
            error_heading_line.set_data(range(len(heading_errors)), heading_errors)
            ax2.set_xlim(0, max(100, len(pos_errors)))
            ax3.set_xlim(0, max(100, len(heading_errors)))

            pos_text.set_text(f"Atual: {pos_err:.3f} m")
            heading_text.set_text(f"Atual: {heading_err:.2f}°")

            # ======== logging condicional ========
            if logger is not None:
                true_state = true_traj[-1]
                est_state  = est_traj[-1]
                pred_state = pred_traj[-1] if len(pred_traj) > 0 else None
                # v_meas e w_meas: use os medidos que o Simulator passou ao EKF.
                # Se quiser, exponha em Simulator (ex.: self.last_v_noisy / self.last_w_noisy)
                v_meas = getattr(sim, "last_v_noisy", v_cmd)
                w_meas = getattr(sim, "last_w_noisy", w_cmd)

                logger.log_step(
                    true_state=true_state,
                    pred_state=pred_state,
                    est_state=est_state,
                    v_cmd=v_cmd, w_cmd=w_cmd,
                    v_meas=v_meas, w_meas=w_meas,
                    pos_err=pos_err, heading_err_deg=heading_err
                )

        plt.pause(DT)

except KeyboardInterrupt:
    print("\nSimulação encerrada.")
finally:
    if logger is not None:
        logger.close()
        print(f"[LOG] Arquivos salvos em: {logger.out_path}")