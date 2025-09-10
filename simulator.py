# simulator.py
import numpy as np
from robot import Robot
from bc_ekf import run_bc_ekf_step
import utils

class Simulator:
    def __init__(self, anchors, baseline, z_c, Q, R, dt=0.05, config=None):
        """
        Simulador do robô + EKF.
        Args:
            anchors: matriz 3xN com posições das âncoras.
            baseline: distância entre as duas tags (m).
            z_c: altura das tags (m).
            Q: matriz de covariância do processo.
            R: matriz de covariância da medição.
            dt: passo de tempo da simulação.
            config: módulo de configuração com parâmetros do robô.
        """
        # Parâmetros principais
        self.dt = dt
        self.anchors = anchors
        self.l = baseline / 2
        self.z_c = z_c
        self.Q = Q
        self.R = R

        # Robô simulado (agora usa módulo config diretamente)
        self.robot = Robot(config)

        # EKF inicial
        self.x_est = np.array([2.5, 0, 0])
        self.P = np.diag([0.1, 0.1, 0.1])

        # Logs
        self.history_true = []
        self.history_est = []
        self.history_pred = []  # no __init__

    def step(self, v, w, noisy=True):
        # Atualiza estado real
        self.robot.update(v, w, self.dt)
        x_true, y_true, theta_true = self.robot.x, self.robot.y, self.robot.theta

        # Gera leituras ruidosas
        v_noisy = v + (0.02 * np.random.randn() if noisy else 0)
        w_noisy = w + (0.05 * np.random.randn() if noisy else 0)

        # Medições UWB
        z_k = utils.generate_uwb_single_measurement(
            [x_true, y_true, theta_true],
            self.anchors,
            self.l,
            self.z_c,
            0.05 if noisy else 0
        )

        # ===== EKF: Predição =====
        # Predição isolada (sem correção ainda)
        x_pred, P_pred = run_bc_ekf_step(
            self.x_est,
            self.P,
            np.array([v_noisy, w_noisy]),
            None,  # <-- sem medições = só predição
            self.anchors,
            self.l,
            self.z_c,
            self.Q,
            self.R
        )
        self.history_pred.append(x_pred.copy())  # salva para análise

        # ===== EKF: Correção =====
        self.x_est, self.P = run_bc_ekf_step(
            x_pred,
            P_pred,
            np.array([v_noisy, w_noisy]),
            z_k,
            self.anchors,
            self.l,
            self.z_c,
            self.Q,
            self.R
        )

        # Logs
        self.history_true.append([x_true, y_true, theta_true])
        self.history_est.append(self.x_est.copy())


    def get_logs(self):
        """Retorna logs de trajetória real e estimada."""
        return np.array(self.history_true), np.array(self.history_est)
