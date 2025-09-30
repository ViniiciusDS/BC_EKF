# simulator.py
# motor do "mundo" simulado + EKF
import numpy as np
from src.robot import Robot
from src.bc_ekf import run_bc_ekf_step
import src.utils as utils

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
        # 1) Atualiza o estado verdadeiro
        self.robot.update(v, w, self.dt)
        x_true, y_true, theta_true = self.robot.x, self.robot.y, self.robot.theta

        # 2) Odometria ruidosa
        v_noisy = v + (0.02 * np.random.randn() if noisy else 0.0)
        w_noisy = w + (0.05 * np.random.randn() if noisy else 0.0)

        # 3) Medição UWB e matriz R compatíveis com nº de âncoras
        num_anchors = 0 if self.anchors is None else self.anchors.shape[1]
        if num_anchors > 0:
            z_k = utils.generate_uwb_single_measurement(
                [x_true, y_true, theta_true],
                self.anchors, self.l, self.z_c,
                0.05 if noisy else 0.0
            )
            self.R = np.eye(2 * num_anchors) * (0.05**2)  # sigma_uwb^2
        else:
            z_k = np.array([])
            self.R = np.zeros((0, 0))

        # 4) EKF (predição sempre; correção só se houver z_k)
        x_next, P_next, dbg = run_bc_ekf_step(
            self.x_est, self.P, np.array([v_noisy, w_noisy]),
            z_k, self.anchors, self.l, self.z_c, self.Q, self.R,
            debug=True
        )
        self.x_est, self.P = x_next, P_next
        self.last_debug = dbg  # guarda diagnósticos para UI

        # 5) Logs
        self.history_true.append([x_true, y_true, theta_true])
        self.history_est.append(self.x_est.copy())

        # registra a PREDIÇÃO (antes da correção), se disponível no debug
        if dbg is not None and isinstance(dbg, dict) and ('x_pred' in dbg):
            self.history_pred.append(dbg['x_pred'].copy())
        else:
            # fallback: se não houver, registra o estimado (não ideal, mas evita falhas)
            self.history_pred.append(self.x_est.copy())


    def get_logs(self):
        """Retorna logs de trajetória real e estimada."""
        return np.array(self.history_true), np.array(self.history_est)
