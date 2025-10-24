# simulator.py
import numpy as np
from src.robot import Robot
from src.bc_ekf import run_bc_ekf_step
import src.utils as utils
import src.config as config

class Simulator:
    def __init__(self, anchors, baseline, z_c, Q, R, dt=0.05, config=config, logger=None):
        """
        Simulador do robô + EKF.
        anchors: 3xN
        baseline: distância entre as tags (m)
        z_c: altura das tags (m)
        Q, R: covariâncias
        dt: passo de simulação (s)
        config: módulo de configuração
        logger: utils.RunLogger opcional
        """
        # Parâmetros principais
        self.dt = float(dt)
        self.anchors = anchors
        self.l = float(baseline) / 2.0
        self.z_c = float(z_c)
        self.Q = Q
        self.R = R

        # Robô simulado
        self.robot = Robot(config)

        # EKF inicial
        self.x_est = np.array([2.5, 0.0, 0.0], dtype=float)
        self.P = np.diag([0.1, 0.1, 0.1]).astype(float)

        # Logs
        self.history_true = []
        self.history_est  = []
        self.history_pred = []

        # Debug EKF (preenche no step)
        self.last_debug = None

        # Logger opcional
        self.logger = logger

        # Ruídos (centralizados)
        self.sigma_v   = getattr(config, "NOISE_STD_V", 0.02)
        self.sigma_w   = getattr(config, "NOISE_STD_W", 0.05)
        self.sigma_uwb = np.sqrt(0.0025)  # pode virar config também

    def step(self, v_cmd, w_cmd, noisy=True):
        """Executa um passo: integra robô, simula sensores, roda EKF e loga."""
        # 1) Atualiza o estado verdadeiro (modelo cinemático do robô já limita aceleração)
        self.robot.update(v_cmd, w_cmd, self.dt)
        x_true, y_true, theta_true = self.robot.x, self.robot.y, self.robot.theta

        # 2) Odometria ruidosa (medição)
        if noisy:
            v_meas = v_cmd + np.random.randn() * self.sigma_v
            w_meas = w_cmd + np.random.randn() * self.sigma_w
        else:
            v_meas, w_meas = v_cmd, w_cmd

        # 3) Medidas UWB e R conforme nº de âncoras
        num_anchors = 0 if self.anchors is None else self.anchors.shape[1]
        if num_anchors > 0:
            z_k = utils.generate_uwb_single_measurement(
                [x_true, y_true, theta_true],
                self.anchors, self.l, self.z_c,
                self.sigma_uwb if noisy else 0.0
            )
            # Variância = sigma^2 para cada distância (frente e trás)
            self.R = np.eye(2 * num_anchors) * (self.sigma_uwb ** 2)
        else:
            z_k = np.array([])        # sem medições
            self.R = np.zeros((0, 0)) # garante forma compatível

        # 4) EKF (predição sempre; correção só se houver z_k)
        x_next, P_next, dbg = run_bc_ekf_step(
            self.x_est, self.P,
            np.array([v_meas, w_meas], dtype=float),
            z_k, self.anchors, self.l, self.z_c, self.Q, self.R,
            dt=self.dt,
            debug=True
        )
        self.x_est, self.P = x_next, P_next
        self.last_debug = dbg

        # 5) Logs de trajetória
        self.history_true.append([x_true, y_true, theta_true])
        self.history_est.append(self.x_est.copy())

        # 5.1) Guarda a predição (se disponível no debug)
        if isinstance(dbg, dict) and ('x_pred' in dbg):
            self.history_pred.append(dbg['x_pred'].copy())
        else:
            self.history_pred.append(self.x_est.copy())

        # 6) Métricas instantâneas de erro
        pos_err = float(np.linalg.norm(np.array([x_true, y_true]) - self.x_est[:2]))
        dth = float(np.arctan2(np.sin(theta_true - self.x_est[2]), np.cos(theta_true - self.x_est[2])))
        head_err_deg = abs(np.degrees(dth))

        # 7) Logger opcional
        if self.logger is not None:
            pred = dbg['x_pred'] if (isinstance(dbg, dict) and ('x_pred' in dbg)) else None
            self.logger.log_step(
                true_state=[x_true, y_true, theta_true],
                pred_state=pred,
                est_state=self.x_est.copy(),
                v_cmd=float(v_cmd), w_cmd=float(w_cmd),
                v_meas=float(v_meas), w_meas=float(w_meas),
                pos_err=pos_err, heading_err_deg=head_err_deg
            )

        # Retorna um snapshot útil para quem chama
        return {
            "true":   np.array([x_true, y_true, theta_true], dtype=float),
            "est":    self.x_est.copy(),
            "pred":   (dbg['x_pred'].copy() if (isinstance(dbg, dict) and ('x_pred' in dbg)) else None),
            "P":      self.P.copy(),
            "pos_err": pos_err,
            "head_err_deg": head_err_deg,
            "innov":  (dbg.get("innov") if isinstance(dbg, dict) else None),
            "S":      (dbg.get("S") if isinstance(dbg, dict) else None),
        }

    def get_logs(self):
        """Retorna histórico de trajetória real e estimada."""
        return np.array(self.history_true, dtype=float), np.array(self.history_est, dtype=float)
