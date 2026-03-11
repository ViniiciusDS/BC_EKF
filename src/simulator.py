# simulator.py
import numpy as np
from src.robot import Robot
from src.bc_ekf import run_bc_ekf_step
import src.utils as utils
import src.config as config
from typing import Optional, Any, Dict
from src.uwb.uwb_sim import UwbSimPipeline

class Simulator:
    def __init__(self, 
                 anchors: Optional[np.ndarray], 
                 baseline: float, 
                 z_c: float, 
                 Q: np.ndarray, 
                 R: np.ndarray, 
                 dt: float = 0.05, 
                 config=config, 
                 logger: Optional[Any] = None, 
                 env: Optional[Any] = None, 
                 channel_params: Optional[Dict[str, Any]] = None,
                 uwb_pipeline: Optional[UwbSimPipeline] = None
                 ) -> None:
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
        self.sigma_uwb = getattr(config, "UWB_NOISE_STD", 0.05)
        self.env = env
        self.channel_params = channel_params or {'disable_dropout': True}
        self.last_meas_meta = None  # <- para o HUD
        self.last_meas = (0.0, 0.0)  # <- para o HUD
        self.x_true = np.array([2.5,0,0], dtype=float)

        # UWB Pipeline (opcional, para simulações mais realistas)
        # Se fornecido, o pipeline pode ser usado para gerar medições UWB mais realistas, considerando efeitos de canal, dropout, etc.
        # Se none, o simulador gera medições UWB simples com ruído gaussiano. (legado)
        self.uwb_pipeline = uwb_pipeline

        self.last_zk = None  # para debug e HUD


    def step(
        self,
        v_cmd: float,
        w_cmd: float,
        *,
        perfect_motion: bool = False,
        perfect_odometry: bool = False,
        perfect_uwb: bool = False,
        perfect_filter_model: bool = False,
        true_override: Optional[np.ndarray] = None,
        use_odometry: bool = True 
    ) -> None:
        """Executa um passo: integra robô, simula sensores, roda EKF e loga."""

        # ----------------------------
        # 1) Movimento verdadeiro
        # ----------------------------
        if perfect_motion and (true_override is not None):
            x_true, y_true, theta_true = float(true_override[0]), float(true_override[1]), float(true_override[2])
            # força o robô interno a bater com o ground truth
            self.robot.x = x_true
            self.robot.y = y_true
            self.robot.theta = theta_true
        else:
            self.robot.update(v_cmd, w_cmd, self.dt)
            x_true, y_true, theta_true = self.robot.x, self.robot.y, self.robot.theta

        # Estado verdadeiro (usado para log e EKF)
        x_true, y_true, theta_true = float(self.robot.x), float(self.robot.y), float(self.robot.theta)

        # Pega velocidades "aplicadas" no robo
        v_applied = getattr(self.robot, "v", v_cmd)
        w_applied = getattr(self.robot, "w", w_cmd)
        v_applied = float(v_applied)
        w_applied = float(w_applied)

        self.x_true = np.array([x_true, y_true, theta_true], dtype=float)

        # ----------------------------
        # 2) Odometria medida
        # ----------------------------
        if not use_odometry:
            # modo "sem odometria": filtro não recebe v,w medidos
            v_meas, w_meas = 0.0, 0.0
        elif perfect_odometry:
            v_meas, w_meas = v_applied, w_applied
        else:
            v_meas = v_applied + np.random.randn() * self.sigma_v
            w_meas = w_applied + np.random.randn() * self.sigma_w

        self.last_meas = (float(v_meas), float(w_meas))

        # ----------------------------
        # 3) Medidas UWB
        # ----------------------------
        num_anchors = 0 if self.anchors is None else int(self.anchors.shape[1])

        # valores estáveis
        SIGMA_PERFECT = 0.005   # 5 mm (bom para “perfeito”, sem ficar numericamente agressivo)
        SIGMA_NOMINAL = float(getattr(config, "UWB_NOISE_STD", 0.05))

        if num_anchors > 0:

            if perfect_uwb:
                # >>> UWB realmente perfeito: determinístico e compatível com o modelo do EKF <<<
                z_k = self._ideal_uwb_measurement(x_true, y_true, theta_true)
                meta = {"perfect_uwb": True}
                sigma_world = SIGMA_PERFECT

            else:
                # UWB normal (pipeline ou legado)
                if self.uwb_pipeline is not None:
                    z_k, meta = self.uwb_pipeline.measure(
                        np.array([x_true, y_true, theta_true], dtype=float),
                        self.anchors, self.l, self.z_c,
                        return_meta=True
                    )
                    sigma_los = float(getattr(self.uwb_pipeline.ranging_model.cfg, "sigma_los", SIGMA_NOMINAL))
                    sigma_world = sigma_los
                else:
                    z_k, meta = utils.generate_uwb_single_measurement(
                        [x_true, y_true, theta_true],
                        self.anchors, self.l, self.z_c,
                        SIGMA_NOMINAL,
                        env=self.env,
                        channel_params=self.channel_params,
                        return_meta=True
                    )
                    sigma_world = SIGMA_NOMINAL

            z_k = np.asarray(z_k, dtype=float).reshape(-1)
            expected = 2 * num_anchors
            if z_k.size != expected:
                print(f"[WARN] z_k size={z_k.size}, expected={expected}.")
                # melhor: invalidar medida do que “duplicar” e bagunçar o EKF
                z_k = np.array([], dtype=float)

            # mundo
            R_world = np.eye(expected) * (sigma_world ** 2)

            # filtro (casado ou não)
            if perfect_filter_model:
                R_filter = R_world.copy()
            else:
                R_filter = np.eye(expected) * (SIGMA_NOMINAL ** 2)

            self.R = R_filter
            self.last_meas_meta = meta

        else:
            z_k = np.array([], dtype=float)
            self.last_meas_meta = []
            self.R = np.zeros((0, 0), dtype=float)


        # ----------------------------
        # 4) EKF
        # ----------------------------
        EPS_Q = 1e-6  # para evitar singularidade em Q do filtro quando modelo perfeito

        no_odo = (not use_odometry)

        u_for_filter = np.array([v_meas, w_meas], dtype=float)
        Q_for_filter = self.Q

        # --- entrada do filtro ---
        if no_odo:
            u_for_filter = np.array([0.0, 0.0], dtype=float)

            # quando não tem odometria, o modelo fica muito mais incerto:
            # aumenta Q para o filtro conseguir "andar" via correção do UWB
            Q_for_filter = self.Q * float(getattr(config, "Q_NO_ODOMETRY_SCALE", 25.0))

        elif perfect_filter_model:
            u_for_filter = np.array([v_applied, w_applied], dtype=float)
            Q_for_filter = np.eye(self.Q.shape[0]) * EPS_Q
        else:
            u_for_filter = np.array([v_meas, w_meas], dtype=float)
            Q_for_filter = self.Q

        # Garante formato de z_k
        if num_anchors > 0:
            z_k = np.asarray(z_k, dtype=float).reshape(-1)
            expected = 2 * num_anchors
            if z_k.size != expected:
                print(f"[WARN] z_k size={z_k.size}, expected={expected}.")
                if z_k.size == num_anchors:
                    z_k = np.repeat(z_k, 2)  # patch compatibilidade
                else:
                    z_k = np.array([], dtype=float)
                    self.R = np.zeros((0, 0), dtype=float)
        else:
            z_k = np.array([], dtype=float)

        x_next, P_next, dbg = run_bc_ekf_step(
            self.x_est, self.P,
            u_for_filter,
            z_k, self.anchors, self.l, self.z_c, Q_for_filter, self.R,
            dt=self.dt,
            debug=True
        )

        self.x_est, self.P = x_next, P_next
        self.last_debug = dbg

        # ----------------------------
        # 5) Logs
        # ----------------------------
        self.history_true.append([x_true, y_true, theta_true])
        self.history_est.append(self.x_est.copy())
        

        if isinstance(dbg, dict) and ('x_pred' in dbg) and (dbg['x_pred'] is not None):
            xpred = np.asarray(dbg['x_pred'], dtype=float).reshape(3,)
            self.history_pred.append(xpred.copy())
        else:
            self.history_pred.append(np.asarray(self.x_est, dtype=float).reshape(3,).copy())  # fallback para evitar listas de tamanhos diferentes

        # ----------------------------
        # 6) Erros
        # ----------------------------
        pos_err = float(np.linalg.norm(np.array([x_true, y_true]) - self.x_est[:2]))
        dth = float(np.arctan2(np.sin(theta_true - self.x_est[2]), np.cos(theta_true - self.x_est[2])))
        head_err_deg = abs(float(np.degrees(dth)))

        # Logger opcional
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

        if self.uwb_pipeline and self.anchors is not None:
            z_k = self.uwb_pipeline.measure(
                np.array([x_true, y_true, theta_true], dtype=float),
                self.anchors, self.l, self.z_c,
                return_meta=False
            )
            self.last_zk = z_k  # para debug e HUD

        return {
            "true": np.array([x_true, y_true, theta_true], dtype=float),
            "est": self.x_est.copy(),
            "pred": (dbg['x_pred'].copy() if (isinstance(dbg, dict) and ('x_pred' in dbg)) else None),
            "P": self.P.copy(),
            "pos_err": pos_err,
            "head_err_deg": head_err_deg,
            "innov": (dbg.get("innov") if isinstance(dbg, dict) else None),
            "S": (dbg.get("S") if isinstance(dbg, dict) else None),
        }

    def get_logs(self):
        """Retorna histórico de trajetória real e estimada."""
        return np.array(self.history_true, dtype=float), np.array(self.history_est, dtype=float)

    def _uwb_measure_perfect(self, x_true, y_true, theta_true):
        # posições das duas tags (frente e trás)
        ct, st = np.cos(theta_true), np.sin(theta_true)
        xf, yf = x_true + self.l * ct, y_true + self.l * st
        xb, yb = x_true - self.l * ct, y_true - self.l * st

        # âncoras: 3xN (x,y,z)
        ax = self.anchors[0, :]
        ay = self.anchors[1, :]
        az = self.anchors[2, :]

        # distância 3D considerando altura z_c das tags
        zt = self.z_c
        df = np.sqrt((ax - xf)**2 + (ay - yf)**2 + (az - zt)**2)
        db = np.sqrt((ax - xb)**2 + (ay - yb)**2 + (az - zt)**2)

        # vetor: [d_f1..d_fN, d_b1..d_bN]
        z_k = np.concatenate([df, db]).astype(float)
        meta = {"perfect": True}
        return z_k, meta

    def _ideal_uwb_measurement(self, x_true: float, y_true: float, theta_true: float) -> np.ndarray:
        """
        Mede UWB ideal (sem ruído/canal), no formato esperado pelo EKF:
        [Df0, Dr0, Df1, Dr1, ..., DfN-1, DrN-1]
        """
        anchors = self.anchors
        if anchors is None or anchors.size == 0:
            return np.array([], dtype=float)

        num_anchors = anchors.shape[1]

        ax = anchors[0, :]
        ay = anchors[1, :]
        az = anchors[2, :] if anchors.shape[0] >= 3 else np.zeros(num_anchors)

        c = float(np.cos(theta_true))
        s = float(np.sin(theta_true))

        # posição das tags
        pf = np.array([x_true + self.l * c, y_true + self.l * s, self.z_c], dtype=float)
        pr = np.array([x_true - self.l * c, y_true - self.l * s, self.z_c], dtype=float)

        # distâncias para cada âncora
        Df = np.sqrt((ax - pf[0])**2 + (ay - pf[1])**2 + (az - pf[2])**2)
        Dr = np.sqrt((ax - pr[0])**2 + (ay - pr[1])**2 + (az - pr[2])**2)

        # >>> INTERCALAR do jeito que o EKF espera <<<
        z = np.empty(2 * num_anchors, dtype=float)
        z[0::2] = Df
        z[1::2] = Dr
        return z

    def compute_gdop(self, x: float, y: float) -> float:
        """
        GDOP 2D baseado nas âncoras e posição (x,y).
        Implementa a ideia do artigo: GDOP = tr((B^T B)^-1).  :contentReference[oaicite:2]{index=2}

        Observação: em literatura GNSS às vezes aparece sqrt(trace(...)).
        Aqui seguimos o artigo (sem sqrt).
        """
        if self.anchors is None or self.anchors.size == 0 or self.anchors.shape[1] < 3:
            return float("nan")

        xa = self.anchors[0, :]
        ya = self.anchors[1, :]

        dx = xa - float(x)
        dy = ya - float(y)
        r = np.sqrt(dx * dx + dy * dy)

        # evita divisão por zero (robô em cima de âncora)
        eps = 1e-9
        r = np.maximum(r, eps)

        # B: n x 2 (direção unitária)
        B = np.stack([dx / r, dy / r], axis=1)  # shape (n,2)

        BtB = B.T @ B
        try:
            inv = np.linalg.inv(BtB)
        except np.linalg.LinAlgError:
            return float("inf")

        gdop = float(np.trace(inv))
        return gdop