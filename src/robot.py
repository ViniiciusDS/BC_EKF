# robot.py
import numpy as np
try:
    import src.utils as utils
except Exception:
    import utils
import numpy as np

class Robot:
    """
    Classe que representa o estado do robô e seu modelo cinemático.
    """
    def __init__(self, config):
        self.x = self.y = self.theta = 0.0
        self.v = self.omega = 0.0
        self.config = config

    def update(self, v_target, omega_target, dt):
        """Atualiza o estado do robô aplicando limites físicos."""
        # Linear
        dv = np.clip(
            v_target - self.v,
            -self.config.MAX_LINEAR_ACCEL * dt,
            self.config.MAX_LINEAR_ACCEL * dt
        )
        self.v = np.clip(self.v + dv, -self.config.MAX_LINEAR_VELOCITY, self.config.MAX_LINEAR_VELOCITY)

        # Angular
        domega = np.clip(
            omega_target - self.omega,
            -self.config.MAX_ANGULAR_ACCEL * dt,
            self.config.MAX_ANGULAR_ACCEL * dt
        )
        self.omega = np.clip(self.omega + domega, -self.config.MAX_ANGULAR_VELOCITY, self.config.MAX_ANGULAR_VELOCITY)

        # Pose
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta = np.arctan2(np.sin(self.theta + self.omega * dt), np.cos(self.theta + self.omega * dt))

    def get_wheel_velocities(self):
        """
        Retorna velocidades das rodas esquerda e direita.
        """
        v_r = (2*self.v + self.omega*self.config.WHEEL_BASE) / (2*self.config.WHEEL_RADIUS)
        v_l = (2*self.v - self.omega*self.config.WHEEL_BASE) / (2*self.config.WHEEL_RADIUS)
        return v_r, v_l


def simulate_trajectory_motion(
    T,
    t_final,
    trajectory,
    v_max,
    w_max,
    sigma_v,
    sigma_w,
    sigma_uwb,
    anchors,
    baseline,
    z_c=0.5,
    debug=False
):
    """
    Simula execução de uma trajetória real com ruído.

    Retorna:
        - t: vetor de tempo
        - x_hist_true: trajetoria real (3 x N)
        - odometry_noisy: velocidades ruidosas [v,w] (2 x N)
        - z_hist: medições UWB (2*num_anchors x N)
    """
    t = np.arange(0, t_final, T)
    num_steps = len(t)
    num_anchors = 0 if anchors is None else anchors.shape[1]

    x_hist_true = np.zeros((3, num_steps))
    odometry_noisy = np.zeros((2, num_steps))
    z_hist = np.zeros((2 * num_anchors, num_steps))

    x_true = np.array([2.5, 0, 0])
    x_hist_true[:, 0] = x_true

    for k in range(1, num_steps):
        target = trajectory.get_target()
        if target is None:
            break
        x_t, y_t = target
        dx, dy = x_t - x_true[0], y_t - x_true[1]
        dist = np.hypot(dx, dy)
        angle_to_target = np.arctan2(dy, dx)
        angle_diff = np.arctan2(np.sin(angle_to_target - x_true[2]), np.cos(angle_to_target - x_true[2]))

        v_cmd = np.clip(dist, 0, v_max)
        w_cmd = np.clip(angle_diff, -w_max, w_max)

        if dist < 0.2:
            trajectory.advance_if_reached(x_true[0], x_true[1])

        v_noisy = v_cmd + sigma_v * np.random.randn()
        w_noisy = w_cmd + sigma_w * np.random.randn()
        odometry_noisy[:, k] = [v_noisy, w_noisy]

        x_true[0] += v_cmd * T * np.cos(x_true[2] + w_cmd * T / 2)
        x_true[1] += v_cmd * T * np.sin(x_true[2] + w_cmd * T / 2)
        x_true[2] = np.arctan2(np.sin(x_true[2] + w_cmd * T), np.cos(x_true[2] + w_cmd * T))
        x_hist_true[:, k] = x_true

        if num_anchors > 0:
            pf = [x_true[0] + baseline * np.cos(x_true[2]),
                  x_true[1] + baseline * np.sin(x_true[2]), z_c]
            pr = [x_true[0] - baseline * np.cos(x_true[2]),
                  x_true[1] - baseline * np.sin(x_true[2]), z_c]
            for i in range(num_anchors):
                z_hist[2*i, k]     = utils.apply_uwb_errors(np.linalg.norm(pf - anchors[:, i]), sigma_uwb)
                z_hist[2*i + 1, k] = utils.apply_uwb_errors(np.linalg.norm(pr - anchors[:, i]), sigma_uwb)

        if debug:
            print(f"[step {k}] pos=({x_true[0]:.2f},{x_true[1]:.2f}) θ={x_true[2]:.2f} v={v_cmd:.2f} w={w_cmd:.2f}")

    return t, x_hist_true, odometry_noisy, z_hist
