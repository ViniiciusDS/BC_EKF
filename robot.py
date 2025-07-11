# robot.py
import numpy as np

class Robot:
    def __init__(self, config):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.v = 0.0
        self.omega = 0.0

        self.config = config

    def update(self, v_target, omega_target, dt):
        # Atualizar velocidade linear com aceleração limitada
        dv = v_target - self.v
        max_dv = self.config.MAX_LINEAR_ACCEL * dt
        if abs(dv) > max_dv:
            dv = max_dv * np.sign(dv)
        self.v += dv

        # Atualizar velocidade angular com aceleração limitada
        domega = omega_target - self.omega
        max_domega = self.config.MAX_ANGULAR_ACCEL * dt
        if abs(domega) > max_domega:
            domega = max_domega * np.sign(domega)
        self.omega += domega

        # Atualizar pose
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.omega * dt

    def get_wheel_velocities(self):
        """
        Retorna velocidades das rodas esquerda e direita.
        """
        v_r = (2*self.v + self.omega*self.config.WHEEL_BASE) / (2*self.config.WHEEL_RADIUS)
        v_l = (2*self.v - self.omega*self.config.WHEEL_BASE) / (2*self.config.WHEEL_RADIUS)
        return v_r, v_l
