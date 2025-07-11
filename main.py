# main.py
import config
from robot import Robot
from trajectory import Trajectory
from noise import add_gaussian_noise
from utils import save_data, plot_trajectory
import numpy as np

def main():
    # Definir waypoints
    waypoints = [
        (0.5, 0.5),
        (4.5, 0.5),
        (4.5, 4.5),
        (0.5, 4.5)
    ]

    traj = Trajectory(waypoints)
    robot = Robot(config)

    time = 0.0
    data_log = []
    path_log = []

    while time <= config.SIM_DURATION and not traj.is_finished():
        target = traj.get_target()
        dx = target[0] - robot.x
        dy = target[1] - robot.y
        distance = np.hypot(dx, dy)
        angle_to_target = np.arctan2(dy, dx)

        # Calcular erro angular
        angle_error = angle_to_target - robot.theta
        angle_error = (angle_error + np.pi) % (2*np.pi) - np.pi

        # Simples controle proporcional
        k_v = 0.5
        k_w = 2.0

        v_desired = min(k_v * distance, config.MAX_LINEAR_VELOCITY)
        w_desired = k_w * angle_error

        # Atualizar estado do robô
        robot.update(v_desired, w_desired, config.TIME_STEP)

        # Velocidades das rodas (com ruído)
        v_r, v_l = robot.get_wheel_velocities()
        v_r_noisy = add_gaussian_noise(v_r, config.NOISE_STD_V)
        v_l_noisy = add_gaussian_noise(v_l, config.NOISE_STD_V)

        # Log
        data_log.append([
            time,
            v_r_noisy,
            v_l_noisy,
            robot.v,
            robot.omega,
            robot.x,
            robot.y,
            robot.theta
        ])

        path_log.append((robot.x, robot.y))

        # Checar se waypoint foi alcançado
        traj.advance_if_reached(robot.x, robot.y)

        # Incrementar tempo
        time += config.TIME_STEP

    # Salvar CSV
    headers = ["Time(s)", "v_r_noisy(m/s)", "v_l_noisy(m/s)", "v(m/s)", "omega(rad/s)", "x(m)", "y(m)", "theta(rad)"]
    save_data("data/simulation_output.csv", data_log, headers, precision=config.CSV_PRECISION)

    # Plotar
    plot_trajectory((config.MAP_WIDTH, config.MAP_HEIGHT), waypoints, path_log)

if __name__ == "__main__":
    main()
