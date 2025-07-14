from trajectory import Trajectory
from robot import simulate_trajectory_motion
from bc_ekf import run_bc_ekf_from_data
import visualization as viz
import numpy as np

# Trajetória retangular
waypoints = [
    (2,2),
    (17,2),
    (17,17),
    (2,17),
    (2,2)
]
traj = Trajectory(waypoints)

# Parâmetros
T = 0.05
t_final = 200
v_max = 0.3
w_max = np.deg2rad(30)
sigma_v = 0.02
sigma_w = 0.05
sigma_uwb = np.sqrt(0.05)
baseline = 0.65/2
z_c = 0.5

anchors = np.array([
    [6.351, -4.966, 1.0],
    [5.947, -4.966, 1.0],
    [17.173, 10.477, 1.0],
    [14.876, 15.244, 1.0],
    [9.530, 17.474, 1.0],
    [5.115, 14.920, 1.0],
    [2.492, 10.007, 1.0],
    [11.865, 24.855, 1.0],
    [15.298, -1.511, 1.0]
]).T

# Simulação
t, x_true, odometry_noisy, z_hist = simulate_trajectory_motion(
    T, t_final, traj, v_max, w_max,
    sigma_v, sigma_w, sigma_uwb, anchors, baseline, z_c
)

# EKF
x_est = run_bc_ekf_from_data(
    T, anchors, odometry_noisy, z_hist,
    baseline, z_c, sigma_uwb
)

# Plots
viz.plot_trajectory(x_true, x_est, anchors, title="Trajetória Real x Estimada")
