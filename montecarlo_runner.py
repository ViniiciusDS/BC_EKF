import numpy as np
from bc_ekf import run_bc_ekf

def run_monte_carlo(
    T,
    t_final,
    anchors,
    v_true,
    w_true,
    l,
    z_c,
    sigma_v,
    sigma_w,
    sigma_uwb,
    N_runs
):
    rmse_pos_list = []
    rmse_heading_list = []
    pos_errors_all_runs = []
    heading_errors_all_runs = []
    trajectories_true = []
    trajectories_est = []

    for run in range(N_runs):
        rmse_pos, rmse_heading, t, x_hist_true, x_hist_est = run_bc_ekf(
            T,
            t_final,
            anchors,
            v_true,
            w_true,
            l,
            z_c,
            sigma_v,
            sigma_w,
            sigma_uwb
        )

        error = x_hist_true - x_hist_est
        error[2,:] = np.arctan2(np.sin(error[2,:]), np.cos(error[2,:]))
        pos_error = np.linalg.norm(error[0:2,:], axis=0)
        heading_error_deg = np.abs(error[2,:])*(180/np.pi)

        rmse_pos_list.append(rmse_pos)
        rmse_heading_list.append(rmse_heading)
        pos_errors_all_runs.append(pos_error)
        heading_errors_all_runs.append(heading_error_deg)
        trajectories_true.append(x_hist_true)
        trajectories_est.append(x_hist_est)

    return (
        np.array(rmse_pos_list),
        np.array(rmse_heading_list),
        np.array(pos_errors_all_runs),
        np.array(heading_errors_all_runs),
        trajectories_true,
        trajectories_est
    )
