# compare_configurations.py

import sys
from scenarios import (
    scenarios_group1, scenarios_group2, scenarios_group3,
    scenarios_group4, scenarios_rectangular, scenarios_all
)
from montecarlo_runner import run_monte_carlo
import visualization as viz
import numpy as np
from bc_ekf import run_bc_ekf, run_bc_ekf_custom_commands

# Parâmetros gerais fixos
T = 0.05
t_final = 50
z_c = 0.5
sigma_v = 0.02
sigma_w = 0.05
N_runs = 30

# Determina qual grupo rodar
group = sys.argv[1] if len(sys.argv) > 1 else "group1"

if group == "group1":
    scenarios = scenarios_group1
elif group == "group2":
    scenarios = scenarios_group2
elif group == "group3":
    scenarios = scenarios_group3
elif group == "group4":
    scenarios = scenarios_group4
elif group == "real_tectrol":
    scenarios = scenarios_rectangular
elif group == "all":
    scenarios = scenarios_all
else:
    raise ValueError(f"Grupo inválido: {group}")

results = {}
t = np.arange(0, t_final+T, T)

for scenario in scenarios:
    label = scenario["label"]
    anchors = scenario["anchors"]
    l = scenario["baseline"]
    sigma_uwb = scenario["sigma_uwb"]

    print(f"\nRodando cenário: {label}")

    if "Retangular" in label:
        # ===============================
        # Cenário especial com rota retangular
        # ===============================
        v_seg = 0.3
        w_turn = np.pi / 2 / 2.5  # gira 90° em 2.5s

        # Define segmentos da trajetória
        steps_straight = int(7.5 / T)
        steps_turn = int(2.5 / T)

        v_straight = np.full(steps_straight, v_seg)
        w_straight = np.zeros(steps_straight)
        v_curve = np.zeros(steps_turn)
        w_curve = np.full(steps_turn, w_turn)

        v_commands = np.concatenate([v_straight, v_curve]*4)
        w_commands = np.concatenate([w_straight, w_curve]*4)

        t_final_actual = len(v_commands)*T
        t = np.arange(0, t_final_actual, T)

        # Executa várias rodadas com comandos customizados
        rmse_pos_list = []
        rmse_heading_list = []
        pos_errors_all_runs = []
        heading_errors_all_runs = []
        trajectories_true = []
        trajectories_est = []

        for run in range(N_runs):
            rmse_pos, rmse_heading, t_sim, x_hist_true, x_hist_est = run_bc_ekf_custom_commands(
                T,
                t_final_actual,
                anchors,
                v_commands,
                w_commands,
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

        results[label] = {
            "rmse_pos": np.array(rmse_pos_list),
            "rmse_heading": np.array(rmse_heading_list),
            "pos_errors": np.array(pos_errors_all_runs),
            "heading_errors": np.array(heading_errors_all_runs),
            "true_traj": trajectories_true,
            "est_traj": trajectories_est,
            "anchors": anchors
        }

    else:
        # ===============================
        # Cenários normais (trajetória circular)
        # ===============================
        v_true = scenario.get("v_true", 0.3)
        w_true = scenario.get("w_true", np.deg2rad(7.5))

        (
            rmse_pos,
            rmse_heading,
            pos_errors_all,
            heading_errors_all,
            trajectories_true,
            trajectories_est
        ) = run_monte_carlo(
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
        )

        results[label] = {
            "rmse_pos": rmse_pos,
            "rmse_heading": rmse_heading,
            "pos_errors": pos_errors_all,
            "heading_errors": heading_errors_all,
            "true_traj": trajectories_true,
            "est_traj": trajectories_est,
            "anchors": anchors
        }

scenario_labels = [s["label"] for s in scenarios]

# Visualizações
viz.plot_rmse_boxplots(results, scenario_labels)
viz.plot_error_over_time(results, scenario_labels, t)
viz.plot_example_trajectories(results, scenario_labels, t)
viz.plot_comparative_position_error(results, scenario_labels, t)
viz.plot_comparative_heading_error(results, scenario_labels, t)
