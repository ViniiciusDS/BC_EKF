import sys
from scenarios import scenarios_group1, scenarios_group2, scenarios_group3, scenarios_all
from montecarlo_runner import run_monte_carlo
import visualization as viz
import numpy as np

# Parâmetros fixos
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
    v_true = scenario.get("v_true", 0.3)
    w_true = scenario.get("w_true", np.deg2rad(7.5))

    print(f"\nRodando cenário: {label}")

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

viz.plot_rmse_boxplots(results, scenario_labels)
viz.plot_error_over_time(results, scenario_labels, t)
viz.plot_example_trajectories(results, scenario_labels, t)
viz.plot_comparative_position_error(results, scenario_labels, t)
viz.plot_comparative_heading_error(results, scenario_labels, t)
