# visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_rmse_boxplots(results, scenario_labels):
    """
    Plota boxplots do RMSE de posição e heading para todos os cenários.
    """
    plt.figure(figsize=(12, 5))

    # Boxplot RMSE posição
    plt.subplot(1, 2, 1)
    plt.boxplot([results[label]["rmse_pos"] for label in scenario_labels], labels=scenario_labels)
    plt.title("Comparação RMSE Posição")
    plt.ylabel("Erro (m)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)

    # Boxplot RMSE heading
    plt.subplot(1, 2, 2)
    plt.boxplot([results[label]["rmse_heading"] for label in scenario_labels], labels=scenario_labels)
    plt.title("Comparação RMSE Heading")
    plt.ylabel("Erro (graus)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_error_over_time(results, scenario_labels, t):
    """
    Plota erro médio e desvio padrão ao longo do tempo para cada cenário.
    """
    for label in scenario_labels:
        pos_errors = results[label]["pos_errors"]
        heading_errors = results[label]["heading_errors"]

        # Cálculo das estatísticas
        mean_pos = np.mean(pos_errors, axis=0)
        std_pos = np.std(pos_errors, axis=0)

        mean_heading = np.mean(heading_errors, axis=0)
        std_heading = np.std(heading_errors, axis=0)

        plt.figure(figsize=(10, 6))

        # Erro de posição
        plt.subplot(2, 1, 1)
        plt.plot(t, mean_pos, label="Média")
        plt.fill_between(t, mean_pos - std_pos, mean_pos + std_pos, alpha=0.3)
        plt.title(f"Erro de Posição - {label}")
        plt.ylabel("Erro (m)")
        plt.grid()

        # Erro de heading
        plt.subplot(2, 1, 2)
        plt.plot(t, mean_heading, label="Média")
        plt.fill_between(t, mean_heading - std_heading, mean_heading + std_heading, alpha=0.3)
        plt.title(f"Erro de Heading - {label}")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Erro (graus)")
        plt.grid()

        plt.tight_layout()
        plt.show()

def plot_example_trajectories(results, scenario_labels, t, n_examples=5, seed=42):
    """
    Plota exemplos de trajetórias reais e estimadas de cada cenário.
    """
    np.random.seed(seed)

    for label in scenario_labels:
        idx_examples = np.random.choice(len(results[label]["true_traj"]), size=n_examples, replace=False)
        anchors = results[label]["anchors"]

        plt.figure(figsize=(12, 10))

        for i, idx in enumerate(idx_examples, 1):
            true_traj = results[label]["true_traj"][idx]
            est_traj = results[label]["est_traj"][idx]

            plt.subplot(3, 2, i)
            plt.plot(true_traj[0, :], true_traj[1, :], 'k-', label="Real")
            plt.plot(est_traj[0, :], est_traj[1, :], 'b--', label="Estimado")
            plt.scatter(anchors[0, :], anchors[1, :], c='r', marker='*', s=80, label="Âncoras")
            plt.title(f"{label} - Execução {idx + 1}")
            plt.axis("equal")
            plt.grid()
            if i == 1:
                plt.legend()

        plt.tight_layout()
        plt.show()

def plot_comparative_position_error(results, scenario_labels, t):
    """
    Plota a média do erro de posição de todos os cenários em um único gráfico.
    """
    plt.figure(figsize=(10, 6))
    for label in scenario_labels:
        pos_errors = results[label]["pos_errors"]
        mean_pos = np.mean(pos_errors, axis=0)
        plt.plot(t, mean_pos, label=label)

    plt.title("Erro Médio de Posição - Comparação entre Cenários")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Erro de Posição (m)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_comparative_heading_error(results, scenario_labels, t):
    """
    Plota a média do erro de heading de todos os cenários em um único gráfico.
    """
    plt.figure(figsize=(10, 6))
    for label in scenario_labels:
        heading_errors = results[label]["heading_errors"]
        mean_heading = np.mean(heading_errors, axis=0)
        plt.plot(t, mean_heading, label=label)

    plt.title("Erro Médio de Heading - Comparação entre Cenários")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Erro de Heading (graus)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_trajectory(x_true, x_est, anchors, title="Trajetória Real x Estimada"):
    """
    Plota a trajetória real e estimada junto com as âncoras.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(x_true[0, :], x_true[1, :], 'k-', label="Trajetória Real")
    plt.plot(x_est[0, :], x_est[1, :], 'b--', label="Trajetória Estimada")
    plt.scatter(anchors[0, :], anchors[1, :], marker="*", c="red", s=100, label="Âncoras")
    plt.axis("equal")
    plt.grid()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
