import matplotlib.pyplot as plt
import os


def save_comparison_plot(history_fedavg, history_fedprox, metric, save_path="results"):
    """
    Saves a plot comparing the performance of FedAvg and FedProx over rounds.

    Args:
        history_fedavg (flwr.common.History): History object from a FedAvg simulation.
        history_fedprox (flwr.common.History): History object from a FedProx simulation.
        metric (str): The metric to plot (e.g., 'dice').
        save_path (str): Directory to save the plot.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Extract data for FedAvg
    rounds_fedavg, metric_fedavg = zip(*history_fedavg.metrics_centralized[metric])

    # Extract data for FedProx
    rounds_fedprox, metric_fedprox = zip(*history_fedprox.metrics_centralized[metric])

    plt.figure(figsize=(10, 6))
    plt.plot(
        rounds_fedavg,
        metric_fedavg,
        marker="o",
        linestyle="-",
        label="FedAvg (Non-IID)",
    )
    plt.plot(
        rounds_fedprox,
        metric_fedprox,
        marker="x",
        linestyle="--",
        label="FedProx (Non-IID)",
    )

    plt.title(f"FedAvg vs. FedProx: Centralized {metric.capitalize()} on Test Set")
    plt.xlabel("Federated Learning Round")
    plt.ylabel(f"{metric.capitalize()} Score")
    plt.grid(True)
    plt.legend()
    plt.xticks(range(min(rounds_fedavg), max(rounds_fedavg) + 1))

    plot_filename = os.path.join(save_path, "fedavg_vs_fedprox_comparison.png")
    plt.savefig(plot_filename)
    print(f"Comparison plot saved to {plot_filename}")
