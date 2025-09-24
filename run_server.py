import flwr as fl
import argparse


def main():
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument(
        "--rounds", type=int, default=5, help="Number of federated learning rounds."
    )
    args = parser.parse_args()

    # Define a simple FedAvg strategy for the cloud server
    # The clients connecting to this will handle their own logic
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Train on all connected clients
        fraction_evaluate=1.0,
        min_available_clients=2,  # Minimum number of clients to start a round
    )

    print(f"Starting Flower server for {args.rounds} rounds...")

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
