# run_client.py
import flwr as fl
import argparse
import torch
from src.client import FlowerClient
from src.data_loader import load_and_prepare_data, partition_data


def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    parser.add_argument(
        "--num_clients", type=int, default=10, help="Total number of clients"
    )
    args = parser.parse_args()

    # Load and partition data
    train_img, train_mask, test_img, test_mask = load_and_prepare_data()
    # We partition all data but only use the part for this client
    partitions_img, partitions_mask = partition_data(
        train_img, train_mask, args.num_clients, is_iid=False
    )

    client_train_imgs = partitions_img[args.cid]
    client_train_masks = partitions_mask[args.cid]

    # Start Flower client
    # client = FlowerClient(
    #     str(args.cid), client_train_imgs, client_train_masks, test_img, test_mask
    # )
    # fl.client.start_client(server_address="127.0.0.1:8080", client=client)
    # This is the NEW, corrected code
    numpy_client = FlowerClient(
        str(args.cid), client_train_imgs, client_train_masks, test_img, test_mask
    )
    fl.client.start_client(
        server_address="127.0.0.1:8080", client=numpy_client.to_client()
    )


if __name__ == "__main__":
    main()
