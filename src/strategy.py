import flwr as fl
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg

from opacus import PrivacyEngine


class FedAvgWithDP(FedAvg):
    """
    Custom FedAvg strategy that adds Differential Privacy noise on the server side.
    This demonstrates Central Differential Privacy.
    """

    def __init__(self, *args, noise_multiplier=1.0, clipping_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[
            Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]
        ],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # 1. Aggregate weights as usual
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # 2. Convert aggregated parameters to a list of numpy arrays
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

            # 3. Add DP noise
            # A simplified central DP model: add noise to the aggregated update.
            # A more correct implementation would clip each client's update before aggregation.
            # This is a simplification to work within Flower's Strategy API.

            noised_ndarrays = []
            for arr in aggregated_ndarrays:
                noise = np.random.normal(
                    0, self.noise_multiplier * self.clipping_norm, arr.shape
                )
                noised_ndarrays.append(arr + noise)

            print("Applied Central DP noise to aggregated weights.")
            aggregated_parameters = ndarrays_to_parameters(noised_ndarrays)

        return aggregated_parameters, aggregated_metrics
