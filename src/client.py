import torch
import collections
from flwr.common import (
    NDArrays,
    Scalar,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr as fl
from src.model import TransUNet
from src.data_loader import get_dataloader
from src.train_utils import ComboLoss, train_one_epoch, evaluate


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, train_imgs, train_masks, test_imgs, test_masks):
        self.cid = cid
        self.train_imgs = train_imgs
        self.train_masks = train_masks
        self.test_imgs = test_imgs
        self.test_masks = test_masks

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransUNet(n_classes=1).to(self.device)
        self.trainloader = get_dataloader(
            self.train_imgs, self.train_masks, batch_size=16
        )
        self.valloader = get_dataloader(self.test_imgs, self.test_masks, batch_size=16)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = collections.OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Extract strategy-specific config
        local_epochs = config.get("local_epochs", 1)
        learning_rate = config.get("lr", 1e-4)
        proximal_mu = config.get("proximal_mu", 0.0)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = ComboLoss()

        # Prepare global params for FedProx
        global_params = None
        if proximal_mu > 0:
            global_params = [
                torch.tensor(p, requires_grad=False).to(self.device) for p in parameters
            ]

        for epoch in range(local_epochs):
            train_one_epoch(
                self.model,
                self.trainloader,
                optimizer,
                criterion,
                self.device,
                proximal_mu,
                global_params,
            )

        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = ComboLoss()
        loss, dice, iou = evaluate(self.model, self.valloader, criterion, self.device)

        return (
            float(loss),
            len(self.valloader.dataset),
            {"dice": float(dice), "iou": float(iou)},
        )
