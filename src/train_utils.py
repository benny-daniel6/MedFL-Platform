import torch
import torch.nn as nn
from tqdm import tqdm


# --- Loss Function (Combo Loss) ---
class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss()

    def dice_loss(self, y_pred, y_true, epsilon=1e-6):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()

        intersection = (y_pred * y_true).sum(dim=2).sum(dim=2)
        union = y_pred.sum(dim=2).sum(dim=2) + y_true.sum(dim=2).sum(dim=2)

        dice = (2.0 * intersection + epsilon) / (union + epsilon)
        return 1 - dice.mean()

    def forward(self, y_pred, y_true):
        return self.alpha * self.bce(y_pred, y_true) + self.beta * self.dice_loss(
            y_pred, y_true
        )


# --- Evaluation Metrics ---
def dice_coefficient(preds, targets, epsilon=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.item()


def iou(preds, targets, epsilon=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou_score = (intersection + epsilon) / (union + epsilon)
    return iou_score.item()


# --- Training and Testing Loops ---
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    proximal_term=0.0,
    global_params=None,
):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # For FedProx: add the proximal term to the loss
        if proximal_term > 0 and global_params is not None:
            prox_loss = 0.0
            for local_param, global_param in zip(model.parameters(), global_params):
                prox_loss += ((local_param - global_param) ** 2).sum()
            loss += (proximal_term / 2) * prox_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_dice += dice_coefficient(outputs, masks)
            total_iou += iou(outputs, masks)

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_dice, avg_iou
