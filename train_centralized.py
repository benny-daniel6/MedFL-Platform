import torch
import torch.optim as optim
from src.model import TransUNet
from src.data_loader import (
    download_and_unzip_data,
    load_and_prepare_data,
    get_dataloader,
)
from src.train_utils import ComboLoss, train_one_epoch, evaluate


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    download_and_unzip_data()
    train_img, train_mask, test_img, test_mask = load_and_prepare_data()

    train_loader = get_dataloader(train_img, train_mask, batch_size=16, is_train=True)
    test_loader = get_dataloader(test_img, test_mask, batch_size=16, is_train=False)

    model = TransUNet(n_classes=1).to(DEVICE)
    criterion = ComboLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # --- The Final Change: Use a dynamic learning rate scheduler ---
    # This will reduce the learning rate by a factor of 0.1 if the validation loss
    # doesn't improve for 5 epochs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=5
    )

    NUM_EPOCHS = 65
    best_dice = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

        val_loss, val_dice, val_iou = evaluate(model, test_loader, criterion, DEVICE)
        print(
            f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}"
        )

        # The scheduler now steps using the validation loss
        scheduler.step(val_loss)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "transunet_centralized_best.pth")
            print("Saved new best model.")

    print(f"\nTraining complete. Best validation Dice Score: {best_dice:.4f}")


if __name__ == "__main__":
    main()
