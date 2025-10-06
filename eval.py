import torch
from pandas import read_csv
from models.Pulse3D import Pulse3D
from dataloader_own import get_data_loader


def main():
    df = read_csv("D:/DATA/LBxSF_labeled_segmented_radius.csv")

    data_loader = get_data_loader(
        "D:/DATA/own dataset crops",
        df,
        mode="3D",
        workers=4,
        batch_size=2,
        rotations=None,
        translations=None,
        size_mm=50,
        size_px=64,
    )

    model = Pulse3D(num_classes=1, input_channels=1, freeze_bn=True)
    device = torch.device("cpu")
    model = model.to(device)

    model.eval()

    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y_true = torch.tensor([], dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float().unsqueeze(1)

            outputs = model(images)

            y_pred = torch.cat([y_pred, outputs], dim=0)
            y_true = torch.cat([y_true, labels], dim=0)
            print(f"Processed batch with {images.size(0)} samples.")

    print("✅ Evaluation complete.")
    print(f"Predictions: {y_pred.shape}, Labels: {y_true.shape}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # for Windows safety
    main()
