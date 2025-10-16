import torch
import os, sys
import numpy as np
from pandas import read_csv, DataFrame
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.Pulse3D import Pulse3D
from dataloader_own import get_data_loader

# ------------------------------------------------------------
FOV = 50
model_path = r"D:\PULSE\results\auc check\lr_1e-4_auc_check-3D-20251009\0.8788_model.pth"
csv_dir = r"D:/DATA/LBxSF_labeled_segmented_radius.csv"
data_dir = rf"D:/DATA/own crops {FOV}"
# ------------------------------------------------------------


def evaluate_auc(model, data_loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float().view(-1, 1)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    model.train()
    return auc


def main():
    df = read_csv(csv_dir)
    labels = df["label"].values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # 🔁 Stratified K-Fold setup
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        print(f"\n{'='*30}")
        print(f"🧩 Fold {fold+1}/{k_folds}")
        print(f"{'='*30}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # Data loaders
        train_loader = get_data_loader(
            data_dir, train_df, mode="3D", workers=4, batch_size=4,
            rotations=None, translations=None, size_mm=FOV, size_px=64,
        )
        val_loader = get_data_loader(
            data_dir, val_df, mode="3D", workers=2, batch_size=4,
            rotations=None, translations=None, size_mm=FOV, size_px=64,
        )

        # ------------------------------------------------------------
        # Model setup — reinitialize head each fold
        model = Pulse3D(num_classes=1, input_channels=1, freeze_bn=True)
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        model = model.to(device)

        # Freeze backbone, only fine-tune head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=1e-4)
        num_epochs = 5

        # ------------------------------------------------------------
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).float().view(-1, 1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            train_auc = evaluate_auc(model, train_loader, device)
            val_auc = evaluate_auc(model, val_loader, device)

            print(f"Fold {fold+1} | Epoch [{epoch+1}/{num_epochs}] "
                  f"| Loss: {avg_loss:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

        # ------------------------------------------------------------
        # Save model for this fold
        fold_model_path = model_path.replace(".pth", f"_fold{fold+1}_finetuned_head.pth")
        torch.save(model.state_dict(), fold_model_path)
        print(f"✅ Fold {fold+1} model saved to {fold_model_path}")

        # Final AUC after last epoch
        fold_results.append({
            "fold": fold + 1,
            "train_auc": train_auc,
            "val_auc": val_auc
        })

    # ------------------------------------------------------------
    # Aggregate results
    results_df = DataFrame(fold_results)
    print("\n📊 Cross-validation summary:")
    print(results_df)
    print(f"\nMean Train AUC: {results_df.train_auc.mean():.4f} ± {results_df.train_auc.std():.4f}")
    print(f"Mean Val AUC:   {results_df.val_auc.mean():.4f} ± {results_df.val_auc.std():.4f}")

    results_csv = model_path.replace(".pth", "_cv_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"✅ Saved fold AUCs to {results_csv}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
