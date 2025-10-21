"""
Script for training a Pulse-3D to classify a pulmonary nodule as benign or malignant.
"""
from models.model_2d import ResNet18
from models.Pulse3D import Pulse3D
from dataloader import get_data_loader
import logging
import numpy as np
import torch
import sklearn.metrics as metrics
from tqdm import tqdm
import warnings
import random
import pandas
from experiment_config import config
from datetime import datetime
import argparse
from torch.nn.functional import binary_cross_entropy_with_logits

import matplotlib
import matplotlib.pyplot as plt
matplotlib.set_loglevel("warning")


torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)


# -----------------------------
# Loss and Metric Functions
# -----------------------------

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    bce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def make_weights_for_balanced_classes(labels):
    """Create sampling weights for dealing with class imbalance."""
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))
    weights = [n_samples / float(cnt_dict[label]) for label in labels]
    return weights


def compute_metrics(y_true, y_pred, threshold=0.5):
    """Compute AUC, sensitivity, specificity, PPV, and NPV."""
    y_pred_bin = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_bin).ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    auc = metrics.roc_auc_score(y_true, y_pred)
    return {"AUC": auc, "Sensitivity": sens, "Specificity": spec, "PPV": ppv, "NPV": npv}

def save_metrics_plot(metrics_df, save_path, best_epoch=None):
    """Plot all metrics (and loss) in a single 2×3 grid figure."""
    from pathlib import Path
    save_path = Path(save_path)
    metrics_to_plot = ["AUC", "Sensitivity", "Specificity",  "Loss", "PPV", "NPV"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]

        if metric == "Loss":
            ax.plot(metrics_df["epoch"], metrics_df["train_loss"], "--", label="Train Loss")
            ax.plot(metrics_df["epoch"], metrics_df["valid_loss"], label="Valid Loss", color="C1")
        else:
            ax.plot(metrics_df["epoch"], metrics_df[f"train_{metric}"], "--", label=f"Train {metric}")
            ax.plot(metrics_df["epoch"], metrics_df[f"valid_{metric}"], label=f"Valid {metric}", color="C1")

        if best_epoch is not None:
            ax.axvline(best_epoch, color="r", linestyle=":", label=f"Best Epoch ({best_epoch})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path / "metrics_overview.png", dpi=150)
    plt.close()

# Training Function

def train(train_csv_path, valid_csv_path, test_csv_path, exp_save_root):
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    logging.info(f"Training with {train_csv_path}")
    logging.info(f"Validating with {valid_csv_path}")

    train_df = pandas.read_csv(train_csv_path)
    valid_df = pandas.read_csv(valid_csv_path)
    test_df = pandas.read_csv(test_csv_path)
    train_df = pandas.concat([train_df, test_df], ignore_index=True)

    print()
    logging.info(f"Number of malignant training samples: {train_df.label.sum()}")
    logging.info(f"Number of benign training samples: {len(train_df) - train_df.label.sum()}")
    print()
    logging.info(f"Number of malignant validation samples: {valid_df.label.sum()}")
    logging.info(f"Number of benign validation samples: {len(valid_df) - valid_df.label.sum()}")

    # Data loaders
    weights = make_weights_for_balanced_classes(train_df.label.values)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_df))

    train_loader = get_data_loader(
        config.DATADIR,
        train_df,
        mode=config.MODE,
        sampler=sampler,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    valid_loader = get_data_loader(
        config.DATADIR,
        valid_df,
        mode=config.MODE,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    device = torch.device("cuda:0")

    if config.MODE == "2D":
        model = ResNet18().to(device)
    elif config.MODE == "3D":
        model = Pulse3D(
            num_classes=1,
            input_channels=1,
            freeze_bn=False,
        ).to(device)

    # Resume training (optional)
    resume_checkpoint = None
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        logging.info(f"Resumed training")

    # Optimizer & loss
    loss_function = focal_loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    best_metric = -1
    best_metric_epoch = -1
    epochs = config.EPOCHS
    patience = config.PATIENCE
    counter = 0
    metrics_log = []

    # Training Loop
    for epoch in range(epochs):
        if counter > patience:
            logging.info(f"Model not improving for {patience} epochs")
            break

        logging.info("-" * 10)
        logging.info(f"Epoch {epoch + 1}/{epochs}")

        # Train phase
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_epoch_loss = epoch_loss / step
        logging.info(f"Epoch {epoch + 1} average train loss: {train_epoch_loss:.7f}")

        # Validation phase
        model.eval()
        epoch_loss = 0
        step = 0
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y_true = torch.tensor([], dtype=torch.float32, device=device)

        with torch.no_grad():
            for val_data in valid_loader:
                step += 1
                val_images, val_labels = val_data["image"].to(device), val_data["label"].float().to(device)
                outputs = model(val_images)
                loss = loss_function(outputs.squeeze(), val_labels.squeeze())
                epoch_loss += loss.item()
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y_true = torch.cat([y_true, val_labels], dim=0)

        valid_epoch_loss = epoch_loss / step
        logging.info(f"Epoch {epoch + 1} average valid loss: {valid_epoch_loss:.7f}")

        # Compute metrics
        y_pred_np = torch.sigmoid(y_pred.reshape(-1)).cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        valid_metrics = compute_metrics(y_true_np, y_pred_np)

        # Compute training metrics for this epoch
        model.eval()
        with torch.no_grad():
            y_train_pred = torch.tensor([], dtype=torch.float32, device=device)
            y_train_true = torch.tensor([], dtype=torch.float32, device=device)
            for train_batch in train_loader:
                imgs, lbls = train_batch["image"].to(device), train_batch["label"].float().to(device)
                outputs = model(imgs)
                y_train_pred = torch.cat([y_train_pred, outputs], dim=0)
                y_train_true = torch.cat([y_train_true, lbls], dim=0)
            y_train_pred_np = torch.sigmoid(y_train_pred.reshape(-1)).cpu().numpy()
            y_train_true_np = y_train_true.cpu().numpy()
            train_metrics = compute_metrics(y_train_true_np, y_train_pred_np)

        # Log metrics
        logging.info(f"Epoch {epoch+1} Train Metrics: {train_metrics}")
        logging.info(f"Epoch {epoch+1} Valid Metrics: {valid_metrics}")

        metrics_log.append({
            "epoch": epoch + 1,
            "train_loss": train_epoch_loss,
            "valid_loss": valid_epoch_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        })

        # Check for improvement
        auc_metric = valid_metrics["AUC"]
        if auc_metric > best_metric:
            counter = 0
            best_metric = auc_metric
            best_metric_epoch = epoch + 1

            torch.save(model.state_dict(), exp_save_root / "best_metric_model.pth")
            np.save(exp_save_root / "config.npy", {
                "train_csv": train_csv_path,
                "valid_csv": valid_csv_path,
                "config": config,
                "best_auc": best_metric,
                "epoch": best_metric_epoch,
            })
            logging.info("Saved new best metric model")
        else:
            counter += 1

        logging.info(
            f"Current epoch: {epoch + 1} | Valid AUC: {auc_metric:.4f} | "
            f"Best AUC: {best_metric:.4f} (epoch {best_metric_epoch})"
        )

    # Save final metrics and plots
    metrics_df = pandas.DataFrame(metrics_log)
    metrics_df.to_csv(exp_save_root / "metrics_log.csv", index=False)
    save_metrics_plot(metrics_df, exp_save_root, best_epoch=best_metric_epoch)
    logging.info(f"Saved metrics CSV and plot to {exp_save_root}")

    logging.info(f"Training completed. Best AUC: {best_metric:.4f} at epoch {best_metric_epoch}")



# Main Entry Point

if __name__ == "__main__":
    experiment_name = f"{config.EXPERIMENT_NAME}"
    exp_save_root = config.EXPERIMENT_DIR / experiment_name
    exp_save_root.mkdir(parents=True, exist_ok=True)

    train(
        train_csv_path=config.CSV_DIR_TRAIN,
        valid_csv_path=config.CSV_DIR_VALID,
        test_csv_path=config.CSV_DIR_TEST,
        exp_save_root=exp_save_root,
    )
