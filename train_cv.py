"""
Script for training a Pulse-3D to classify a pulmonary nodule as benign or malignant
using GroupKFold cross-validation.
"""
import sys
import logging
import random
from datetime import datetime
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.model_2d import ResNet18
from models.Pulse3D import Pulse3D
from dataloader import get_data_loader
from experiment_config import config

# Set matplotlib log level to warning to suppress debug messages
plt.set_loglevel("warning")
torch.backends.cudnn.benchmark = True

# --- Utility Functions ---

def setup_logger(log_dir: Path, name="train"):
    """Sets up a logger that writes to a file and the console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"
    # Remove any existing handlers to avoid duplicate logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers = [logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(name)

# --- Loss and Metric Functions ---

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """Computes the focal loss for binary classification."""
    bce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt)**gamma * bce_loss
    return focal_loss.mean()

def make_weights_for_balanced_classes(labels):
    """Creates sampling weights to handle class imbalance."""
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))
    weights = [n_samples / float(cnt_dict[label]) for label in labels]
    return weights

def compute_metrics(y_true, y_pred, threshold=0.5):
    """Computes AUC, sensitivity, specificity, PPV, and NPV."""
    y_pred_bin = (y_pred >= threshold).astype(int)
    # Handle cases where only one class is present in y_true or y_pred_bin
    if len(np.unique(y_true)) < 2:
        return {"AUC": 0.5, "Sensitivity": 0.0, "Specificity": 0.0, "PPV": 0.0, "NPV": 0.0}
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_bin).ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    auc = metrics.roc_auc_score(y_true, y_pred)
    return {"AUC": auc, "Sensitivity": sens, "Specificity": spec, "PPV": ppv, "NPV": npv}

def save_fold_metrics_plot(fold_dir, history, fold_idx):
    """Saves a 2x3 grid of metric plots for a given fold."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Fold {fold_idx} Metrics", fontsize=16)

    def plot(ax, key_train, key_val, title):
        ax.plot(epochs, history[key_train], label="Train", alpha=0.7)
        ax.plot(epochs, history[key_val], label="Validation", alpha=0.7)
        if "best_epoch" in history:
             ax.axvline(x=history["best_epoch"], color="r", linestyle="--", label=f"Best (Epoch {history['best_epoch']})")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    plot_map = {
        (0, 0): ("train_AUC", "val_AUC", "AUC"),
        (0, 1): ("train_Sensitivity", "val_Sensitivity", "Sensitivity"),
        (0, 2): ("train_Specificity", "val_Specificity", "Specificity"),
        (1, 0): ("train_loss", "val_loss", "Loss"),
        (1, 1): ("train_NPV", "val_NPV", "NPV"),
        (1, 2): ("train_PPV", "val_PPV", "PPV"),
    }
    
    for (r, c), (k_train, k_val, title) in plot_map.items():
        plot(axes[r, c], k_train, k_val, title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fold_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fold_dir / f"metrics_fold_{fold_idx}.png", dpi=300)
    plt.close()


def create_kfold_splits(csv_path, n_splits=5, shuffle=True, random_state=None):
    """Creates GroupKFold splits to ensure patients are not split across train/val sets."""
    df = pd.read_csv(csv_path)
    gkf = GroupKFold(n_splits=n_splits)
    fold_dfs = []
    for train_idx, val_idx in gkf.split(df, groups=df.PatientID):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        val_fold = df.iloc[val_idx].reset_index(drop=True)
        fold_dfs.append((train_fold, val_fold))
    return fold_dfs

# --- Main Training Logic ---

def train_fold(train_df, val_df, exp_root, fold_idx, args, logger):
    """Trains and validates a model for a single fold."""
    fold_dir = exp_root / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 30)
    logger.info(f" FOLD {fold_idx} ")
    logger.info("=" * 30)

    # --- DataLoaders ---
    weights = make_weights_for_balanced_classes(train_df.label.values)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), len(train_df))

    train_loader = get_data_loader(config.DATADIR, train_df, mode=config.MODE, sampler=sampler, workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE, rotations=config.ROTATION, translations=config.TRANSLATION, size_mm=config.SIZE_MM, size_px=config.SIZE_PX)
    valid_loader = get_data_loader(config.DATADIR, val_df, mode=config.MODE, workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE, size_mm=config.SIZE_MM, size_px=config.SIZE_PX)
    
    # --- Model, Optimizer, Loss ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config.MODE == "2D":
        model = ResNet18().to(device)
    elif config.MODE == "3D":
        model = Pulse3D(num_classes=1, input_channels=1, freeze_bn=False).to(device)
    
    loss_function = focal_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # --- Training Loop ---
    best_metric = -1
    best_metric_epoch = -1
    patience_counter = 0
    history = {k: [] for k in ["train_loss", "val_loss", "train_AUC", "val_AUC", "train_Sensitivity", "val_Sensitivity", "train_Specificity", "val_Specificity", "train_PPV", "val_PPV", "train_NPV", "val_NPV"]}
    
    for epoch in range(config.EPOCHS):
        if patience_counter >= config.PATIENCE:
            logger.info(f"Early stopping triggered after {patience_counter} epochs with no improvement.")
            break
        
        logger.info(f"--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        
        # Train phase
        model.train()
        epoch_loss = 0
        train_preds, train_trues = [], []
        for batch_data in tqdm(train_loader, desc="Training"):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_trues.extend(labels.cpu().numpy())
        
        history["train_loss"].append(epoch_loss / len(train_loader))
        train_metrics = compute_metrics(np.array(train_trues), np.array(train_preds))
        for key, val in train_metrics.items():
            history[f"train_{key}"].append(val)

        # Validation phase
        model.eval()
        epoch_loss = 0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for val_data in tqdm(valid_loader, desc="Validating"):
                inputs, labels = val_data["image"].to(device), val_data["label"].float().to(device)
                outputs = model(inputs)
                loss = loss_function(outputs.squeeze(), labels.squeeze())
                epoch_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_trues.extend(labels.cpu().numpy())

        history["val_loss"].append(epoch_loss / len(valid_loader))
        val_metrics = compute_metrics(np.array(val_trues), np.array(val_preds))
        for key, val in val_metrics.items():
            history[f"val_{key}"].append(val)

        logger.info(f"Epoch {epoch + 1} | Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f}")
        logger.info(f"Epoch {epoch + 1} | Train AUC: {train_metrics['AUC']:.4f}, Val AUC: {val_metrics['AUC']:.4f}")

        # Check for improvement
        if val_metrics["AUC"] > best_metric:
            best_metric = val_metrics["AUC"]
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), fold_dir / "best_metric_model.pth")
            logger.info(f"Saved new best model with AUC: {best_metric:.4f} at epoch {best_metric_epoch}")
            patience_counter = 0
        else:
            patience_counter += 1
    
    # After all epochs, load the best model and get its final metrics
    model.load_state_dict(torch.load(fold_dir / "best_metric_model.pth"))
    best_val_metrics = {k.replace("val_", ""): v[best_metric_epoch - 1] for k, v in history.items() if k.startswith("val_")}

    # Save plots and metrics
    history["best_epoch"] = best_metric_epoch
    save_fold_metrics_plot(fold_dir, history, fold_idx)
    pd.DataFrame(history).to_csv(fold_dir / "metrics_log.csv", index=False)
    
    logger.info(f"Fold {fold_idx} finished. Best Val AUC: {best_metric:.4f} at epoch {best_metric_epoch}")
    
    return {
        "fold": fold_idx,
        "best_auc": best_metric,
        "best_epoch": best_metric_epoch,
        **best_val_metrics
    }

def train_cross_validation(csv_path, exp_root, n_folds=5, only_fold=-1, args=None, logger=None):
    """Manages the K-fold cross-validation process."""
    exp_root.mkdir(parents=True, exist_ok=True)
    fold_data = create_kfold_splits(csv_path, n_splits=n_folds, random_state=config.SEED)

    all_results = []
    for fold_idx, (train_df, val_df) in enumerate(fold_data):
        if only_fold != -1 and fold_idx != only_fold:
            continue
        
        result = train_fold(train_df, val_df, exp_root, fold_idx, args, logger)
        all_results.append(result)

        # Append to summary CSV after each fold
        summary_path = exp_root / "cv_summary.csv"
        mode = "a" if summary_path.exists() else "w"
        with open(summary_path, mode) as f:
            if mode == "w":
                f.write("fold,best_auc,best_epoch,val_loss,Sensitivity,Specificity,PPV,NPV\n")
            
            f.write(
                f"{result['fold']},{result['best_auc']:.4f},{result['best_epoch']},"
                f"{result['loss']:.6f},{result['Sensitivity']:.4f},{result['Specificity']:.4f},"
                f"{result['PPV']:.4f},{result['NPV']:.4f}\n"
            )
            
    # Calculate and log average results
    if all_results:
        avg_auc = np.mean([res["best_auc"] for res in all_results])
        logger.info("=" * 30)
        logger.info(" Cross-Validation Summary ")
        logger.info("=" * 30)
        logger.info(f"Average Best AUC across {len(all_results)} folds: {avg_auc:.4f}")

    return all_results

# --- Main Entry Point ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=-1, help="Run only a specific fold (e.g., 0, 1, ...)")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging (not implemented)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (not implemented)")
    args = parser.parse_args()

    # Create a unique identifier for this specific run
    run_identifier = f"{config.EXPERIMENT_NAME}"

    # The experiment root is the directory where all files for this run will go
    exp_root = config.EXPERIMENT_DIR / run_identifier
    
    # Simple log filename (just "train" - the directory is already unique)
    logger = setup_logger(exp_root, name="train")

    # Consolidate all data into one CSV if it doesn't exist
    csv_path = config.CSV_DIR / "all_data.csv"
    if not csv_path.exists():
        train_df = pd.read_csv(config.CSV_DIR_TRAIN)
        valid_df = pd.read_csv(config.CSV_DIR_VALID)
        test_df = pd.read_csv(config.CSV_DIR_TEST)
        all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        all_df.to_csv(csv_path, index=False)
        print(f"Created consolidated dataset at {csv_path}")

    logger.info(f"Starting experiment: {run_identifier}")
    logger.info(f"Using data from: {csv_path}")
    logger.info(f"Saving results to: {exp_root}")

    results = train_cross_validation(csv_path, exp_root, n_folds=5, only_fold=args.fold, args=args, logger=logger)
    logger.info("=" * 30)
    logger.info(f"Finished all folds. Final results summary saved in {exp_root / 'cv_summary.csv'}")