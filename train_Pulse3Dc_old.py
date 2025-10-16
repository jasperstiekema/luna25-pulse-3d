from __future__ import annotations

import os
import sys
import math
import json
import random
import shutil
import logging
from datetime import datetime
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import binary_cross_entropy_with_logits
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

from dataloader_cls import get_data_loader
from experiment_config import config
from models.Pulse3D import Pulse3D
from tensorboard_visuals import (
    log_prediction_samples,
    log_roc_curve,
)


# -------------------------------------------------------------------------
# Utility logging setup
# -------------------------------------------------------------------------
def setup_logger(log_dir: Path, name="train"):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(name)


# -------------------------------------------------------------------------
# Loss functions
# -------------------------------------------------------------------------
def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    if inputs.dim() == 1 and targets.dim() == 2:
        inputs = inputs.view(-1, 1)
    elif inputs.dim() == 2 and targets.dim() == 1:
        targets = targets.view(-1, 1)
    bce = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def focal_loss_with_smoothing(inputs, targets, alpha=0.25, gamma=2.0, smoothing=0.1):
    if inputs.dim() == 1 and targets.dim() == 2:
        inputs = inputs.view(-1, 1)
    elif inputs.dim() == 2 and targets.dim() == 1:
        targets = targets.view(-1, 1)
    targets = targets * (1 - smoothing) + smoothing / 2
    bce = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def make_weights_for_balanced_classes(labels: np.ndarray) -> torch.DoubleTensor:
    n = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    frac = {cls: n / (2 * c) for cls, c in zip(*np.unique(labels, return_counts=True))}
    return torch.DoubleTensor([frac[l] for l in labels])


def create_kfold_splits(csv_path, n_splits=5, shuffle=True, random_state=None):
    df = pd.read_csv(csv_path)
    gkf = GroupKFold(n_splits=n_splits)
    fold_dfs = []
    for train_idx, val_idx in gkf.split(df, groups=df.PatientID):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        val_fold = df.iloc[val_idx].reset_index(drop=True)
        fold_dfs.append((train_fold, val_fold))
    return fold_dfs


def calculate_metrics(y_true, y_pred):
    """Calculate AUC, Sensitivity, Specificity, PPV, NPV"""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_binary).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return auc, sensitivity, specificity, ppv, npv


def save_fold_metrics_plot(fold_dir, history):
    """Save a plot of all metrics for train and validation"""
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Fold Metrics", fontsize=16)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', alpha=0.7)
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation', alpha=0.7)
    axes[0, 0].axvline(x=history['best_epoch'], color='r', linestyle='--', label=f"Best ({history['best_epoch']})")
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # AUC
    axes[0, 1].plot(epochs, history['train_auc'], label='Train', alpha=0.7)
    axes[0, 1].plot(epochs, history['val_auc'], label='Validation', alpha=0.7)
    axes[0, 1].axvline(x=history['best_epoch'], color='r', linestyle='--', label=f"Best ({history['best_epoch']})")
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_title('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Sensitivity
    axes[0, 2].plot(epochs, history['train_sens'], label='Train', alpha=0.7)
    axes[0, 2].plot(epochs, history['val_sens'], label='Validation', alpha=0.7)
    axes[0, 2].axvline(x=history['best_epoch'], color='r', linestyle='--', label=f"Best ({history['best_epoch']})")
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Sensitivity')
    axes[0, 2].set_title('Sensitivity (TPR)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Specificity
    axes[1, 0].plot(epochs, history['train_spec'], label='Train', alpha=0.7)
    axes[1, 0].plot(epochs, history['val_spec'], label='Validation', alpha=0.7)
    axes[1, 0].axvline(x=history['best_epoch'], color='r', linestyle='--', label=f"Best ({history['best_epoch']})")
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Specificity')
    axes[1, 0].set_title('Specificity (TNR)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # PPV
    axes[1, 1].plot(epochs, history['train_ppv'], label='Train', alpha=0.7)
    axes[1, 1].plot(epochs, history['val_ppv'], label='Validation', alpha=0.7)
    axes[1, 1].axvline(x=history['best_epoch'], color='r', linestyle='--', label=f"Best ({history['best_epoch']})")
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('PPV')
    axes[1, 1].set_title('PPV (Precision)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # NPV
    axes[1, 2].plot(epochs, history['train_npv'], label='Train', alpha=0.7)
    axes[1, 2].plot(epochs, history['val_npv'], label='Validation', alpha=0.7)
    axes[1, 2].axvline(x=history['best_epoch'], color='r', linestyle='--', label=f"Best ({history['best_epoch']})")
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('NPV')
    axes[1, 2].set_title('NPV')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(fold_dir / "fold_graph.png", dpi=300, bbox_inches='tight')
    plt.close()


# -------------------------------------------------------------------------
# Core training
# -------------------------------------------------------------------------
def train_fold(train_df, valid_df, exp_root, fold_idx, args, logger):
    fold_dir = exp_root / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"========== Training Fold {fold_idx} ==========")
    tensorboard_dir = fold_dir / "tensorboard"

    if tensorboard_dir.exists():
        shutil.rmtree(tensorboard_dir)
        logger.info("Removed previous TensorBoard directory.")

    writer = None if args.no_tensorboard else SummaryWriter(log_dir=tensorboard_dir)

    logger.info(
        f"Training set: {len(train_df)} (malignant={train_df.label.sum()}, benign={len(train_df) - train_df.label.sum()})"
    )
    logger.info(
        f"Validation set: {len(valid_df)} (malignant={valid_df.label.sum()}, benign={len(valid_df) - valid_df.label.sum()})"
    )

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=make_weights_for_balanced_classes(train_df.label.values),
        num_samples=len(train_df),
        replacement=True,
    )

    common_loader_args = dict(
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    train_loader = get_data_loader(
        config.DATADIR,
        train_df,
        mode="3D",
        sampler=sampler,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        use_monai_transforms=True,
        **common_loader_args,
    )

    valid_loader = get_data_loader(
        config.DATADIR,
        valid_df,
        mode="3D",
        rotations=None,
        translations=None,
        use_monai_transforms=False,
        **common_loader_args,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pulse3D().to(device)

    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        logger.info(f"Resumed from checkpoint: {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = torch.nn.BCEWithLogitsLoss()

    phase1_epochs = 20
    best_auc, best_epoch = -1.0, -1
    patience_counter = 0
    
    # History for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'train_sens': [], 'val_sens': [],
        'train_spec': [], 'val_spec': [],
        'train_ppv': [], 'val_ppv': [],
        'train_npv': [], 'val_npv': [],
        'best_epoch': -1
    }

    disable_tqdm = not sys.stdout.isatty()

    for epoch in range(phase1_epochs):
        logger.info(f"\n[Phase 1][Fold {fold_idx}] Epoch {epoch+1}/{phase1_epochs}")

        # ---- Train ----
        model.train()
        train_loss = 0.0
        y_train_true, y_train_raw = [], []
        for step, batch in enumerate(tqdm(train_loader, desc="Train", disable=disable_tqdm), 1):
            optimizer.zero_grad(set_to_none=True)
            ct = batch["image"].to(device)
            lbl = batch["label"].float().to(device)

            logits = model(ct).squeeze(1)
            loss = criterion(logits.view(-1), lbl.view(-1))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            y_train_true.append(lbl.cpu())
            y_train_raw.append(logits.cpu())

            if step % 100 == 0:
                logger.info(f"Step {step}: loss={loss.item():.4f}")
                if writer:
                    writer.add_scalar("Loss/train_step", loss.item(), epoch * len(train_loader) + step)

        avg_train_loss = train_loss / step
        y_train_true = torch.cat(y_train_true).numpy()
        y_train_pred = torch.sigmoid(torch.cat(y_train_raw)).detach().numpy()
        train_auc, train_sens, train_spec, train_ppv, train_npv = calculate_metrics(y_train_true, y_train_pred)
        
        history['train_loss'].append(avg_train_loss)
        history['train_auc'].append(train_auc)
        history['train_sens'].append(train_sens)
        history['train_spec'].append(train_spec)
        history['train_ppv'].append(train_ppv)
        history['train_npv'].append(train_npv)
        
        if writer:
            writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        logger.info(f"Train loss: {avg_train_loss:.6f} | AUC={train_auc:.4f}")

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        y_true, y_raw = [], []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(valid_loader, desc="Val", disable=disable_tqdm), 1):
                ct = batch["image"].to(device)
                lbl = batch["label"].float().to(device)
                logits = model(ct).squeeze(1)
                loss = criterion(logits.view(-1), lbl.view(-1))
                val_loss += loss.item()
                y_true.append(lbl.cpu())
                y_raw.append(logits.cpu())

        avg_val_loss = val_loss / step
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.sigmoid(torch.cat(y_raw)).detach().numpy()
        auc, sens, spec, ppv, npv = calculate_metrics(y_true, y_pred)
        
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(auc)
        history['val_sens'].append(sens)
        history['val_spec'].append(spec)
        history['val_ppv'].append(ppv)
        history['val_npv'].append(npv)

        logger.info(f"Val loss: {avg_val_loss:.6f} | AUC={auc:.4f} | Sens={sens:.4f} | Spec={spec:.4f}")
        if writer:
            writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
            writer.add_scalar("Metrics/AUC", auc, epoch)

        if auc > best_auc:
            best_auc, best_epoch = auc, epoch + 1
            torch.save(model.state_dict(), fold_dir / "fold_best.pth")
            logger.info(f"Saved new best model at epoch {best_epoch} (AUC={best_auc:.4f})")
            history['best_epoch'] = best_epoch

    # ---- Save final metrics ----
    results = {"best_auc": best_auc, "best_epoch": best_epoch}
    with open(fold_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save plot
    save_fold_metrics_plot(fold_dir, history)
    logger.info(f"Saved metrics plot to {fold_dir / 'fold_graph.png'}")

    if writer:
        writer.close()
    return results


# -------------------------------------------------------------------------
# Cross-validation routine
# -------------------------------------------------------------------------
def train_cross_validation(csv_path, exp_root, n_folds=5, only_fold=-1, args=None, logger=None):
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    exp_root.mkdir(parents=True, exist_ok=True)
    fold_data = create_kfold_splits(csv_path, n_splits=n_folds, random_state=config.SEED)

    all_results = []

    for fold_idx, (train_df, val_df) in enumerate(fold_data):
        if only_fold != -1 and fold_idx != only_fold:
            continue
        res = train_fold(train_df, val_df, exp_root, fold_idx, args, logger)
        all_results.append(res)

        # Append to summary CSV
        summary_path = exp_root / "cv_summary.csv"
        mode = "a" if summary_path.exists() else "w"
        with open(summary_path, mode) as f:
            if mode == "w":
                f.write("fold,best_auc,best_epoch\n")
            f.write(f"{fold_idx},{res['best_auc']:.4f},{res['best_epoch']}\n")

    return all_results


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=-1, help="Run only this fold")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    exp_name = f"{config.EXPERIMENT_NAME}-Pulse3D-{datetime.today().strftime('%Y%m%d')}"
    exp_root = config.EXPERIMENT_DIR / exp_name

    csv_path = config.CSV_DIR / "all_data_.csv"
    if not csv_path.exists():
        train_df = pd.read_csv(config.CSV_DIR_TRAIN)
        valid_df = pd.read_csv(config.CSV_DIR_VALID)
        test_df = pd.read_csv(config.CSV_DIR_TEST)
        all_df = pd.concat([train_df, valid_df], ignore_index=True)
        all_df_ = pd.concat([all_df, test_df], ignore_index=True)
        all_df_.to_csv(csv_path, index=False)

    logger = setup_logger(exp_root)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Data CSV: {csv_path}")
    logger.info(f"TensorBoard: {'Disabled' if args.no_tensorboard else 'Enabled'}")

    results = train_cross_validation(csv_path, exp_root, n_folds=5, only_fold=args.fold, args=args, logger=logger)
    logger.info(f"Finished all folds. Results: {results}")