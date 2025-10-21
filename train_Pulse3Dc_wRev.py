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


# Utility logging setup
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


# Loss functions
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


# Helpers
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


def save_fold_metrics_plot(fold_dir, history, fold_idx):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)) 
    fig.suptitle(f"Fold {fold_idx} Metrics", fontsize=16)

    def plot(ax, key_train, key_val, title):
        ax.plot(epochs, history[key_train], label="Train", alpha=0.7)
        ax.plot(epochs, history[key_val], label="Validation", alpha=0.7)
        ax.axvline(x=history["best_epoch"], color="r", linestyle="--")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    # Row 1: auc, sens, spec
    plot(axes[0, 0], "train_auc", "val_auc", "AUC")
    plot(axes[0, 1], "train_sens", "val_sens", "Sensitivity")
    plot(axes[0, 2], "train_spec", "val_spec", "Specificity")
    
    # Row 2: loss, npv, ppv (The order you requested)
    plot(axes[1, 0], "train_loss", "val_loss", "Loss")
    plot(axes[1, 1], "train_npv", "val_npv", "NPV")
    plot(axes[1, 2], "train_ppv", "val_ppv", "PPV")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fold_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fold_dir / f"metrics_fold{fold_idx}.png", dpi=300, bbox_inches="tight")
    plt.close()


def train_fold(train_df, valid_df, exp_root, fold_idx, args, logger):
    logger.info(f"========== Training Fold {fold_idx} ==========")

    # Create fold-specific directory
    fold_dir = exp_root / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = fold_dir / "tensorboard"
    if tensorboard_dir.exists():
        shutil.rmtree(tensorboard_dir)
        logger.info(f"Removed previous TensorBoard directory for fold {fold_idx}.")
    writer = None if args.no_tensorboard else SummaryWriter(log_dir=tensorboard_dir)

    logger.info(
        f"Training set: {len(train_df)} (positive={train_df.label.sum()}, negative={len(train_df) - train_df.label.sum()})"
    )
    logger.info(
        f"Validation set: {len(valid_df)} (positive={valid_df.label.sum()}, negative={len(valid_df) - valid_df.label.sum()})"
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

    criterion = torch.nn.BCEWithLogitsLoss()

    # ---- Phase 1 ----
    logger.info(f"--- Phase 1: Feature Learning (lr=1e-4, epochs=8) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=config.WEIGHT_DECAY)
    phase1_epochs = 8
    best_auc, best_epoch = -1.0, -1

    disable_tqdm = not sys.stdout.isatty()

    for epoch in range(phase1_epochs):
        model.train()
        train_loss = 0.0
        y_train_true, y_train_raw = [], []

        for step, batch in enumerate(tqdm(train_loader, desc=f"Train [Phase1/F{fold_idx}]", disable=disable_tqdm), 1):
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

        avg_train_loss = train_loss / step
        y_train_true = torch.cat(y_train_true).detach().numpy()
        y_train_pred = torch.sigmoid(torch.cat(y_train_raw).detach()).numpy()
        train_auc, *_ = calculate_metrics(y_train_true, y_train_pred)

        # Validation
        model.eval()
        val_loss = 0.0
        y_true, y_raw = [], []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(valid_loader, desc="Val [Phase1]", disable=disable_tqdm), 1):
                ct = batch["image"].to(device)
                lbl = batch["label"].float().to(device)
                logits = model(ct).squeeze(1)
                loss = criterion(logits.view(-1), lbl.view(-1))
                val_loss += loss.item()
                y_true.append(lbl.cpu())
                y_raw.append(logits.cpu())

        avg_val_loss = val_loss / step
        y_true = torch.cat(y_true).detach().numpy()
        y_pred = torch.sigmoid(torch.cat(y_raw).detach()).numpy()
        val_auc, *_ = calculate_metrics(y_true, y_pred)

        logger.info(f"[Phase1][Epoch {epoch+1}/{phase1_epochs}] Val loss={avg_val_loss:.5f} | AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch + 1
            torch.save(model.state_dict(), fold_dir / f"phase1_best_fold{fold_idx}.pth")
            logger.info(f"New best Phase1 model (AUC={best_auc:.4f})")

    # ---- Phase 2 ----
    logger.info(f"--- Phase 2: Fine-tuning (lr=1e-6, patience={config.PATIENCE}) ---")
    model.load_state_dict(torch.load(fold_dir / f"phase1_best_fold{fold_idx}.pth"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=config.WEIGHT_DECAY)

    best_auc_phase2 = best_auc
    patience_counter = 0

    history = {k: [] for k in ["train_loss", "val_loss", "train_auc", "val_auc",
                               "train_sens", "val_sens", "train_spec", "val_spec",
                               "train_ppv", "val_ppv", "train_npv", "val_npv"]}
    history["best_epoch"] = 0

    disable_tqdm = not sys.stdout.isatty()

    for epoch in range(config.EPOCHS):
        logger.info(f"[Phase2][Fold {fold_idx}] Epoch {epoch+1}/{config.EPOCHS}")
        model.train()
        train_loss = 0.0
        y_train_true, y_train_raw = [], []

        for step, batch in enumerate(tqdm(train_loader, desc="Train [Phase2]", disable=disable_tqdm), 1):
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

        avg_train_loss = train_loss / step
        y_train_true = torch.cat(y_train_true).detach().numpy()
        y_train_pred = torch.sigmoid(torch.cat(y_train_raw).detach()).numpy()
        train_auc, train_sens, train_spec, train_ppv, train_npv = calculate_metrics(y_train_true, y_train_pred)

        # Validation
        model.eval()
        val_loss = 0.0
        y_true, y_raw = [], []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(valid_loader, desc="Val [Phase2]", disable=disable_tqdm), 1):
                ct = batch["image"].to(device)
                lbl = batch["label"].float().to(device)
                logits = model(ct).squeeze(1)
                loss = criterion(logits.view(-1), lbl.view(-1))
                val_loss += loss.item()
                y_true.append(lbl.cpu())
                y_raw.append(logits.cpu())

        avg_val_loss = val_loss / step
        y_true = torch.cat(y_true).detach().numpy()
        y_pred = torch.sigmoid(torch.cat(y_raw).detach()).numpy()
        val_auc, val_sens, val_spec, val_ppv, val_npv = calculate_metrics(y_true, y_pred)

        logger.info(f"[Epoch {epoch+1}] Val AUC={val_auc:.4f}")

        # Record history
        for k, v in zip(
            ["train_loss", "val_loss", "train_auc", "val_auc",
             "train_sens", "val_sens", "train_spec", "val_spec",
             "train_ppv", "val_ppv", "train_npv", "val_npv"],
            [avg_train_loss, avg_val_loss, train_auc, val_auc,
             train_sens, val_sens, train_spec, val_spec,
             train_ppv, val_ppv, train_npv, val_npv]):
            history[k].append(v)

        # Save best model
        if val_auc > best_auc_phase2:
            best_auc_phase2 = val_auc
            history["best_epoch"] = epoch + 1
            best_model_path = fold_dir / f"fold_{fold_idx}_best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            logger.info(f"New best model saved (AUC={best_auc_phase2:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # ---- Revert to best model before saving metrics ----
    best_model_path = fold_dir / f"fold_{fold_idx}_best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"Reverted model to best epoch ({history['best_epoch']}) with AUC={best_auc_phase2:.4f}")
    else:
        logger.warning("Best model file not found — continuing with final weights.")

    # ---- Save metrics and plot ----
    best_idx = history["best_epoch"] - 1 if history["best_epoch"] > 0 else -1
    results = {
        "fold": fold_idx,
        "best_auc": best_auc_phase2,
        "best_epoch": history["best_epoch"],
        "val_loss": history["val_loss"][best_idx] if best_idx >= 0 else None,
        "sensitivity": history["val_sens"][best_idx] if best_idx >= 0 else None,
        "specificity": history["val_spec"][best_idx] if best_idx >= 0 else None,
        "ppv": history["val_ppv"][best_idx] if best_idx >= 0 else None,
        "npv": history["val_npv"][best_idx] if best_idx >= 0 else None,
    }

    with open(fold_dir / f"results_fold{fold_idx}.json", "w") as f:
        json.dump(results, f, indent=2)

    save_fold_metrics_plot(fold_dir, history, fold_idx)
    logger.info(f"Saved metrics plot → {fold_dir}/metrics_fold{fold_idx}.png")

    if writer:
        writer.close()

    return results



# Cross-validation
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
                f.write("fold,best_auc,best_epoch,val_loss,sensitivity,specificity,ppv,npv\n")

            # Safely format each metric (handle None values)
            val_loss = f"{res['val_loss']:.6f}" if res["val_loss"] is not None else "NA"
            sens = f"{res['sensitivity']:.4f}" if res["sensitivity"] is not None else "NA"
            spec = f"{res['specificity']:.4f}" if res["specificity"] is not None else "NA"
            ppv = f"{res['ppv']:.4f}" if res["ppv"] is not None else "NA"
            npv = f"{res['npv']:.4f}" if res["npv"] is not None else "NA"

            f.write(
                f"{res['fold']},{res['best_auc']:.4f},{res['best_epoch']},{val_loss},{sens},{spec},{ppv},{npv}\n"
            )


    return all_results


# Entry point
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
