# train_Pulse3D.py
from __future__ import annotations

import logging
import random
import shutil
import os
import math
from datetime import datetime
from pathlib import Path
from torch.utils.data import BatchSampler, RandomSampler

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import KFold, GroupKFold
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataloader_cls import get_data_loader
from experiment_config_original import config
from models.Pulse3Dp import Pulse3D
from tensorboard_visuals import (
    log_conv_filters,
    log_feature_maps,
    register_activation_hook,
    log_roc_curve,
    log_prediction_samples,
    log_all_conv_filters,
)


# -----------------------------------------------------------------------------
# Hard-example-mining helper
# -----------------------------------------------------------------------------
def select_hard_examples(logits: torch.Tensor,
                         targets: torch.Tensor,
                         keep_ratio: float = 0.5) -> torch.Tensor:
    """
    Return indices of the hardest samples in the current batch, measured by
    |p − y|, where p = σ(logit) and y is the ground-truth label.

    Args
    ----
    logits : Tensor shape [B] or [B, 1]
    targets: Tensor shape [B] or [B, 1]
    keep_ratio: fraction of the batch to keep (e.g. 0.5 keeps the hardest 50 %)

    Returns
    -------
    hard_idx : 1-D LongTensor of selected indices
    """
    # flatten to [B]
    logits  = logits.view(-1)
    targets = targets.view(-1)
    diff    = (torch.sigmoid(logits) - targets).abs()      # difficulty score
    k       = max(1, int(keep_ratio * diff.numel()))
    hard_idx = torch.topk(diff, k=k, largest=True).indices
    return hard_idx


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, *, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Binary focal loss."""
    # Ensure inputs and targets have compatible shapes
    if inputs.dim() == 1 and targets.dim() == 2:
        # If inputs is [batch_size] and targets is [batch_size, 1]
        inputs = inputs.view(-1, 1)
    elif inputs.dim() == 2 and targets.dim() == 1:
        # If inputs is [batch_size, 1] and targets is [batch_size]
        targets = targets.view(-1, 1)
    
    bce = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()

def focal_loss_with_smoothing(inputs, targets, alpha=0.25, gamma=2.0, smoothing=0.1):
    # Ensure inputs and targets have compatible shapes
    if inputs.dim() == 1 and targets.dim() == 2:
        # If inputs is [batch_size] and targets is [batch_size, 1]
        inputs = inputs.view(-1, 1)
    elif inputs.dim() == 2 and targets.dim() == 1:
        # If inputs is [batch_size, 1] and targets is [batch_size]
        targets = targets.view(-1, 1)
        
    targets = targets * (1 - smoothing) + smoothing / 2
    bce = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


class AUCPairwiseLoss(torch.nn.Module):
    def __init__(self, delta: float = 0.25):
        super().__init__()
        self.delta = delta

    def forward(self, logits, targets):
        logits  = logits.view(-1)
        targets = targets.view(-1)

        pos = logits[targets == 1]
        neg = logits[targets == 0]

        # —— NEW: handle “mono-class” batch ——
        if pos.numel() == 0 or neg.numel() == 0:
            # return zero *but* keep the graph
            return logits.sum() * 0.0

        diffs  = pos[:, None] - neg[None, :]
        losses = torch.nn.functional.relu(self.delta - diffs)
        return losses.mean()


# class AUCPairwiseLoss(torch.nn.Module):
#     """
#     AUC Pairwise Loss for binary classification.
#     This loss approximates the Area Under the ROC Curve (AUC) by comparing
#     positive and negative pairs and penalizing incorrect orderings.
    
#     Reference:
#     "Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic"
#     by Yan et al.
#     """
#     def __init__(self, delta=1.0, sigmoid_output=False):
#         """
#         Args:
#             delta: Margin parameter for the hinge loss
#             sigmoid_output: Whether the model outputs sigmoid probabilities or raw logits
#         """
#         super().__init__()
#         self.delta = delta
#         self.sigmoid_output = sigmoid_output
    
#     def forward(self, logits, targets):
#         """
#         Args:
#             logits: Tensor of shape (batch_size, 1) or (batch_size,)
#             targets: Tensor of shape (batch_size, 1) or (batch_size,)
#         """
#         # Ensure inputs and targets have compatible shapes
#         if logits.dim() == 2 and logits.size(1) == 1:
#             logits = logits.squeeze(1)
#         if targets.dim() == 2 and targets.size(1) == 1:
#             targets = targets.squeeze(1)
            
#         # Convert logits to probabilities if needed
#         if not self.sigmoid_output:
#             probs = torch.sigmoid(logits)
#         else:
#             probs = logits
            
#         # Get positive and negative indices
#         pos_indices = (targets == 1).nonzero(as_tuple=True)[0]
#         neg_indices = (targets == 0).nonzero(as_tuple=True)[0]
        
#         # If either class has no samples, return 0 loss
#         if len(pos_indices) == 0 or len(neg_indices) == 0:
#             return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
#         # Get positive and negative scores
#         pos_scores = probs[pos_indices]
#         neg_scores = probs[neg_indices]
        
#         # Create all positive-negative pairs
#         pos_scores_expanded = pos_scores.unsqueeze(1)  # [P, 1]
#         neg_scores_expanded = neg_scores.unsqueeze(0)  # [1, N]
        
#         # Calculate pairwise differences: pos_scores - neg_scores
#         # Shape: [P, N]
#         differences = pos_scores_expanded - neg_scores_expanded
        
#         # Apply hinge loss with margin delta
#         # max(0, delta - (pos_score - neg_score))
#         losses = torch.relu(self.delta - differences)
        
#         # Average over all pairs
#         return losses.mean()


class LossWrapper(torch.nn.Module):
    """
    A wrapper class for different loss functions that ensures consistent input/output shapes.
    This allows for easy switching between different loss functions.
    """
    def __init__(self, loss_type='bce', **kwargs):
        """
        Args:
            loss_type: Type of loss function to use ('bce', 'focal', 'focal_smooth', 'auc_pairwise')
            **kwargs: Additional arguments for the specific loss function
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'bce':
            self.loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        elif loss_type == 'focal':
            self.loss_fn = lambda x, y: focal_loss(x, y, **kwargs)
        elif loss_type == 'focal_smooth':
            self.loss_fn = lambda x, y: focal_loss_with_smoothing(x, y, **kwargs)
        elif loss_type == 'auc_pairwise':
            self.loss_fn = AUCPairwiseLoss(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, logits, targets):
        """
        Ensure consistent shape handling for all loss functions.
        
        Args:
            logits: Tensor of shape (batch_size, num_classes) or (batch_size,)
            targets: Tensor of shape (batch_size, num_classes) or (batch_size,)
        """
        # Handle shape consistency
        if logits.dim() == 1 and targets.dim() == 2:
            # If logits is [batch_size] and targets is [batch_size, 1]
            logits = logits.view(-1, 1)
        elif logits.dim() == 2 and targets.dim() == 1:
            # If logits is [batch_size, 1] and targets is [batch_size]
            targets = targets.view(-1, 1)
        
        return self.loss_fn(logits, targets)


def make_weights_for_balanced_classes(labels: np.ndarray) -> torch.DoubleTensor:
    n = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    # classic inverse-frequency weights → each class sums to the same total weight
    frac = {cls: n / (2 * c) for cls, c in zip(*np.unique(labels, return_counts=True))}
    return torch.DoubleTensor([frac[l] for l in labels])

def create_kfold_splits(csv_path, n_splits=5, shuffle=True, random_state=None):
    """Create k-fold splits from a CSV file, ensuring patients don't leak between folds."""
    df = pd.read_csv(csv_path)
    
    # Use GroupKFold to ensure patients don't appear in both train and validation sets
    gkf = GroupKFold(n_splits=n_splits)
    
    # Create a list to store the fold dataframes
    fold_dfs = []
    
    # Use PatientID as the grouping variable
    for train_idx, val_idx in gkf.split(df, groups=df.PatientID):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        val_fold = df.iloc[val_idx].reset_index(drop=True)
        
        # Verify no patient leakage between train and validation sets
        train_patients = set(train_fold.PatientID)
        val_patients = set(val_fold.PatientID)
        common_patients = train_patients.intersection(val_patients)
        
        if common_patients:
            logging.warning(f"Patient leakage detected! {len(common_patients)} patients appear in both train and validation sets.")
        else:
            logging.info(f"No patient leakage detected between train and validation sets.")
        
        # Log distribution statistics
        logging.info(f"Train set: {len(train_fold)} samples from {len(train_patients)} patients")
        logging.info(f"Validation set: {len(val_fold)} samples from {len(val_patients)} patients")
        
        fold_dfs.append((train_fold, val_fold))
    
    return fold_dfs


# -----------------------------------------------------------------------------
# Core training routine
# -----------------------------------------------------------------------------

def train_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    exp_save_root: Path,
    fold_idx: int,
    cls_ckpt: str | Path = None,
):
    """Train a NoduleMalignancyClassifier for a single fold."""
    # Create fold-specific directory
    fold_dir = exp_save_root / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------- logging ---------------------------
    logging.info(f"Training fold {fold_idx}")
    
    # Clean up previous TensorBoard logs for this fold
    tensorboard_dir = fold_dir / "tensorboard"
    if tensorboard_dir.exists():
        shutil.rmtree(tensorboard_dir)
        logging.info("Cleaned up previous TensorBoard logs from %s", tensorboard_dir)

    writer = SummaryWriter(log_dir=tensorboard_dir)

    logging.info(
        "Training – malignant: %d | benign: %d",
        train_df.label.sum(),
        len(train_df) - train_df.label.sum(),
    )
    logging.info(
        "Validation – malignant: %d | benign: %d",
        valid_df.label.sum(),
        len(valid_df) - valid_df.label.sum(),
    )

    # -------------------- data loaders ----------------------

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=make_weights_for_balanced_classes(train_df.label.values),
        num_samples=len(train_df),
        replacement=True
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

    # -------------------- model setup ----------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Pulse3D(
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Using fixed learning rate (no scheduler)

    # Create loss function using the wrapper
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = LossWrapper(loss_type='auc_pairwise', delta=1.0)
    # Alternatively, you can use other loss functions:
    # criterion = LossWrapper(loss_type='bce')
    # criterion = LossWrapper(loss_type='focal', alpha=0.25, gamma=2.0)
    # criterion = LossWrapper(loss_type='focal_smooth', alpha=0.25, gamma=2.0, smoothing=0.1)

    # -------------------- training loop ---------------------
    # Phase 1: Train for fixed 17 epochs with LR=1e-4
    phase1_epochs = 20
    best_auc, best_epoch = -1.0, -1
    activation_store = {}

    logging.info("Starting Phase 1: Training for %d epochs with LR=%.6f", phase1_epochs, config.LEARNING_RATE)

    for epoch in range(phase1_epochs):
        logging.info("\n===== Fold %d | Phase 1 | Epoch %d / %d =====", fold_idx, epoch + 1, phase1_epochs)
        
        # Training phase
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc="Train", leave=False), 1):

            optimizer.zero_grad(set_to_none=True)

            ct = batch["image"].to(device)
            lbl = batch["label"].float().to(device)
            age = batch["age"].float().to(device)
            gender = batch["gender"].float().to(device)
            meta = torch.cat((age.unsqueeze(1), gender.unsqueeze(1)), dim=1)

            # Forward pass
            cls_logit  = model(ct)
            logits = cls_logit.squeeze(1)
            # Compute loss and backprop
            loss = criterion(cls_logit, lbl)
            loss.backward()
            
            optimizer.step()

            train_loss += loss.item()
            if step % 100 == 0:
                logging.debug("Step %d | Train loss %.5f", step, loss.item())
                writer.add_scalar('Loss/train_step', loss.item(), epoch * len(train_loader) + step)

        avg_train_loss = train_loss / step
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Learning_rate', config.LEARNING_RATE, epoch)
        logging.info("Fold %d | Epoch %d | Avg train loss %.6f | LR %.6f", fold_idx, epoch + 1, avg_train_loss, config.LEARNING_RATE)

        # Validation phase
        model.eval()
        val_loss = 0.0
        y_true, y_raw = [], []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(valid_loader, desc="Val", leave=False), 1):
                ct = batch["image"].to(device)
                lbl = batch["label"].float().to(device)
                age = batch["age"].float().to(device)
                gender = batch["gender"].float().to(device)
                meta = torch.cat((age.unsqueeze(1), gender.unsqueeze(1)), dim=1)

                cls_logit = model(ct)
                logits = cls_logit.squeeze(1)
                loss = criterion(cls_logit, lbl)

                val_loss += loss.item()
                # Ensure tensors have proper dimensions before appending
                y_true.append(lbl.view(-1).cpu())
                y_raw.append(logits.view(-1).cpu())

        avg_val_loss = val_loss / step
        # Ensure proper dimensions before concatenation
        y_true = torch.cat([t.view(-1) for t in y_true]).numpy()
        y_pred = torch.sigmoid(torch.cat([t.view(-1) for t in y_raw])).numpy()
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        auc = metrics.auc(fpr, tpr)
        

        # Calculate additional metrics
        y_pred_binary = (y_pred > 0.5).astype(int)  # Apply threshold of 0.5
        accuracy = metrics.accuracy_score(y_true, y_pred_binary)
        precision = metrics.precision_score(y_true, y_pred_binary, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Create confusion matrix
        conf_matrix = metrics.confusion_matrix(y_true, y_pred_binary)
        
        # Log metrics
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Metrics/AUC', auc, epoch)
        writer.add_scalar('Metrics/Accuracy', accuracy, epoch)
        writer.add_scalar('Metrics/Precision', precision, epoch)
        writer.add_scalar('Metrics/Recall', recall, epoch)
        writer.add_scalar('Metrics/F1', f1, epoch)

        log_prediction_samples(model, valid_loader, writer, epoch, device)
        log_roc_curve(writer, fpr, tpr, epoch)
        
        logging.info("Fold %d | Epoch %d | Val loss %.6f | AUC %.4f | Acc %.4f | Prec %.4f | Rec %.4f | F1 %.4f", 
                    fold_idx, epoch + 1, avg_val_loss, auc, accuracy, precision, recall, f1)
        logging.info("Confusion Matrix:\n %s", conf_matrix)
        logging.info("Best so far – AUC %.4f at epoch %d", best_auc, best_epoch)

        # Checkpoint if better
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch + 1
            torch.save(model.state_dict(), fold_dir / "best_metric_cls_model.pth")
            np.save(fold_dir / "config.npy", {
                "fold": fold_idx,
                "config": config,
                "best_auc": best_auc,
                "best_accuracy": accuracy,
                "best_precision": precision,
                "best_recall": recall,
                "best_f1": f1,
                "epoch": best_epoch,
                "confusion_matrix": conf_matrix.tolist(),
            })
            logging.info("New best model saved (AUC %.4f, Acc %.4f, Prec %.4f, Rec %.4f, F1 %.4f at epoch %d)", 
                        best_auc, accuracy, precision, recall, f1, best_epoch)

    logging.info("Phase 1 complete – best AUC %.4f at epoch %d", best_auc, best_epoch)

    # Phase 2: Load best model from Phase 1 and continue with LR=1e-6 until patience
    logging.info("Starting Phase 2: Continuing from best Phase 1 model with LR=1e-6")
    model.load_state_dict(torch.load(fold_dir / "best_metric_cls_model.pth"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,
        weight_decay=config.WEIGHT_DECAY
    )
    epochs_no_improve = 0
    start_epoch = phase1_epochs

    for epoch in range(start_epoch, config.EPOCHS):
        if epochs_no_improve > config.PATIENCE:
            logging.info("Early stop – no AUC improvement for %d epochs", config.PATIENCE)
            break

        logging.info("\n===== Fold %d | Phase 2 | Epoch %d / %d =====", fold_idx, epoch + 1, config.EPOCHS)
        
        # Training phase
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc="Train", leave=False), 1):

            optimizer.zero_grad(set_to_none=True)

            ct = batch["image"].to(device)
            lbl = batch["label"].float().to(device)
            age = batch["age"].float().to(device)
            gender = batch["gender"].float().to(device)
            meta = torch.cat((age.unsqueeze(1), gender.unsqueeze(1)), dim=1)

            # Forward pass
            cls_logit  = model(ct)
            logits = cls_logit.squeeze(1)
            # Compute loss and backprop
            loss = criterion(cls_logit, lbl)
            loss.backward()
            
            optimizer.step()

            train_loss += loss.item()
            if step % 100 == 0:
                logging.debug("Step %d | Train loss %.5f", step, loss.item())
                writer.add_scalar('Loss/train_step', loss.item(), epoch * len(train_loader) + step)

        avg_train_loss = train_loss / step
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Learning_rate', 1e-6, epoch)
        logging.info("Fold %d | Epoch %d | Avg train loss %.6f | LR %.6f", fold_idx, epoch + 1, avg_train_loss, 1e-6)

        # Validation phase
        model.eval()
        val_loss = 0.0
        y_true, y_raw = [], []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(valid_loader, desc="Val", leave=False), 1):
                ct = batch["image"].to(device)
                lbl = batch["label"].float().to(device)
                age = batch["age"].float().to(device)
                gender = batch["gender"].float().to(device)
                meta = torch.cat((age.unsqueeze(1), gender.unsqueeze(1)), dim=1)

                cls_logit = model(ct)
                logits = cls_logit.squeeze(1)
                loss = criterion(cls_logit, lbl)

                val_loss += loss.item()
                # Ensure tensors have proper dimensions before appending
                y_true.append(lbl.view(-1).cpu())
                y_raw.append(logits.view(-1).cpu())

        avg_val_loss = val_loss / step
        # Ensure proper dimensions before concatenation
        y_true = torch.cat([t.view(-1) for t in y_true]).numpy()
        y_pred = torch.sigmoid(torch.cat([t.view(-1) for t in y_raw])).numpy()
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        auc = metrics.auc(fpr, tpr)
        

        # Calculate additional metrics
        y_pred_binary = (y_pred > 0.5).astype(int)  # Apply threshold of 0.5
        accuracy = metrics.accuracy_score(y_true, y_pred_binary)
        precision = metrics.precision_score(y_true, y_pred_binary, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Create confusion matrix
        conf_matrix = metrics.confusion_matrix(y_true, y_pred_binary)
        
        # Log metrics
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Metrics/AUC', auc, epoch)
        writer.add_scalar('Metrics/Accuracy', accuracy, epoch)
        writer.add_scalar('Metrics/Precision', precision, epoch)
        writer.add_scalar('Metrics/Recall', recall, epoch)
        writer.add_scalar('Metrics/F1', f1, epoch)

        log_prediction_samples(model, valid_loader, writer, epoch, device)
        log_roc_curve(writer, fpr, tpr, epoch)
        
        logging.info("Fold %d | Epoch %d | Val loss %.6f | AUC %.4f | Acc %.4f | Prec %.4f | Rec %.4f | F1 %.4f", 
                    fold_idx, epoch + 1, avg_val_loss, auc, accuracy, precision, recall, f1)
        logging.info("Confusion Matrix:\n %s", conf_matrix)
        logging.info("Best so far – AUC %.4f at epoch %d", best_auc, best_epoch)

        # Checkpoint
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), fold_dir / "best_metric_cls_model.pth")
            np.save(fold_dir / "config.npy", {
                "fold": fold_idx,
                "config": config,
                "best_auc": best_auc,
                "best_accuracy": accuracy,
                "best_precision": precision,
                "best_recall": recall,
                "best_f1": f1,
                "epoch": best_epoch,
                "confusion_matrix": conf_matrix.tolist(),
            })
            logging.info("New best model saved (AUC %.4f, Acc %.4f, Prec %.4f, Rec %.4f, F1 %.4f at epoch %d)", 
                        best_auc, accuracy, precision, recall, f1, best_epoch)
        else:
            epochs_no_improve += 1

    logging.info("Fold %d training complete – best AUC %.4f at epoch %d", fold_idx, best_auc, best_epoch)
    writer.close()
    
    # Load the best model to calculate final metrics
    best_model = Pulse3D().to(device)
    best_model.load_state_dict(torch.load(fold_dir / "best_metric_cls_model.pth"))
    best_model.eval()
    
    # Calculate final metrics on validation set
    y_true, y_raw = [], []
    all_ages, all_genders = [], []
    with torch.no_grad():
        for batch in valid_loader:
            ct = batch["image"].to(device)
            lbl = batch["label"].float().to(device)
            age = batch["age"].float().to(device)
            gender = batch["gender"].float().to(device)
            meta = torch.cat((age.unsqueeze(1), gender.unsqueeze(1)), dim=1)
            
            cls_logit = best_model(ct)
            logits = cls_logit.squeeze(1)
            # Ensure tensors have proper dimensions
            y_true.append(lbl.view(-1).cpu())
            y_raw.append(logits.view(-1).cpu())
            all_ages.extend(age.cpu().numpy())
            all_genders.extend(gender.cpu().numpy())
    
    # Ensure proper dimensions before concatenation
    y_true = torch.cat([t.view(-1) for t in y_true]).numpy()
    y_pred = torch.sigmoid(torch.cat([t.view(-1) for t in y_raw])).numpy()
    y_pred_binary = (y_pred > 0.5).astype(int)
    all_ages = np.array(all_ages)
    all_genders = np.array(all_genders)
    
    # Calculate metrics with best model
    best_auc = metrics.roc_auc_score(y_true, y_pred)
    best_accuracy = metrics.accuracy_score(y_true, y_pred_binary)
    best_precision = metrics.precision_score(y_true, y_pred_binary, zero_division=0)
    best_recall = metrics.recall_score(y_true, y_pred_binary, zero_division=0)
    best_f1 = metrics.f1_score(y_true, y_pred_binary, zero_division=0)
    
    # Calculate demographic metrics
    male_indices = np.array(all_genders) < 0.5
    female_indices = np.array(all_genders) >= 0.5
    young_indices = np.array(all_ages) < 0.6  # Age < 60
    old_indices = np.array(all_ages) >= 0.6   # Age >= 60
    
    # Gender analysis
    demographic_metrics = {}
    if np.sum(male_indices) > 0 and np.sum(y_true[male_indices]) > 0 and np.sum(y_true[male_indices]) < len(y_true[male_indices]):
        male_auc = metrics.roc_auc_score(y_true[male_indices], y_pred[male_indices])
        logging.info(f"Male AUC: {male_auc:.4f}")
        demographic_metrics['male_auc'] = male_auc
    
    if np.sum(female_indices) > 0 and np.sum(y_true[female_indices]) > 0 and np.sum(y_true[female_indices]) < len(y_true[female_indices]):
        female_auc = metrics.roc_auc_score(y_true[female_indices], y_pred[female_indices])
        logging.info(f"Female AUC: {female_auc:.4f}")
        demographic_metrics['female_auc'] = female_auc
    
    # Age analysis
    if np.sum(young_indices) > 0 and np.sum(y_true[young_indices]) > 0 and np.sum(y_true[young_indices]) < len(y_true[young_indices]):
        young_auc = metrics.roc_auc_score(y_true[young_indices], y_pred[young_indices])
        logging.info(f"Young (<60) AUC: {young_auc:.4f}")
        demographic_metrics['young_auc'] = young_auc
    
    if np.sum(old_indices) > 0 and np.sum(y_true[old_indices]) > 0 and np.sum(y_true[old_indices]) < len(y_true[old_indices]):
        old_auc = metrics.roc_auc_score(y_true[old_indices], y_pred[old_indices])
        logging.info(f"Old (>=60) AUC: {old_auc:.4f}")
        demographic_metrics['old_auc'] = old_auc
    
    # Log final metrics for this fold
    logging.info(f"Fold {fold_idx} final metrics with best model:")
    logging.info(f"AUC: {best_auc:.4f}, Accuracy: {best_accuracy:.4f}")
    logging.info(f"Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
    
    # Return all metrics
    return {
        'auc': best_auc,
        'accuracy': best_accuracy,
        'precision': best_precision,
        'recall': best_recall,
        'f1': best_f1,
        'demographics': demographic_metrics
    }


def train_cross_validation(
    csv_path: Path | str,
    exp_save_root: Path,
    n_folds: int = 5,
    cls_ckpt: str | Path = None,
    only_fold: int = -1,
):
    """Train with k-fold cross-validation."""
    # -------------------- reproducibility --------------------
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # -------------------- logging ---------------------------
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)s][%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Create experiment directory
    exp_save_root.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file
    log_file = exp_save_root / "cross_validation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("[%(levelname)s][%(asctime)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"Starting {n_folds}-fold cross-validation")
    logging.info(f"Data CSV: {csv_path}")
    logging.info(f"Checkpoint: {cls_ckpt}")
    
    # Create the folds
    fold_data = create_kfold_splits(csv_path, n_splits=n_folds, random_state=config.SEED)
    
    # Analyze patient distribution across folds
    all_df = pd.read_csv(csv_path)
    total_patients = set(all_df.PatientID)
    logging.info(f"Total dataset: {len(all_df)} samples from {len(total_patients)} unique patients")
    
    # Check that each patient appears in exactly one validation fold
    patient_to_val_fold = {}
    for fold_idx, (_, val_df) in enumerate(fold_data):
        for patient_id in set(val_df.PatientID):  # Use set to prevent duplicate entries
            if patient_id in patient_to_val_fold and patient_to_val_fold[patient_id] != fold_idx:
                logging.error(f"Patient {patient_id} appears in validation sets of both fold {patient_to_val_fold[patient_id]} and fold {fold_idx}!")
            else:
                patient_to_val_fold[patient_id] = fold_idx
    
    # Verify all patients are covered
    covered_patients = set(patient_to_val_fold.keys())
    if covered_patients != total_patients:
        missing = total_patients - covered_patients
        if missing:
            logging.warning(f"{len(missing)} patients are not included in any validation fold!")
    
    # Train each fold
    fold_metrics = []
    for fold_idx, (train_df, valid_df) in enumerate(fold_data):
        if only_fold != -1 and fold_idx != only_fold:
            continue
        fold_result = train_fold(train_df, valid_df, exp_save_root, fold_idx, cls_ckpt)
        fold_metrics.append(fold_result)
    
    # Summarize results
    mean_metrics = {}
    std_metrics = {}
    
    # Calculate mean and std for each metric
    for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
        values = [fold[metric] for fold in fold_metrics]
        mean_metrics[metric] = np.mean(values)
        std_metrics[metric] = np.std(values)
    
    logging.info("\n===== Cross-Validation Results =====")
    for fold_idx, metrics_dict in enumerate(fold_metrics):
        logging.info(f"Fold {fold_idx}: AUC = {metrics_dict['auc']:.4f}, Acc = {metrics_dict['accuracy']:.4f}, "
                    f"Prec = {metrics_dict['precision']:.4f}, Rec = {metrics_dict['recall']:.4f}, F1 = {metrics_dict['f1']:.4f}")
    
    logging.info(f"Mean AUC: {mean_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
    logging.info(f"Mean Accuracy: {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    logging.info(f"Mean Precision: {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    logging.info(f"Mean Recall: {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    logging.info(f"Mean F1: {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    
    # Save overall results
    results = {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "config": config,
    }
    np.save(exp_save_root / "cv_results.npy", results)
    
    return mean_metrics, std_metrics


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    exp_name = f"{config.EXPERIMENT_NAME}-Pulse3D-{datetime.today().strftime('%Y%m%d')}"
    exp_root = config.EXPERIMENT_DIR / exp_name
    
    # For cross-validation, we use a single CSV file containing all data
    csv_path = config.CSV_DIR / "all_data.csv"
    
    # If all_data.csv doesn't exist, create it by combining train, valid, and test CSVs
    if not csv_path.exists():
        train_df = pd.read_csv(config.CSV_DIR_TRAIN)
        valid_df = pd.read_csv(config.CSV_DIR_VALID)
        test_df = pd.read_csv(config.CSV_DIR_TEST)
        all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        all_df.to_csv(csv_path, index=False)
        logging.info(f"Created combined dataset at {csv_path}")
    
    # Using GroupKFold for patient-level splitting to prevent data leakage
    # Each patient's data will only appear in one fold's validation set
    mean_metrics, std_metrics = train_cross_validation(
        csv_path=csv_path,
        exp_save_root=exp_root,
        n_folds=5,
        cls_ckpt=None
    )
    
    # Print final summary
    logging.info("\n===== Final Cross-Validation Results =====")
    logging.info(f"Mean AUC: {mean_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
    logging.info(f"Mean Accuracy: {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    logging.info(f"Mean Precision: {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    logging.info(f"Mean Recall: {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    logging.info(f"Mean F1: {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")