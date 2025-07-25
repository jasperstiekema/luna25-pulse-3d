"""
Inference script for predicting malignancy of lung nodules
"""
import numpy as np
import dataloader
import torch
import torch.nn as nn
from torchvision import models
from models.model_3d import I3D
from models.Pulse3D import Pulse3D
from models.model_2d import ResNet18
import os
import math
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

# define processor
class MalignancyProcessor:
    """
    Inference with k-fold ensemble of models for lung nodule malignancy prediction.
    Supports 2D or 3D classification (no segmentation).
    """

    def __init__(self, mode="3D", model_root="/opt/app/resources/", 
                 model_name="LUNA25-pulse-Pulse3D-20250720", k_folds=5,
                 size_px=64, size_mm=50, suppress_logs=False):
        self.mode = mode.upper()
        self.model_root = model_root
        self.model_name = model_name
        self.k_folds = k_folds
        self.size_px = size_px
        self.size_mm = size_mm
        self.suppress_logs = suppress_logs
        if not self.suppress_logs:
            logging.info("Initializing the deep learning system")
        # Paths to each fold directory
        self.fold_paths = [
            os.path.join(self.model_root, self.model_name, f"fold_{i}")
            for i in range(self.k_folds)
        ]

        if self.mode == "2D":
            self._build_2d_model()
        else:
            self._build_3d_model()

    def _build_2d_model(self):
        self.base_model = ResNet18(weights=None).cuda().eval()

    def _build_3d_model(self):
        # only classification model, no segmentation
        self.base_cls = Pulse3D().cuda().eval()

    def define_inputs(self, image, header, coords):
        self.image = image
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):
        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header['origin'],
            srcWorldMatrix=self.header['transform'],
            srcVoxelSpacing=self.header['spacing'],
            output_shape=output_shape,
            voxel_spacing=(self.size_mm/self.size_px,)*3,
            coord_space_world=True,
            mode=mode,
        )
        patch = patch.astype(np.float32)
        patch = dataloader.clip_and_scale(patch)
        return patch

    def _predict_fold(self, fold_path):
        """
        Load classification weights from fold_path and predict probabilities.
        """
        nodules = []
        if self.mode == "2D":
            output_shape = [1, self.size_px, self.size_px]
        else:
            output_shape = [self.size_px, self.size_px, self.size_px]

        for coord in self.coords:
            patch = self.extract_patch(coord, output_shape, self.mode)
            nodules.append(patch)
        nodules = np.stack(nodules)
        tensor = torch.from_numpy(nodules).cuda()

        if self.mode == "2D":
            model = self.base_model
            # load checkpoint for classification
            ckpt = torch.load(os.path.join(fold_path, 'best_metric_cls_model.pth'))
            model.load_state_dict(ckpt)
            with torch.no_grad():
                logits = model(tensor)
        else:
            cls_model = self.base_cls
            model_path = os.path.join(fold_path, 'best_metric_cls_model.pth')
            # print(f"Use model {model_path}")
            ckpt = torch.load(model_path)
            cls_model.load_state_dict(ckpt)
            with torch.no_grad():
                logits = cls_model(tensor)

        logits = logits.cpu().numpy()
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return probs.squeeze()

    def predict(self):
        """
        Run inference across all folds and return mean probability and per-fold probabilities.
        """
        all_probs = []
        for fold_path in self.fold_paths:
            # logging.info(f"Predicting with fold: {os.path.basename(fold_path)}")
            probs = self._predict_fold(fold_path)
            all_probs.append(probs)

        # shape: (k_folds, n_samples)
        ensemble = np.stack(all_probs, axis=0)
        mean_prob = ensemble.mean(axis=0)
        return mean_prob, ensemble