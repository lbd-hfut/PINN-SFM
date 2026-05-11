#!/usr/bin/env python3
"""
Run PINN-SfM training with synthetic data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinn_sfm.training import train_pinn_sfm
from pinn_sfm.visualization import print_metrics, plot_results
import torch


if __name__ == '__main__':
    model, K_pred, R_pred, t_pred, history, correspondences, gt_cameras, gt_points = \
        train_pinn_sfm(
            num_cameras=5,
            num_points=1000,
            num_epochs=1500,
            lr=5e-3,
            triang_views=2,
            lambda_gauge=20.0,
            lambda_reg=1e-4,
            image_size=(1000, 1000),
            rotation_deg=0.5,
            shared_K=True,
        )

    device = next(model.parameters()).device
    print_metrics(K_pred, R_pred, t_pred, correspondences, gt_cameras, device)

    all_indices = torch.arange(len(gt_cameras), device=device)
    plot_results(history, gt_cameras, gt_points, K_pred, R_pred, t_pred,
                 correspondences, all_indices, model, device, (1000, 1000))
