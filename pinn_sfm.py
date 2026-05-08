#!/usr/bin/env python3
"""
Legacy entry point — delegates to pinn_sfm.training.
Use: python -m pinn_sfm.training  or  scripts/run_pinn_sfm.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinn_sfm.training import train_pinn_sfm
from pinn_sfm.visualization import print_metrics, plot_results
import torch

if __name__ == '__main__':
    model, K_pred, R_pred, t_pred, history, correspondences, gt_cameras, gt_points = \
        train_pinn_sfm(
            num_cameras=3, num_points=1000, num_epochs=200,
            lr=5e-3, triang_views=2, lambda_gauge=20.0,
            lambda_reg=1e-4, image_size=(1000, 1000),
        )
    device = next(model.parameters()).device
    print_metrics(K_pred, R_pred, t_pred, correspondences, gt_cameras, device)
    all_indices = torch.arange(len(gt_cameras), device=device)
    plot_results(history, gt_cameras, gt_points, K_pred, R_pred, t_pred,
                 correspondences, all_indices, model, device, (1000, 1000))
