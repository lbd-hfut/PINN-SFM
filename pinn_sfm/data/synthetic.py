"""
Simplified synthetic data generation (original dataGeneration.py).
Generates correspondences directly without speckle rendering.
"""

import torch
import numpy as np


def generate_synthetic_data_realistic(
    num_cameras=5,
    num_points=1000,
    image_size=(1000, 1000),
    pixel_size=3.45e-3,
    f_mm=40.0,
    work_dist=500.0,
    noise_std=0.0,
    triang_views=2,
    seed=0,
):
    """
    Generate synthetic SfM correspondences based on a realistic DIC camera model.
    Cameras are placed on a line with step-pixel separation.
    Points lie on a plane perpendicular to the optical axis.
    """
    rng = np.random.default_rng(seed)
    W, H = image_size

    fx = f_mm / pixel_size
    fy = f_mm / pixel_size
    cx = W / 2
    cy = H / 2

    K_gt = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)

    step_px = 20
    Tx_list = (np.arange(num_cameras) * step_px * work_dist) / fx
    Ty_list = np.zeros_like(Tx_list)

    gt_cameras = []
    for i in range(num_cameras):
        R = np.eye(3, dtype=np.float32)
        t = np.array([-Tx_list[i], -Ty_list[i], work_dist], dtype=np.float32)
        gt_cameras.append((K_gt.copy(), R, t))

    L_view = work_dist * (W * pixel_size) / f_mm
    H_view = work_dist * (H * pixel_size) / f_mm

    X = rng.uniform(-L_view/4, L_view/4, num_points)
    Y = rng.uniform(-H_view/4, H_view/4, num_points)
    Z = np.zeros_like(X)
    gt_points = np.stack([X, Y, Z], axis=1).astype(np.float32)

    correspondences = []
    for j, Xw in enumerate(gt_points):
        obs_all = []
        for i, (K, R, t) in enumerate(gt_cameras):
            Xc = R @ Xw + t
            if Xc[2] <= 1:
                continue
            xh = K @ Xc
            u = xh[0] / xh[2]
            v = xh[1] / xh[2]
            if 0 < u < W and 0 < v < H:
                u_n = u + rng.normal(0, noise_std)
                v_n = v + rng.normal(0, noise_std)
                obs_all.append((i, torch.tensor([u_n, v_n], dtype=torch.float32)))

        if len(obs_all) >= triang_views + 1:
            correspondences.append({
                'point_idx': j,
                'triang_obs': obs_all[:triang_views],
                'reproj_obs': obs_all[triang_views:],
            })

    gt_cameras_aligned, gt_points_aligned = convert_to_cam0_world(gt_cameras, gt_points)
    return correspondences, gt_cameras_aligned, gt_points_aligned


def convert_to_cam0_world(gt_cameras, gt_points):
    """Re-align so camera-0 is at the world origin."""
    K0, R0, t0 = gt_cameras[0]
    R0_inv = R0.T

    gt_points_new = (R0 @ gt_points.T + t0.reshape(3, 1)).T
    gt_cameras_new = []
    for K, R, t in gt_cameras:
        R_new = R @ R0_inv
        t_new = t - R @ R0_inv @ t0
        gt_cameras_new.append((K.copy(), R_new, t_new))

    return gt_cameras_new, gt_points_new
