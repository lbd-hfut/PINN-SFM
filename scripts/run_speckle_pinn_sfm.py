#!/usr/bin/env python3
"""
Combined pipeline: generate speckle images → run PINN-SfM.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from skimage import io

from pinn_sfm.data.camera_array import build_grid_array
from pinn_sfm.data.speckle import generate_speckle_scene
from pinn_sfm.data.render import generate_image
from pinn_sfm.training import train_pinn_sfm
from pinn_sfm.visualization import print_metrics, plot_results


def extract_correspondences_from_speckle(R_arr, T_arr, scene_pts, scene_intensities,
                                          Fx, Fy, Cx, Cy, k1, k2, L_pixel, H_pixel,
                                          triang_views=2, seed=0):
    """
    Extract 2D-3D correspondences by projecting speckle points into each camera view.
    Simplified for demonstration — in practice, DIC-based matching should be used.
    """
    rng = np.random.default_rng(seed)
    num_views = R_arr.shape[2]
    correspondences = []
    gt_cameras = []
    gt_points = scene_pts

    K_gt = np.array([[Fx, 0, Cx], [0, Fy, Cy], [0, 0, 1]], dtype=np.float32)

    for i in range(num_views):
        gt_cameras.append((K_gt.copy(), R_arr[:, :, i].copy(), T_arr[:, i].copy()))

    # Sample a subset of points and build correspondences
    num_points = min(1000, scene_pts.shape[0])
    indices = rng.choice(scene_pts.shape[0], num_points, replace=False)

    for j in indices:
        Xw = scene_pts[j]
        obs_all = []
        for i, (K, R, t) in enumerate(gt_cameras):
            Xc = R @ Xw + t
            if Xc[2] <= 1:
                continue
            xh = K @ Xc
            u = xh[0] / xh[2]
            v = xh[1] / xh[2]
            if 0 < u < L_pixel and 0 < v < H_pixel:
                obs_all.append((i, torch.tensor([u, v], dtype=torch.float32)))

        if len(obs_all) >= triang_views + 1:
            correspondences.append({
                'point_idx': int(j),
                'triang_obs': obs_all[:triang_views],
                'reproj_obs': obs_all[triang_views:],
            })

    return correspondences, gt_cameras, gt_points[indices]


def main():
    print("=" * 50)
    print("Step 1: Generate speckle images")
    print("=" * 50)

    # ── Movement parameters ───────────────────────────────────
    M, N = 1, 1
    step_u, step_v = 0.5, 0.5

    # ── Camera parameters ─────────────────────────────────────
    L_pixel, H_pixel = 1000, 1000
    pixel_size = 3.45 / 1000
    f = 40
    Fx = Fy = f / pixel_size
    Cx = L_pixel / 2 + 0.5
    Cy = H_pixel / 2 + 0.5
    k1, k2 = 0, 0
    Tz = 500.0

    # Camera array
    R_arr, T_arr, _, _, _, _ = build_grid_array(
        M=M, N=N, step_u=step_u, step_v=step_v,
        Fx=Fx, Fy=Fy, Tz_world2refcam=Tz,
    )

    # Speckle scene
    speckle_img = io.imread('pinn_sfm/data/002.bmp')
    print("Generating speckle scene...")
    scene_pts, scene_intensities = generate_speckle_scene(
        speckle_img, num_points=1_000_000,
        x_range=(-50, 50), y_range=(-50, 50),
        surface_type='Plane', seed=42, verbose=False,
    )

    # Render
    img_index = 1
    for v in range(N + 1):
        for h in range(M + 1):
            print(f'  Rendering image {img_index}/{(M+1)*(N+1)}...')
            seed = v * (M + 1) + h + 100
            _, image = generate_image(
                Fx, Fy, Cx, Cy,
                R_arr[:, :, h, v], T_arr[:, 0, h, v],
                k1, k2, scene_pts.T, scene_intensities,
                noise_std=1, sigma=0.0001,
                image_width=L_pixel, image_height=H_pixel,
                noise_seed=seed,
            )
            io.imsave(f'RG_{img_index}.bmp', image)
            img_index += 1

    # ── Extract correspondences and run PINN-SfM ───────────────
    print("\n" + "=" * 50)
    print("Step 2: Extract correspondences")
    print("=" * 50)
    correspondences, gt_cameras, gt_points_sub = extract_correspondences_from_speckle(
        R_arr, T_arr, scene_pts, scene_intensities,
        Fx, Fy, Cx, Cy, k1, k2, L_pixel, H_pixel,
        triang_views=2,
    )
    print(f"Extracted {len(correspondences)} correspondences")

    print("\n" + "=" * 50)
    print("Step 3: Run PINN-SfM training")
    print("=" * 50)
    model, K_pred, R_pred, t_pred, history, _, _, _ = train_pinn_sfm(
        num_epochs=200, lr=5e-3,
        lambda_gauge=20.0, lambda_reg=1e-4,
        image_size=(L_pixel, H_pixel),
        model=None, correspondences=correspondences,
        gt_cameras=gt_cameras, gt_points=gt_points_sub,
    )

    device = next(model.parameters()).device
    print_metrics(K_pred, R_pred, t_pred, correspondences, gt_cameras, device)

    all_indices = torch.arange(len(gt_cameras), device=device)
    plot_results(history, gt_cameras, gt_points_sub, K_pred, R_pred, t_pred,
                 correspondences, all_indices, model, device, (L_pixel, H_pixel),
                 save_path='speckle_pinn_sfm_results.png')


if __name__ == '__main__':
    main()
