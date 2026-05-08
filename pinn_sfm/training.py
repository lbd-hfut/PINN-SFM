"""
Training loop for PINN-SfM.
"""

import torch
import numpy as np

from .models import CameraNetwork
from .losses import compute_reprojection_loss, gauge_loss
from .data.synthetic import generate_synthetic_data_realistic


def train_pinn_sfm(
    num_cameras:   int   = 5,
    num_points:    int   = 1000,
    num_epochs:    int   = 3000,
    lr:            float = 5e-4,
    triang_views:  int   = 2,
    lambda_gauge:  float = 20.0,
    lambda_reg:    float = 1e-4,
    image_size:    tuple = (1000, 1000),
    model:         CameraNetwork = None,
    correspondences: list = None,
    gt_cameras:    list = None,
    gt_points:     np.ndarray = None,
):
    """
    Full PINN-SfM training loop.

    If `model`, `correspondences`, `gt_cameras`, `gt_points` are provided,
    they are used directly (external data). Otherwise, synthetic data is generated.

    Each epoch:
      1. Forward pass → all camera (K, R, t)
      2. Triangulate 3D points from multi-view observations
      3. Reproject onto held-out views for physical loss
      4. Backpropagate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Data ───────────────────────────────────────────────────
    if correspondences is None:
        print("[Data] Generating synthetic scene...")
        correspondences, gt_cameras, gt_points = generate_synthetic_data_realistic(
            num_cameras=num_cameras, num_points=num_points, image_size=image_size,
            pixel_size=3.45e-3, f_mm=40.0, work_dist=500.0,
            noise_std=0.0, triang_views=triang_views, seed=0,
        )
        print(f"       Cameras: {num_cameras}, Tracks: {len(correspondences)} / {num_points}")
    else:
        num_cameras = len(gt_cameras)

    # ── Model ──────────────────────────────────────────────────
    if model is None:
        model = CameraNetwork(num_cameras=num_cameras, hidden_dim=256, num_freqs=8).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5)

    all_indices = torch.arange(num_cameras, device=device)

    # ── Train ──────────────────────────────────────────────────
    history = {'total': [], 'reproj': [], 'gauge': []}
    print(f"\n{'Epoch':>6}  {'Total':>10}  {'Reproj':>10}  {'Gauge':>10}  {'LR':>10}")
    print('─' * 56)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        K_all, R_all, t_all = model(all_indices, image_size)

        loss_reproj = compute_reprojection_loss(
            K_all, R_all, t_all, correspondences, device)
        loss_gauge = gauge_loss(R_all[0], t_all[0])
        loss_reg = (t_all ** 2).mean()

        loss = loss_reproj + lambda_gauge * loss_gauge + lambda_reg * loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        history['total'].append(loss.item())
        history['reproj'].append(loss_reproj.item())
        history['gauge'].append(loss_gauge.item())

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"{epoch:>6}  {loss.item():>10.3f}  {loss_reproj.item():>10.3f}"
                  f"  {loss_gauge.item():>10.4f}  {lr_now:>10.6f}")

    # ── Evaluate ───────────────────────────────────────────────
    print("\n[Evaluate] Computing final metrics...")
    model.eval()
    with torch.no_grad():
        K_pred, R_pred, t_pred = model(all_indices, image_size)

    return model, K_pred, R_pred, t_pred, history, correspondences, gt_cameras, gt_points


def collect_reproj_errors(K_all, R_all, t_all, correspondences, device):
    """Collect reprojection errors for all correspondences."""
    from .geometry import build_proj_matrix, triangulate_dlt, reproject
    errors = []
    for corr in correspondences:
        t_obs = corr['triang_obs']
        Ps = torch.stack([
            build_proj_matrix(K_all[ci], R_all[ci], t_all[ci])
            for ci, _ in t_obs])
        pts2d = torch.stack([x2d.to(device) for _, x2d in t_obs])
        X_j = triangulate_dlt(Ps, pts2d)
        for ci, x2d_obs in corr['reproj_obs']:
            x2d_pred = reproject(K_all[ci], R_all[ci], t_all[ci], X_j)
            err = (x2d_pred - x2d_obs.to(device)).norm().item()
            errors.append(err)
    return errors
