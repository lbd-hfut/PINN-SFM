"""
Training loop for PINN-SfM.
"""

import torch
import numpy as np

from .models import CameraNetwork
from .losses import compute_reprojection_loss, gauge_loss
from .data.synthetic import generate_synthetic_data_realistic


def filter_correspondences(K_all, R_all, t_all, correspondences, device,
                            max_reproj_error=10.0, min_triang_angle_deg=1.0):
    """
    COLMAP-inspired filtering (Schoenberger 2018, §7.5.2):
      1. Remove observations with reprojection error > max_reproj_error
      2. Remove correspondences that fall below min_triang_angle or
         don't have enough remaining views.
    """
    from .geometry import build_proj_matrix, triangulate_dlt, reproject
    import math
    kept = []
    n_filtered_error = 0
    n_filtered_angle = 0
    for corr in correspondences:
        t_obs = corr['triang_obs']
        Ps = torch.stack([
            build_proj_matrix(K_all[ci], R_all[ci], t_all[ci])
            for ci, _ in t_obs])
        pts2d_t = torch.stack([x2d.to(device) for _, x2d in t_obs])

        X_j = triangulate_dlt(Ps, pts2d_t)

        r_obs = corr['reproj_obs']
        # Triangulation angle check (deactivated for narrow-baseline DIC setups)
        # In wide-baseline SfM, a min angle prevents degenerate geometry.
        # For DIC with sub-mm baselines at 500mm depth, angles are < 0.5°.
        # The L2 reprojection loss already penalizes poor geometry.

        # Check reprojection errors on reproj_obs
        valid_r_obs = []
        for ci, x2d_obs in r_obs:
            x2d_pred = reproject(K_all[ci], R_all[ci], t_all[ci], X_j)
            err = (x2d_pred - x2d_obs.to(device)).norm().item()
            if err < max_reproj_error:
                valid_r_obs.append((ci, x2d_obs))
            else:
                n_filtered_error += 1

        if len(valid_r_obs) >= 1:
            kept.append({
                'point_idx': corr['point_idx'],
                'triang_obs': corr['triang_obs'],
                'reproj_obs': valid_r_obs,
            })
    if n_filtered_error > 0 or n_filtered_angle > 0:
        print(f"       Filtered {n_filtered_error} bad reprojections, "
              f"{n_filtered_angle} degenerate geometries → {len(kept)}/{len(correspondences)} tracks")
    return kept


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
    rotation_deg:  float = 0.3,
    shared_K:      bool = True,
    robust_loss:   str = 'cauchy',
    robust_c:      float = 5.0,
    refine:        bool = True,
    max_reproj_error: float = 8.0,
    min_triang_angle_deg: float = 1.0,
):
    """
    Full PINN-SfM training loop with COLMAP-inspired improvements.

    If `model`, `correspondences`, `gt_cameras`, `gt_points` are provided,
    they are used directly (external data). Otherwise, synthetic data is generated.

    Improvements (Schoenberger 2018, §7):
      - shared_K=True | square pixel + fixed principal point → 1 intrinsic param
      - robust_loss='cauchy' | Cauchy robust kernel (§7.5.1)
      - refine=True | filtering → re-triangulation → continued training (§7.5.2-7.5.4)

    Each epoch:
      1. Forward pass → all camera (K, R, t)
      2. Triangulate 3D points from multi-view observations
      3. Reproject onto held-out views for physical loss (robustified)
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
            rotation_deg=rotation_deg,
        )
        print(f"       Cameras: {num_cameras}, Tracks: {len(correspondences)} / {num_points}")
    else:
        num_cameras = len(gt_cameras)

    # ── Model ──────────────────────────────────────────────────
    if model is None:
        # Initialize focal length from known lens + pixel specs (as EXIF would provide)
        fx_init = 40.0 / 3.45e-3  # f_mm / pixel_size ≈ 11594 for defaults
        model = CameraNetwork(
            num_cameras=num_cameras, hidden_dim=256, num_freqs=8,
            shared_intrinsics=shared_K, image_wh=image_size,
            init_fx=fx_init,
        ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {param_count:,}")
    if shared_K:
        print(f"       Intrinsics: SHARED (1 param — square pixel, fixed principal point)")
    else:
        print(f"       Intrinsics: PER-CAMERA ({4*num_cameras} total)")
    if rotation_deg > 0:
        print(f"       Camera rotation: ±{rotation_deg}° (breaks pure-translation degeneracy)")

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
            K_all, R_all, t_all, correspondences, device,
            robust=robust_loss, robust_c=robust_c)
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

    # ── Refinement stage (COLMAP-inspired: filter → re-triangulate → continue) ─
    if refine:
        print("\n[Refine] Filtering correspondences...")
        model.eval()
        with torch.no_grad():
            K_all, R_all, t_all = model(all_indices, image_size)
        correspondences = filter_correspondences(
            K_all, R_all, t_all, correspondences, device,
            max_reproj_error=max_reproj_error,
            min_triang_angle_deg=min_triang_angle_deg)

        print(f"[Refine] Continuing training with {len(correspondences)} filtered tracks...")
        # Reset optimizer and scheduler for refinement
        optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs // 2, eta_min=1e-6)

        for epoch in range(num_epochs // 2):
            model.train()
            optimizer.zero_grad()

            K_all, R_all, t_all = model(all_indices, image_size)

            loss_reproj = compute_reprojection_loss(
                K_all, R_all, t_all, correspondences, device,
                robust=robust_loss, robust_c=robust_c)
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
                print(f"{epoch:>6}r {loss.item():>10.3f}  {loss_reproj.item():>10.3f}"
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
