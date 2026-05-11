"""
Loss functions for PINN-SfM training.
"""

import torch
from .geometry import build_proj_matrix, triangulate_dlt, reproject


def cauchy_loss(error_sq: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Cauchy robust loss function.
    Behaves like L2 for small errors, saturates for large errors.
    Used in COLMAP's local bundle adjustment (Schoenberger 2018, §7.5.1).

    Args:
        error_sq: squared error (per-observation), shape (*,)
        c:      scale parameter — errors >> c are down-weighted
    Returns:
        loss value per observation, shape (*,)
    """
    return c ** 2 * torch.log1p((error_sq / c ** 2).clamp(min=1e-12))


def compute_reprojection_loss(
    K_all: torch.Tensor,
    R_all: torch.Tensor,
    t_all: torch.Tensor,
    correspondences: list,
    device: torch.device,
    robust: str = 'cauchy',
    robust_c: float = 5.0,
) -> torch.Tensor:
    """
    Physics-informed reprojection loss with optional robust kernel.

    For each 3D point:
      1. Triangulate from `triang_obs` views
      2. Reproject onto `reproj_obs` views
      3. Accumulate robustified reprojection error
    Gradient flows: θ → (K,R,t) → triangulation → X_j → reprojection → loss

    robust: 'l2' (original), 'cauchy' (Cauchy robust loss, default)
    """
    errors = []
    for corr in correspondences:
        t_obs = corr['triang_obs']
        Ps = torch.stack([
            build_proj_matrix(K_all[ci], R_all[ci], t_all[ci])
            for ci, _ in t_obs
        ])
        pts2d_t = torch.stack([x2d.to(device) for _, x2d in t_obs])

        X_j = triangulate_dlt(Ps, pts2d_t)

        for ci, x2d_obs in corr['reproj_obs']:
            x2d_pred = reproject(K_all[ci], R_all[ci], t_all[ci], X_j)
            err_sq = ((x2d_pred - x2d_obs.to(device)) ** 2).sum()
            if robust == 'cauchy':
                errors.append(cauchy_loss(err_sq, c=robust_c))
            else:
                errors.append(err_sq)

    if not errors:
        return torch.zeros(1, requires_grad=True, device=device).squeeze()
    return torch.stack(errors).mean()


def gauge_loss(R0: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
    """
    Gauge constraint: fix camera-0 as world origin.
    Eliminates the 7-DoF ambiguity in SfM.
    """
    I = torch.eye(3, device=R0.device, dtype=R0.dtype)
    return ((R0 - I) ** 2).sum() + (t0 ** 2).sum()
