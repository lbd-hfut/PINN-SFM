"""
Loss functions for PINN-SfM training.
"""

import torch
from .geometry import build_proj_matrix, triangulate_dlt, reproject


def compute_reprojection_loss(
    K_all: torch.Tensor,
    R_all: torch.Tensor,
    t_all: torch.Tensor,
    correspondences: list,
    device: torch.device,
) -> torch.Tensor:
    """
    Physics-informed reprojection loss.
    For each 3D point:
      1. Triangulate from `triang_obs` views
      2. Reproject onto `reproj_obs` views
      3. Accumulate reprojection error
    Gradient flows: θ → (K,R,t) → triangulation → X_j → reprojection → loss
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
            err = ((x2d_pred - x2d_obs.to(device)) ** 2).sum()
            errors.append(err)

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
