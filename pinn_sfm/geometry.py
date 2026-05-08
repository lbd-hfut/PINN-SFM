"""
Geometric utilities: rotation conversions, projection matrices, triangulation, reprojection.
Contains both:
  - PyTorch functions (differentiable, for training)
  - NumPy functions (for data generation / MATLAB conversion)
"""

import torch
import numpy as np


# ══════════════════════════════════════════════════════════════
#  PyTorch (differentiable) — used in PINN-SfM training
# ══════════════════════════════════════════════════════════════

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: (B, 4)  [w, x, y, z]
    Returns: R (B, 3, 3), orthogonal with det=1
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y**2 + z**2),  2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),        1 - 2*(x**2 + z**2),  2*(y*z - w*x),
        2*(x*z - w*y),        2*(y*z + w*x),         1 - 2*(x**2 + y**2)
    ], dim=-1).reshape(-1, 3, 3)
    return R


def build_proj_matrix(K: torch.Tensor,
                      R: torch.Tensor,
                      t: torch.Tensor) -> torch.Tensor:
    """Build projection matrix P = K @ [R | t]."""
    Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (3, 4)
    return K @ Rt


def triangulate_dlt(Ps: torch.Tensor,
                    pts2d: torch.Tensor) -> torch.Tensor:
    """
    Linear DLT triangulation via SVD.
    Ps:    (V, 3, 4)  projection matrices
    pts2d: (V, 2)     observed 2D points
    Returns: X (3,)   world 3D point
    """
    rows = []
    for v in range(Ps.shape[0]):
        P = Ps[v]
        u, vv = pts2d[v, 0], pts2d[v, 1]
        rows.append(u * P[2:3] - P[0:1])
        rows.append(vv * P[2:3] - P[1:2])

    A = torch.cat(rows, dim=0)  # (2V, 4)
    _, _, Vh = torch.linalg.svd(A, full_matrices=False)
    X_h = Vh[-1]
    X = X_h[:3] / (X_h[3:4] + 1e-10)
    return X


def reproject(K: torch.Tensor,
              R: torch.Tensor,
              t: torch.Tensor,
              X: torch.Tensor) -> torch.Tensor:
    """Project 3D point X into camera (K, R, t). Returns (2,) pixel coords."""
    Xc = R @ X + t
    xh = K @ Xc
    return xh[:2] / (xh[2:3] + 1e-8)


# ══════════════════════════════════════════════════════════════
#  NumPy (MATLAB-style) — used for data generation
# ══════════════════════════════════════════════════════════════

def eul2R(Ax: float, Ay: float, Az: float) -> np.ndarray:
    """
    Euler angles (degrees) to rotation matrix.
    Convention: R = Rz @ Ry @ Rx (same as MATLAB).
    """
    ax = np.deg2rad(Ax)
    ay = np.deg2rad(Ay)
    az = np.deg2rad(Az)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), np.sin(ax)],
                   [0, -np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, -np.sin(ay)],
                   [0, 1, 0],
                   [np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), np.sin(az), 0],
                   [-np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx
