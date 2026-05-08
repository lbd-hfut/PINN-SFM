"""
Camera array configuration (MATLAB → Python conversion).
Supports:
  - Arc / linear 1D arrays (MultiViewImaging)
  - 2D grid arrays with pixel-precise displacements (MultiViewImaging2)
"""

import numpy as np
from ..geometry import eul2R


def build_arc_array(
    num_views: int = 5,
    Tz_world2refcam: float = 500.0,
    angle_per_view: float = 10.0,
):
    """
    Build an arc-shaped camera array.  MATLAB equivalent: MultiViewImaging arc mode.

    Parameters
    ----------
    num_views : int — number of cameras
    Tz_world2refcam : float — working distance (mm)
    angle_per_view : float — angle between adjacent cameras (degrees)

    Returns
    -------
    R_world2cam : (3, 3, num_views) rotation matrices
    T_world2cam : (3, num_views)   translation vectors
    """
    R_world2refcam = eul2R(0, 0, 0)
    T_world2refcam = np.array([0, 0, Tz_world2refcam], dtype=np.float64)

    k = np.arange(1, (num_views - 1) // 2 + 1)
    if num_views % 2 == 0:
        k = np.concatenate([k, [k[-1] + 1]])

    # Build alternating angle sequence: [0, -1, +1, -2, +2, ...]
    angles = np.zeros(num_views)
    if num_views > 1:
        seq = np.column_stack([-k, k]).ravel()
        angles[1:] = seq[:num_views - 1]
    angles = np.deg2rad(angles * angle_per_view)

    R_world2cam = np.zeros((3, 3, num_views))
    T_world2cam = np.zeros((3, num_views))

    for i in range(num_views):
        R_i2ref = eul2R(0, np.rad2deg(angles[i]), 0)
        T_i2ref = np.array([
            2 * Tz_world2refcam * np.sin(angles[i]/2) * np.cos(angles[i]/2),
            0,
            2 * Tz_world2refcam * np.sin(angles[i]/2) * np.sin(angles[i]/2),
        ])

        R_ref2i = np.linalg.inv(R_i2ref)
        T_ref2i = -R_ref2i @ T_i2ref

        R_world2cam[:, :, i] = R_ref2i @ R_world2refcam
        T_world2cam[:, i] = R_ref2i @ T_world2refcam + T_ref2i

    return R_world2cam, T_world2cam


def build_linear_array(
    num_views: int = 5,
    Tz_world2refcam: float = 500.0,
    Tx_list: np.ndarray = None,
):
    """
    Build a linear (translation-only) camera array.
    MATLAB equivalent: MultiViewImaging line mode (with user Tx input).

    Parameters
    ----------
    num_views : int
    Tz_world2refcam : float
    Tx_list : (num_views,) array of X translations

    Returns
    -------
    R_world2cam : (3, 3, num_views)
    T_world2cam : (3, num_views)
    """
    if Tx_list is None:
        Tx_list = np.zeros(num_views)
    R_world2refcam = eul2R(0, 0, 0)
    T_world2refcam = np.array([0, 0, Tz_world2refcam], dtype=np.float64)

    R_world2cam = np.zeros((3, 3, num_views))
    T_world2cam = np.zeros((3, num_views))

    for i in range(num_views):
        R_i2ref = np.eye(3)
        T_i2ref = np.array([Tx_list[i], 0, 0], dtype=np.float64)

        R_ref2i = np.linalg.inv(R_i2ref)
        T_ref2i = -R_ref2i @ T_i2ref

        R_world2cam[:, :, i] = R_ref2i @ R_world2refcam
        T_world2cam[:, i] = R_ref2i @ T_world2refcam + T_ref2i

    return R_world2cam, T_world2cam


def build_grid_array(
    M: int = 1,
    N: int = 1,
    step_u: float = 0.5,
    step_v: float = 0.5,
    Fx: float = 500.0,
    Fy: float = 500.0,
    Tz_world2refcam: float = 500.0,
):
    """
    Build an (M+1)×(N+1) grid camera array with pixel-precise displacements.
    MATLAB equivalent: MultiViewImaging2.

    Parameters
    ----------
    M, N : int — horizontal/vertical movement steps
    step_u, step_v : float — pixel displacement per step
    Fx, Fy : float — focal lengths (pixels)
    Tz_world2refcam : float — working distance (mm)

    Returns
    -------
    R_world2cam : (3, 3, M+1, N+1)
    T_world2cam : (3, 1, M+1, N+1)
    u_displacement : (M+1,) pixel displacement in u
    v_displacement : (N+1,) pixel displacement in v
    Tx_list : (M+1,) physical displacement in X (mm)
    Ty_list : (N+1,) physical displacement in Y (mm)
    """
    u_disp = np.arange(M + 1) * step_u
    v_disp = np.arange(N + 1) * step_v

    # Pixel displacement → physical displacement (mm)
    Tx_list = (u_disp * Tz_world2refcam) / Fx
    Ty_list = (v_disp * Tz_world2refcam) / Fy

    R_world2refcam = eul2R(0, 0, 0)
    T_world2refcam = np.array([0, 0, Tz_world2refcam], dtype=np.float64)

    R_world2cam = np.zeros((3, 3, M + 1, N + 1))
    T_world2cam = np.zeros((3, 1, M + 1, N + 1))

    for h in range(M + 1):
        for v in range(N + 1):
            Tx = Tx_list[h]
            Ty = Ty_list[v]

            R_i2ref = np.eye(3)
            T_i2ref = np.array([Tx, Ty, 0], dtype=np.float64)
            R_ref2i = np.linalg.inv(R_i2ref)
            T_ref2i = -R_ref2i @ T_i2ref

            R_world2cam[:, :, h, v] = R_ref2i @ R_world2refcam
            T_world2cam[:, 0, h, v] = R_ref2i @ T_world2refcam + T_ref2i

    return R_world2cam, T_world2cam, u_disp, v_disp, Tx_list, Ty_list
