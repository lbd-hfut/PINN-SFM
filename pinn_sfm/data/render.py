"""
Image rendering pipeline (MATLAB → Python conversion).
Projects 3D speckle points into camera views with optional lens distortion
and noise, producing synthetic speckle images.
"""

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


def project_points_distorted(
    Fx, Fy, Cx, Cy, R, T, k1, k2, world_points: np.ndarray
) -> np.ndarray:
    """
    Project 3D world points to pixel coordinates with radial distortion.
    MATLAB equivalent: CoordPixCal_distortion_v3

    Parameters
    ----------
    Fx, Fy : float  — focal lengths (pixels)
    Cx, Cy : float  — principal point (pixels)
    R      : (3, 3) — rotation matrix (world → camera)
    T      : (3,)   — translation vector (world → camera)
    k1, k2 : float  — radial distortion coefficients
    world_points : (3, N) or (N, 3) — 3D world coordinates

    Returns
    -------
    uv : (2, N) distorted pixel coordinates
    """
    if world_points.ndim == 2 and world_points.shape[1] == 3:
        world_points = world_points.T  # (3, N)
    num_pts = world_points.shape[1]

    K = np.array([[Fx, 0, Cx],
                  [0, Fy, Cy],
                  [0,  0,  1]], dtype=np.float64)

    uv = np.zeros((2, num_pts), dtype=np.float64)
    for i in range(num_pts):
        Xc = R @ world_points[:, i] + T
        xh = K @ Xc
        u_n, v_n, w = xh[0] / xh[2], xh[1] / xh[2], xh[2]

        # Radial distortion
        x_norm = (u_n - Cx) / Fx
        y_norm = (v_n - Cy) / Fy
        r2 = x_norm**2 + y_norm**2
        scale = 1 + k1 * r2 + k2 * r2**2
        u_dist = Fx * x_norm * scale + Cx
        v_dist = Fy * y_norm * scale + Cy
        uv[:, i] = [u_dist, v_dist]

    return uv


def render_speckle_image(
    uv: np.ndarray,
    intensities: np.ndarray,
    image_width: int,
    image_height: int,
    sigma: float = 0.5,
    noise_std: float = 0.0,
    noise_seed: int = None,
) -> np.ndarray:
    """
    Render a speckle image from projected 2D points with intensities.
    MATLAB equivalent: GeneratingImage

    Parameters
    ----------
    uv          : (2, N) pixel coordinates
    intensities : (N,)  grayscale intensities
    image_width, image_height : output image size
    sigma       : Gaussian blur standard deviation
    noise_std   : Gaussian noise standard deviation (grayscale)

    Returns
    -------
    image : (H, W) uint8 speckle image
    """
    u, v = uv[0], uv[1]

    # Filter points within image bounds
    valid = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    u, v = u[valid], v[valid]
    intensities = intensities[valid]

    # Accumulate intensities per pixel (nearest-neighbor)
    intensity_sum = np.zeros((image_height, image_width), dtype=np.float64)
    count = np.zeros((image_height, image_width), dtype=np.int32)

    rows = np.round(v).astype(np.int32)
    cols = np.round(u).astype(np.int32)

    rows = np.clip(rows, 0, image_height - 1)
    cols = np.clip(cols, 0, image_width - 1)

    np.add.at(intensity_sum, (rows, cols), intensities)
    np.add.at(count, (rows, cols), 1)

    # Average intensity per pixel
    mask = count > 0
    image = np.zeros_like(intensity_sum)
    image[mask] = intensity_sum[mask] / count[mask]

    # Normalize to [0, 240]
    img_min = image.min()
    img_max = image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min) * 240
    image = np.round(image).astype(np.uint8)

    # Gaussian blur (simulate camera PSF)
    if sigma > 0:
        image = gaussian_filter(image.astype(np.float32), sigma).astype(np.uint8)

    # Add noise
    if noise_std > 0:
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(0, noise_std, size=image.shape)
        image = np.clip(np.round(image.astype(np.float32) + noise), 0, 255).astype(np.uint8)

    return image


def generate_image(
    Fx, Fy, Cx, Cy, R, T, k1, k2,
    world_points: np.ndarray,
    intensities: np.ndarray,
    noise_std: float,
    sigma: float,
    image_width: int,
    image_height: int,
    noise_seed: int = None,
):
    """
    Full pipeline: project 3D points with distortion → render speckle image.
    MATLAB equivalent: GeneratingImage.

    Returns
    -------
    uv : (2, N) distorted pixel coordinates
    image : (H, W) uint8 speckle image
    """
    uv = project_points_distorted(Fx, Fy, Cx, Cy, R, T, k1, k2, world_points)
    image = render_speckle_image(
        uv, intensities, image_width, image_height,
        sigma=sigma, noise_std=noise_std, noise_seed=noise_seed,
    )
    return uv, image
