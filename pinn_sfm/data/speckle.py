"""
3D speckle scene generation (MATLAB → Python conversion of Generating3DScenes).
Creates a 3D point cloud with grayscale intensities sampled from a speckle image.
Supports planar, cylindrical, and sinusoidal surfaces.
"""

import numpy as np


def generate_speckle_scene(
    speckle_image: np.ndarray,
    num_points: int = 1000000,
    x_range=(-50, 50),
    y_range=(-50, 50),
    surface_type: str = 'Plane',
    seed: int = 42,
    verbose: bool = True,
):
    """
    Generate 3D speckle scene points with intensities.

    Parameters
    ----------
    speckle_image : (H, W) or (H, W, 3) uint8 image
        The speckle pattern used for intensity sampling.
    num_points : int
        Number of 3D points to generate.
    x_range, y_range : (float, float)
        World-coordinate ranges for X and Y.
    surface_type : str
        One of 'Plane', 'Inner cylindrical surface', 'Outer cylindrical surface', 'Sine surface'.
    seed : int
        Random seed.
    verbose : bool
        Print progress for large point counts.

    Returns
    -------
    points : (N, 3) float32 array
        3D world coordinates.
    intensities : (N,) float32 array
        Grayscale intensities sampled from the speckle image.
    """
    rng = np.random.default_rng(seed)

    if speckle_image.ndim == 3:
        speckle_image = color.rgb2gray(speckle_image)
    speckle_image = speckle_image.astype(np.float64)
    img_h, img_w = speckle_image.shape

    x_low, x_upp = x_range
    y_low, y_upp = y_range
    x_span = x_upp - x_low
    y_span = y_upp - y_low

    block_size = 1_000_000
    num_blocks = int(np.ceil(num_points / block_size))

    points = np.zeros((num_points, 3), dtype=np.float32)
    intensities = np.zeros(num_points, dtype=np.float32)

    from scipy.ndimage import map_coordinates

    for block in range(num_blocks):
        start = block * block_size
        end = min(start + block_size, num_points)
        n = end - start

        x = x_low + x_span * rng.random(n)
        y = y_low + y_span * rng.random(n)

        # Map world coords → pixel coords for intensity sampling
        img_x = (x - x_low) / x_span * (img_w - 1)
        img_y = (y - y_low) / y_span * (img_h - 1)

        # Bilinear interpolation from speckle image
        coords = np.stack([img_y, img_x])
        block_intensities = map_coordinates(speckle_image, coords, order=1, mode='constant')

        # Z coordinate based on surface type
        if surface_type == 'Plane':
            z = np.zeros_like(x)
        elif surface_type == 'Inner cylindrical surface':
            radius = x_span
            z = np.sqrt(np.maximum(radius**2 - x**2, 0)) - radius
        elif surface_type == 'Outer cylindrical surface':
            radius = x_span
            z = -np.sqrt(np.maximum(radius**2 - x**2, 0)) + radius
        elif surface_type == 'Sine surface':
            radius = x_span
            amplitude = (radius - radius * np.sin(np.deg2rad(60))) / 2
            z = amplitude * np.sin(2 * np.pi / x_span * x)
        else:
            raise ValueError(f"Unknown surface type: {surface_type}")

        points[start:end] = np.stack([x, y, z], axis=1)
        intensities[start:end] = block_intensities

        if verbose:
            print(f'  Block {block + 1}/{num_blocks} ({n} points)')

    return points, intensities
