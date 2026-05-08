#!/usr/bin/env python3
"""
Speckle image generation with arc camera array (MATLAB MultiViewImaging equivalent).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from skimage import io

from pinn_sfm.data.camera_array import build_arc_array
from pinn_sfm.data.speckle import generate_speckle_scene
from pinn_sfm.data.render import generate_image


def main():
    # ── Camera parameters ─────────────────────────────────────
    L_pixel, H_pixel = 1440, 1080
    pixel_size = 3.45 / 1000  # mm
    f = 4  # mm
    Fx = Fy = f / pixel_size
    Cx = L_pixel / 2 + 0.5
    Cy = H_pixel / 2 + 0.5
    k1, k2 = 0.1, 0.05
    Tz = 500.0  # mm
    num_views = 5
    angle_view = 10  # degrees

    # ── Camera array ───────────────────────────────────────────
    R_arr, T_arr = build_arc_array(num_views, Tz, angle_view)

    # ── Speckle scene ─────────────────────────────────────────
    print("Loading speckle pattern...")
    speckle_img = io.imread('pinn_sfm/data/002.bmp')

    print("Generating 3D speckle scene (12M points)...")
    scene_pts, scene_intensities = generate_speckle_scene(
        speckle_img, num_points=12_000_000,
        x_range=(-50, 50), y_range=(-50, 50),
        surface_type='Outer cylindrical surface',
        seed=42, verbose=True,
    )

    # ── Render images ─────────────────────────────────────────
    noise_std = 2
    sigma = 0.5
    seeds = {0: 25, 1: 259, 2: 856, 3: 125, 4: 68}

    for i in range(num_views):
        print(f"Rendering view {i + 1}/{num_views}...")
        uv, image = generate_image(
            Fx, Fy, Cx, Cy,
            R_arr[:, :, i], T_arr[:, i],
            k1, k2,
            scene_pts.T, scene_intensities,
            noise_std=noise_std, sigma=sigma,
            image_width=L_pixel, image_height=H_pixel,
            noise_seed=seeds.get(i),
        )
        filename = f'Image{i + 1}.bmp'
        io.imsave(filename, image)
        print(f"  Saved {filename}")

    # Save camera parameters
    import scipy.io as sio
    sio.savemat('CameraArray_Parameters.mat', {
        'Fx': Fx, 'Fy': Fy, 'Cx': Cx, 'Cy': Cy,
        'k1': k1, 'k2': k2,
        'R_world2camArr': R_arr,
        'T_world2camArr': T_arr,
    })
    print("Camera parameters saved to CameraArray_Parameters.mat")


if __name__ == '__main__':
    main()
