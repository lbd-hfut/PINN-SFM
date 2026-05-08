#!/usr/bin/env python3
"""
Speckle image generation with grid camera array (MATLAB MultiViewImaging2 equivalent).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from skimage import io

from pinn_sfm.data.camera_array import build_grid_array
from pinn_sfm.data.speckle import generate_speckle_scene
from pinn_sfm.data.render import generate_image


def main():
    # ── Movement parameters ───────────────────────────────────
    M, N = 1, 1
    step_u, step_v = 0.5, 0.5

    # ── Camera parameters ─────────────────────────────────────
    L_pixel, H_pixel = 1000, 1000
    pixel_size = 3.45 / 1000  # mm
    f = 40  # mm
    Fx = Fy = f / pixel_size
    Cx = L_pixel / 2 + 0.5
    Cy = H_pixel / 2 + 0.5
    k1, k2 = 0, 0
    Tz = 500.0  # mm

    # ── Camera array ───────────────────────────────────────────
    R_arr, T_arr, u_disp, v_disp, Tx_list, Ty_list = build_grid_array(
        M=M, N=N, step_u=step_u, step_v=step_v,
        Fx=Fx, Fy=Fy, Tz_world2refcam=Tz,
    )

    total_images = (M + 1) * (N + 1)
    print(f'Total images: {M + 1}x{N + 1} = {total_images}')
    print(f'Horizontal pixel displacement: {u_disp}')
    print(f'Vertical pixel displacement:   {v_disp}')
    print(f'Horizontal physical displacement (mm): {Tx_list}')
    print(f'Vertical physical displacement (mm):   {Ty_list}')

    # ── Speckle scene ─────────────────────────────────────────
    print("Loading speckle pattern...")
    speckle_img = io.imread('pinn_sfm/data/002.bmp')

    print("Generating 3D speckle scene (10M points)...")
    scene_pts, scene_intensities = generate_speckle_scene(
        speckle_img, num_points=10_000_000,
        x_range=(-50, 50), y_range=(-50, 50),
        surface_type='Plane',
        seed=42, verbose=True,
    )
    scene_pts = scene_pts.T

    # ── Render images ─────────────────────────────────────────
    noise_std = 1
    sigma = 0.0001
    img_index = 1

    for v in range(N + 1):
        for h in range(M + 1):
            print(f'Generating image {img_index}/{total_images}...')
            print(f'  h_disp={u_disp[h]:.1f}px, v_disp={v_disp[v]:.1f}px')

            seed = v * (M + 1) + h + 100
            R_cur = R_arr[:, :, h, v]
            T_cur = T_arr[:, 0, h, v]

            uv, image = generate_image(
                Fx, Fy, Cx, Cy,
                R_cur, T_cur, k1, k2,
                scene_pts, scene_intensities,
                noise_std=noise_std, sigma=sigma,
                image_width=L_pixel, image_height=H_pixel,
                noise_seed=seed,
            )

            filename = f'RG_{img_index}.bmp'
            io.imsave(filename, image)
            print(f'  Saved {filename}')
            img_index += 1

    # Save camera parameters
    import scipy.io as sio
    sio.savemat('CameraArray_Parameters.mat', {
        'Fx': Fx, 'Fy': Fy, 'Cx': Cx, 'Cy': Cy,
        'k1': k1, 'k2': k2,
        'R_world2camArr': R_arr,
        'T_world2camArr': T_arr,
        'u_displacement': u_disp,
        'v_displacement': v_disp,
        'Tx_list': Tx_list,
        'Ty_list': Ty_list,
        'M': M, 'N': N,
        'step_u': step_u, 'step_v': step_v,
    })
    print('Camera parameters saved.')


if __name__ == '__main__':
    main()
