#!/usr/bin/env python3
"""
Legacy entry point — delegates to pinn_sfm.data.synthetic.
"""
from pinn_sfm.data.synthetic import generate_synthetic_data_realistic, convert_to_cam0_world

if __name__ == '__main__':
    correspondences, gt_cameras, gt_points = generate_synthetic_data_realistic()
    print(f"Generated {len(correspondences)} correspondences")
    print(f"Example correspondence: {correspondences[0]}")
    print("\n=== Ground Truth Cameras ===")
    for i, (K, R, t) in enumerate(gt_cameras):
        print(f"\n--- Camera {i} ---\nK = {K}\nR = {R}\nt = {t}")
