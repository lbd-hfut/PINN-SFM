"""
Visualization utilities for PINN-SfM results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch

from .geometry import build_proj_matrix, triangulate_dlt
from .training import collect_reproj_errors


def print_metrics(K_pred, R_pred, t_pred, correspondences,
                   gt_cameras, device, model=None, all_indices=None, image_size=None):
    """Print evaluation metrics."""
    with torch.no_grad():
        if model is not None and all_indices is not None and image_size is not None:
            K_all, R_all, t_all = model(all_indices, image_size)
        else:
            K_all, R_all, t_all = K_pred, R_pred, t_pred

        errors = collect_reproj_errors(K_all, R_all, t_all, correspondences, device)

    print(f"  Reprojection error  mean: {np.mean(errors):.3f} px  "
          f"median: {np.median(errors):.3f} px  "
          f"<2px: {np.mean(np.array(errors) < 2) * 100:.1f}%")
    gt_fx = gt_cameras[0][0][0, 0]
    print(f"\n  First 4 cameras focal length (GT fx={gt_fx:.0f}):")
    for i in range(min(4, len(gt_cameras))):
        fx_p = K_pred[i, 0, 0].item()
        fy_p = K_pred[i, 1, 1].item()
        print(f"    Camera {i}: pred fx={fx_p:.1f}  fy={fy_p:.1f}")


def plot_results(history, gt_cameras, gt_points, K_pred, R_pred, t_pred,
                 correspondences, all_indices, model, device, image_size,
                 save_path='pinn_sfm_results.png'):
    """Full diagnostic plot: loss curves, camera trajectory, 3D points, reprojection errors."""
    num_cameras = len(gt_cameras)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('PINN-SfM Training Results', fontsize=13, fontweight='bold')

    # (1) Loss curves
    ax1 = fig.add_subplot(2, 3, 1)
    ep = range(len(history['total']))
    ax1.semilogy(ep, history['total'],  label='Total Loss', linewidth=1.5)
    ax1.semilogy(ep, history['reproj'], label='Reprojection Loss', linewidth=1.5, linestyle='--')
    ax1.semilogy(ep, history['gauge'],  label='Gauge Loss', linewidth=1.5, linestyle=':')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) Camera trajectory: GT vs Pred
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    gt_pos = np.array([-R.T @ t for _, R, t in gt_cameras])
    pred_pos = np.array([
        -(R_pred[i].cpu().numpy().T @ t_pred[i].cpu().numpy())
        for i in range(num_cameras)
    ])
    ax2.plot(*gt_pos.T,   'b-o', markersize=5, label='Ground Truth')
    ax2.plot(*pred_pos.T, 'r--^', markersize=5, label='Predicted')
    ax2.set_title('Camera Trajectory (GT vs Pred)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend(fontsize=8)

    # (3) 3D point cloud: GT vs Reconstructed
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    model.eval()
    with torch.no_grad():
        K_all, R_all, t_all = model(all_indices, image_size)
    pts_recon = []
    with torch.no_grad():
        for corr in correspondences:
            t_obs = corr['triang_obs']
            Ps = torch.stack([
                build_proj_matrix(K_all[ci], R_all[ci], t_all[ci])
                for ci, _ in t_obs])
            pts2d = torch.stack([x.to(device) for _, x in t_obs])
            X_j = triangulate_dlt(Ps, pts2d)
            pts_recon.append(X_j.cpu().numpy())
    pts_recon = np.array(pts_recon)
    ax3.scatter(*gt_points.T, s=10, alpha=0.5, label='Ground Truth')
    if pts_recon.size > 0:
        ax3.scatter(*pts_recon.T, s=10, alpha=0.5, label='Reconstructed')
    else:
        ax3.text(0.5, 0.5, 0.5, 'No points', transform=ax3.transAxes, ha='center')
    ax3.set_title('3D Points (%d GT, %d Recon)' % (gt_points.shape[0], len(pts_recon)))
    ax3.legend(fontsize=8)

    # (4) Reprojection error histogram
    ax4 = fig.add_subplot(2, 3, 4)
    with torch.no_grad():
        errors = collect_reproj_errors(K_all, R_all, t_all, correspondences, device)
    ax4.hist(errors, bins=40, edgecolor='white', linewidth=0.5)
    ax4.axvline(np.median(errors), linestyle='--',
                label=f'Median {np.median(errors):.2f}px')
    ax4.set_xlabel('Reprojection Error (px)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reprojection Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # (5) Focal length comparison
    ax5 = fig.add_subplot(2, 3, 5)
    cam_ids = list(range(num_cameras))
    gt_fx = float(gt_cameras[0][0][0, 0]) if len(gt_cameras) > 0 else 500.0
    pred_fx = [K_pred[i, 0, 0].item() for i in range(num_cameras)]
    pred_fy = [K_pred[i, 1, 1].item() for i in range(num_cameras)]
    ax5.axhline(gt_fx, linewidth=1.5, label=f'GT f={gt_fx:.0f}')
    ax5.plot(cam_ids, pred_fx, '--o', markersize=5, label='Pred fx')
    ax5.plot(cam_ids, pred_fy, ':s',  markersize=5, label='Pred fy')
    ax5.set_xlabel('Camera Index')
    ax5.set_ylabel('Focal Length (px)')
    ax5.set_title('Focal Length per Camera')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # (6) Translation magnitude comparison
    ax6 = fig.add_subplot(2, 3, 6)
    gt_tnorm   = [np.linalg.norm(t) for _, _, t in gt_cameras]
    pred_tnorm = [t_pred[i].norm().item() for i in range(num_cameras)]
    w = 0.35
    x = np.arange(num_cameras)
    ax6.bar(x - w/2, gt_tnorm,   w, label='GT |t|', alpha=0.8)
    ax6.bar(x + w/2, pred_tnorm, w, label='Pred |t|', alpha=0.8)
    ax6.set_xlabel('Camera Index')
    ax6.set_ylabel('Translation Magnitude')
    ax6.set_title('Camera Translation Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n[Result] Figure saved to {save_path}")
