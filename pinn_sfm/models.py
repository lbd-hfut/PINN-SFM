"""
Neural network models for PINN-SfM: camera parameter regression.
"""

import torch
import torch.nn as nn
from .geometry import quat_to_rotmat


class PositionalEncoding(nn.Module):
    """
    NeRF-style positional encoding: maps scalar camera index to high-dim features.
    """
    def __init__(self, num_freqs: int = 8):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_freqs).float() * torch.pi
        self.register_buffer('freqs', freqs)

    @property
    def out_dim(self) -> int:
        return 2 * int(self.freqs.shape[0]) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        args = x * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args), x], dim=-1)


class SharedIntrinsics(nn.Module):
    """
    Shared intrinsic parameters: one focal length for all cameras.

    COLMAP-inspired simplifications:
      - Square pixels (fx = fy) — standard for modern DIC sensors
      - Principal point fixed at image center — well-known to be ill-posed
        to calibrate jointly with other parameters [Schoenberger 2018, §7.5.1]
      - Single log-focal-length parameter with known base_fx from lens specs

    Total trainable intrinsics: 1 parameter (vs 4 earlier, vs 20 originally).
    """
    def __init__(self, W: float, H: float, init_fx: float = 500.0):
        super().__init__()
        self.W = W
        self.H = H
        self.base_fx = init_fx
        self.log_f = nn.Parameter(torch.tensor(0.0))
        # cx, cy — fixed at image center, not trained
        # fy = fx — square pixel assumption, no aspect ratio parameter

    @property
    def fx(self) -> torch.Tensor:
        return torch.exp(self.log_f) * self.base_fx

    def forward(self, B: int) -> torch.Tensor:
        f = self.fx
        cx = self.W / 2.0
        cy = self.H / 2.0
        device = self.log_f.device
        K = torch.zeros(B, 3, 3, device=device)
        K[:, 0, 0] = f
        K[:, 1, 1] = f
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        return K


class CameraNetwork(nn.Module):
    """
    Fully-connected network mapping camera index → (K, R, t).
    One forward pass computes parameters for ALL cameras.

    When `shared_intrinsics=True`, K is shared across all cameras via
    SharedIntrinsics — drastically reduces the intrinsic-extrinsic coupling.
    """

    def __init__(self, num_cameras: int, hidden_dim: int = 256, num_freqs: int = 8,
                 shared_intrinsics: bool = True, image_wh: tuple = (1000, 1000),
                 init_fx: float = 500.0):
        super().__init__()
        self.num_cameras = num_cameras
        self.shared_intrinsics = shared_intrinsics

        self.pe = PositionalEncoding(num_freqs)
        in_dim = self.pe.out_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim), nn.Softplus(),
        )

        if not shared_intrinsics:
            self.head_K = nn.Linear(hidden_dim, 4)
        else:
            self.intrinsics = SharedIntrinsics(image_wh[0], image_wh[1], init_fx=init_fx)

        self.head_E = nn.Linear(hidden_dim, 7)   # [q_w, q_x, q_y, q_z, tx, ty, tz]

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.head_E.weight)
        nn.init.zeros_(self.head_E.bias)
        self.head_E.bias.data[0] = 1.0  # identity quaternion

    def forward(self, indices: torch.Tensor,
                image_wh: tuple = (640, 480)):
        W, H = image_wh
        idx_norm = indices.float() / max(self.num_cameras - 1, 1) * 2 - 1
        feat = self.backbone(self.pe(idx_norm))

        # ── Intrinsics ───────────────────────────────────────
        if self.shared_intrinsics:
            K = self.intrinsics(B=indices.shape[0])
        else:
            raw_K = self.head_K(feat)
            fx = torch.exp(raw_K[:, 0]) * 500.0
            fy = torch.exp(raw_K[:, 1]) * 500.0
            cx = torch.sigmoid(raw_K[:, 2]) * W
            cy = torch.sigmoid(raw_K[:, 3]) * H

            B = indices.shape[0]
            device = indices.device
            K = torch.zeros(B, 3, 3, device=device, dtype=feat.dtype)
            K[:, 0, 0] = fx
            K[:, 1, 1] = fy
            K[:, 0, 2] = cx
            K[:, 1, 2] = cy
            K[:, 2, 2] = 1.0

        # ── Extrinsics ───────────────────────────────────────
        raw_E = self.head_E(feat)
        q = raw_E[:, :4]
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        t = raw_E[:, 4:] * 20
        R = quat_to_rotmat(q)

        return K, R, t
