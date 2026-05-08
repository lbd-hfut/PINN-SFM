# PINN-SfM

Physics-Informed Neural Network for Structure from Motion.

将相机参数编码为神经网络权重，利用可微三角化和重投影误差作为物理约束，端到端学习相机内参、外参和三维点云。

## 架构

```
相机索引 i  ─→  CameraNetwork ─→  (K_i, R_i, t_i)
                                      │
                      2D匹配点 ─→  可微三角化(DLT)
                                      │
                                   三维点 X_j
                                      │
                      未参与三角化的视角 ─→  重投影
                                      │
                                   重投影误差 ← 物理损失
```

- **CameraNetwork** — 全连接网络 + 位置编码，输入相机索引输出内参/外参
- **可微三角化** — 线性DLT，SVD求解，梯度可反传
- **物理损失** — 重投影误差作为无参数物理约束

## 项目结构

```
pinn_sfm/                          # 主包
├── models.py                      # CameraNetwork, PositionalEncoding
├── geometry.py                    # quat_to_rotmat, eul2R, triangulate_dlt, reproject
├── losses.py                      # compute_reprojection_loss, gauge_loss
├── training.py                    # 训练主循环
├── visualization.py               # 绘图工具
├── 002.bmp                        # 散斑图
└── data/
    ├── synthetic.py               # 简化合成数据生成
    ├── speckle.py                 # 3D散斑场景生成（平面/圆柱/正弦曲面）
    ├── render.py                  # 图像渲染（畸变+模糊+噪声）
    └── camera_array.py            # 相机阵列配置（弧线/直线/网格）
scripts/
├── run_pinn_sfm.py                # 合成数据训练入口
├── generate_speckle_arc.py        # 弧线阵列散斑图生成
├── generate_speckle_grid.py       # 网格阵列散斑图生成
└── run_speckle_pinn_sfm.py        # 散斑生成+训练全流程
```

## 依赖

- PyTorch
- NumPy
- Matplotlib
- SciPy
- scikit-image

## 使用

```bash
# 合成数据训练
conda run -n dic python scripts/run_pinn_sfm.py

# 弧线阵列散斑图生成（5视角，10°夹角，外圆柱面）
conda run -n dic python scripts/generate_speckle_arc.py

# 网格阵列散斑图生成（2×2网格，0.5px步长，平面）
conda run -n dic python scripts/generate_speckle_grid.py

# 散斑生成 + PINN-SfM 全流程
conda run -n dic python scripts/run_speckle_pinn_sfm.py
```

## MATLAB → Python

`data/speckle.py`, `data/render.py`, `data/camera_array.py` 由原始 MATLAB 代码 (`MultiViewImaging.m`, `MultiViewImaging2.m`) 转换而来，生成三维散斑场景并渲染为合成相机图像。

## License

MIT
