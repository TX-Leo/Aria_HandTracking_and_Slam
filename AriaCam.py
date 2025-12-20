# -*- coding: utf-8 -*-
# @FileName: AriaCam.py

import numpy as np
from dataclasses import dataclass

@dataclass
class AriaCam:
    """
    [数据结构] 定义单帧 RGB 相机的相关数据。
    包含图像本身以及内参、外参等几何信息。
    """
    rgb: np.ndarray = None  # RGB 图像数据 (H, W, 3)
    h: int = 0              # 图像高度
    w: int = 0              # 图像宽度
    k: np.ndarray = None    # 内参矩阵 (3, 3)
    d: np.ndarray = None    # 畸变系数
    c2w: np.ndarray = None  # 相机到世界的变换矩阵 (4, 4) Pose