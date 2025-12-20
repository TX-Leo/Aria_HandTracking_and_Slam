# TODO:
# 1. 添加AriaStereo相机的参数到class里

# -*- coding: utf-8 -*-
# @FileName: AriaStereo.py

import math
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from projectaria_tools.core import calibration
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeQueryOptions, TimeDomain
from projectaria_tools.core.sophus import SE3

# [Stereo] 立体匹配视场角 (FOV)
# 我们保持 85 度 FOV，但像素分辨率将动态跟随 RGB 相机
STEREO_FOV_DEG = 85  

@dataclass
class AriaStereo:
    """
    [数据结构] 存储一对经过校正和对齐的立体图像。
    """
    left_img: np.ndarray = None
    right_img: np.ndarray = None
    valid: bool = False

# =========================================================================
# [Stereo Core] 立体图像生成器 (Stereo Rectification Generator)
# =========================================================================

class AriaStereoGenerator:
    """
    [类功能]: 负责生成 FoundationStereo 所需的极线校正立体图像。
    
    [V7.2 更新]: 
    - 动态适配 RGB 分辨率。
    - 输出图像顺时针旋转 90 度。
    """
    def __init__(self, provider: VrsDataProvider, device_calib, target_shape: Tuple[int, int]):
        """
        Args:
            provider: VRS 数据提供者
            device_calib: 设备标定信息
            target_shape: (width, height) 这里的宽高指 RGB 原始(旋转前)的宽高。
                          因为我们要在 rectification 之后才旋转，所以虚拟相机设为 landscape。
        """
        self.provider = provider
        self.device_calib = device_calib
        
        # 这里的 target_w/h 是为了匹配 RGB 图像的大小
        self.out_w, self.out_h = target_shape
        
        print(f"[Stereo] 初始化虚拟立体相机: Res=({self.out_w}x{self.out_h}), FOV={STEREO_FOV_DEG}°")

        # 获取流 ID
        self.sid_left = provider.get_stream_id_from_label("camera-slam-left")
        self.sid_right = provider.get_stream_id_from_label("camera-slam-right")
        
        # 获取原始标定 (Fisheye)
        self.src_calib_left = device_calib.get_camera_calib("camera-slam-left")
        self.src_calib_right = device_calib.get_camera_calib("camera-slam-right")
        
        # --- 预计算校正映射表 ---
        
        # 1. 提取原始外参
        T_d_cl = self.src_calib_left.get_transform_device_camera()
        T_d_cr = self.src_calib_right.get_transform_device_camera()
        mat_l = T_d_cl.to_matrix()
        mat_r = T_d_cr.to_matrix()
        pos_l = mat_l[:3, 3]
        pos_r = mat_r[:3, 3]
        
        # 2. 计算基线方向 (New X Axis)
        baseline = pos_r - pos_l
        x_new = baseline / np.linalg.norm(baseline)
        
        # 3. 构建旋转矩阵 (保持左相机 Z 轴朝向)
        z_original = mat_l[:3, 2]
        y_new = np.cross(z_original, x_new)
        y_new /= np.linalg.norm(y_new)
        z_new = np.cross(x_new, y_new)
        z_new /= np.linalg.norm(z_new)
        R_rect = np.column_stack((x_new, y_new, z_new))
        
        # 4. 构建虚拟相机外参
        T_d_new_cl_mat = np.eye(4)
        T_d_new_cl_mat[:3, :3] = R_rect
        T_d_new_cl_mat[:3, 3] = pos_l
        
        T_d_new_cr_mat = np.eye(4)
        T_d_new_cr_mat[:3, :3] = R_rect
        T_d_new_cr_mat[:3, 3] = pos_r
        
        self.T_d_new_cl = SE3.from_matrix(T_d_new_cl_mat)
        self.T_d_new_cr = SE3.from_matrix(T_d_new_cr_mat)
        
        # 5. 构建虚拟相机内参 (Pinhole)
        # 使用 target_w 计算焦距，以保证水平 FOV 覆盖
        f_len = (self.out_w / 2.0) / math.tan(math.radians(STEREO_FOV_DEG / 2.0))
        
        self.dst_calib_left = calibration.get_linear_camera_calibration(
            self.out_w, self.out_h, f_len, "stereo-left", self.T_d_new_cl
        )
        self.dst_calib_right = calibration.get_linear_camera_calibration(
            self.out_w, self.out_h, f_len, "stereo-right", self.T_d_new_cr
        )

    def get_rectified_pair(self, timestamp_ns: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取校正并旋转后的立体图像。
        """
        # [Fix] 增加 TimeDomain 参数
        idx_l = self.provider.get_index_by_time_ns(
            self.sid_left, timestamp_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
        )
        idx_r = self.provider.get_index_by_time_ns(
            self.sid_right, timestamp_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
        )
        
        if idx_l == -1 or idx_r == -1:
            return None, None
            
        data_l = self.provider.get_image_data_by_index(self.sid_left, idx_l)
        data_r = self.provider.get_image_data_by_index(self.sid_right, idx_r)
        
        img_l_raw = data_l[0].to_numpy_array()
        img_r_raw = data_r[0].to_numpy_array()
        
        # 1. 校正 (Rectification) -> 此时是横向的 (RGB_W x RGB_H)
        img_l_rect = calibration.distort_by_calibration(
            img_l_raw, self.dst_calib_left, self.src_calib_left, InterpolationMethod.BILINEAR
        )
        img_r_rect = calibration.distort_by_calibration(
            img_r_raw, self.dst_calib_right, self.src_calib_right, InterpolationMethod.BILINEAR
        )
        
        # 2. 旋转 90 度顺时针 (Matching RGB k=-1)
        # 旋转后尺寸变为 (RGB_H x RGB_W)，与 RGB 处理后的输出完全一致
        img_l_final = np.rot90(img_l_rect, k=-1).copy()
        img_r_final = np.rot90(img_r_rect, k=-1).copy()
        
        return img_l_final, img_r_final