# -*- coding: utf-8 -*-
# @FileName: AriaStereo.py

import math
import numpy as np
import cv2
import os
import json
from typing import Optional, List
from tqdm import tqdm
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import argparse

from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeQueryOptions, TimeDomain
from projectaria_tools.core.sophus import SE3

from utils.utils_media import create_video_from_frames
from aria.AriaCam import AriaCamGenerator

STEREO_FOV_DEG = 85  
TARGET_WIDTH = 640
TARGET_HEIGHT = 640
    
@dataclass
class AriaStereoData:
    ts: int = 0            
    stereo_l: np.ndarray = None
    stereo_r: np.ndarray = None
    h: int = 0             
    w: int = 0
    k: np.ndarray = None
    baseline: float = 0.0  # 基线长度 (米)           

@dataclass
class AriaStereo:
    tss: List[int] = field(default_factory=list)
    fps: int = 0
    stereo: List[AriaStereoData] = field(default_factory=list)
    mps_path: str = None

    def __len__(self):
        return len(self.tss)
    
    @staticmethod
    def _safe_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else arr
    
    def save_aria_stereo_json(self):
        for idx in range(len(self.tss)):
            frame_dir = os.path.join(self.mps_path, "aria", "all_data", f"{idx:05d}")
            os.makedirs(frame_dir, exist_ok=True)
            stereo_l_path = os.path.join(frame_dir, "stereo_l.png")
            stereo_r_path = os.path.join(frame_dir, "stereo_r.png")

            stereo = self.stereo[idx]
            cv2.imwrite(stereo_l_path, stereo.stereo_l)
            cv2.imwrite(stereo_r_path, stereo.stereo_r)

            json_data = {
                "idx": idx, 
                "ts": stereo.ts,
                "h": stereo.h, 
                "w": stereo.w,
                "k": self._safe_list(stereo.k), 
                "baseline": stereo.baseline,
                "stereo_l_path": os.path.join("aria", "all_data", f"{idx:05d}", "stereo_l.png"),
                "stereo_r_path": os.path.join("aria", "all_data", f"{idx:05d}", "stereo_r.png")
            }

            with open(os.path.join(frame_dir, "aria_stereo.json"), 'w') as f:
                json.dump(json_data, f, indent=4)

    def save_aria_stereo_video_orig(self):
        stereos = []
        for idx in range(len(self.tss)):
            stereo = self.stereo[idx]
            stereo_l = stereo.stereo_l
            stereo_r = stereo.stereo_r
            stereos.append(np.hstack((stereo_l, stereo_r)))

        create_video_from_frames(
            frames=stereos, 
            save_path=os.path.join(self.mps_path, "aria", "aria_stereo_video_orig.mp4"),
            fps=self.fps
        )

class AriaStereoGenerator:
    def __init__(self, mps_path: str, vrs_provider: VrsDataProvider):
        self.mps_path = mps_path
        self.vrs_provider = vrs_provider
        self.device_calib = self.vrs_provider.get_device_calibration()
        self.out_w, self.out_h = TARGET_WIDTH, TARGET_HEIGHT
        self._init_aria_stereo()
        print(f"[Stereo] 初始化虚拟立体相机: Res=({self.out_w}x{self.out_h}), FOV={STEREO_FOV_DEG}°")

    def _init_aria_stereo(self):
        # 获取流 ID
        self.sid_left = self.vrs_provider.get_stream_id_from_label("camera-slam-left")
        self.sid_right = self.vrs_provider.get_stream_id_from_label("camera-slam-right")
        
        # 获取原始标定 (Fisheye)
        self.src_calib_left = self.device_calib.get_camera_calib("camera-slam-left")
        self.src_calib_right = self.device_calib.get_camera_calib("camera-slam-right")
        
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
        
        # 计算基线长度 (单位通常是米)
        self.baseline_m = np.linalg.norm(pos_r - pos_l)

        # 5. 构建虚拟相机内参 (Pinhole)
        # 使用 target_w 计算焦距，以保证水平 FOV 覆盖
        self.f_len = (self.out_w / 2.0) / math.tan(math.radians(STEREO_FOV_DEG / 2.0))

        # 记录下中心点，方便后续存入 DataClass
        self.cx = self.out_w / 2.0
        self.cy = self.out_h / 2.0
        
        # already reize image to target size
        self.dst_calib_left = calibration.get_linear_camera_calibration(self.out_w, self.out_h, self.f_len, "stereo-left", self.T_d_new_cl)
        self.dst_calib_right = calibration.get_linear_camera_calibration(self.out_w, self.out_h, self.f_len, "stereo-right", self.T_d_new_cr)

    def get_aria_stereo(self, tss: List[int]) -> Optional[AriaStereo]:
        aria_stereo = AriaStereo()
        for ts in tqdm(tss, desc="Processing Aria Stereo"):
            aria_stereo_data = self._get_aria_stereo_data(ts)
            if aria_stereo_data is not None: # 增加判空
                aria_stereo.tss.append(aria_stereo_data.ts)
                aria_stereo.stereo.append(aria_stereo_data)
        aria_stereo.fps = self._calculate_fps(aria_stereo.tss)
        aria_stereo.mps_path = self.mps_path
        return aria_stereo

    def _calculate_fps(self, tss: List[int]) -> int:
        if len(tss) < 2:
            return 10
        first_ts = tss[0]
        last_ts = tss[-1]
        duration_sec = (last_ts - first_ts) / 1e9
        fps = (len(tss) - 1) / duration_sec
        fps = int(Decimal(str(fps)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        return fps
    
    def _get_aria_stereo_data(self, ts: int) -> Optional[AriaStereoData]:
        idx_l = self.vrs_provider.get_index_by_time_ns(self.sid_left, ts, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
        idx_r = self.vrs_provider.get_index_by_time_ns(self.sid_right, ts, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
        
        if idx_l == -1 or idx_r == -1:
            return None
        
        data_l = self.vrs_provider.get_image_data_by_index(self.sid_left, idx_l)
        data_r = self.vrs_provider.get_image_data_by_index(self.sid_right, idx_r)
        
        img_l_raw = data_l[0].to_numpy_array()
        img_r_raw = data_r[0].to_numpy_array()
        
        # 1. 校正 (Rectification)
        img_l_rect = calibration.distort_by_calibration(img_l_raw, self.dst_calib_left, self.src_calib_left, InterpolationMethod.BILINEAR)
        img_r_rect = calibration.distort_by_calibration(img_r_raw, self.dst_calib_right, self.src_calib_right, InterpolationMethod.BILINEAR)
        
        # # 2. 旋转 90 度顺时针 （旋转 90 度会破坏极线对齐，只是为了显示好看，如果后面需要使用图像，那么应该注意修改回来）
        # img_l_final = np.rot90(img_l_rect, k=-1).copy()
        # img_r_final = np.rot90(img_r_rect, k=-1).copy()
        
        fx=self.f_len
        fy=self.f_len
        cx=self.cx
        cy=self.cy
        k = np.array([
            [fx,           0,            cx         ], 
            [0,            fy,           cy         ], 
            [0,            0,            1          ]
        ])

        return AriaStereoData(
                ts=ts, 
                stereo_l=img_l_rect, 
                stereo_r=img_r_rect, 
                h=self.out_h, 
                w=self.out_w,
                k=k,
                baseline=self.baseline_m
            )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    args = parser.parse_args()
    
    # init vrs_provider and mps_provider
    mps_path = args.mps_path
    vrs_path = os.path.join(mps_path, "sample.vrs")
    vrs_provider = data_provider.create_vrs_data_provider(vrs_path)
    mps_provider = MpsDataProvider(MpsDataPathsProvider(mps_path).get_data_paths())

    # get start_idx and end_idx
    hand_tracking_results_path = os.path.join(mps_path, "hand_tracking", "hand_tracking_results.csv")
    rgb_timestamps_ns = vrs_provider.get_timestamps_ns(vrs_provider.get_stream_id_from_label("camera-rgb"), TimeDomain.DEVICE_TIME)
    aria_hands_mps = mps.hand_tracking.read_hand_tracking_results(hand_tracking_results_path)
    total_rgb_frames = len(rgb_timestamps_ns)
    mps_not_detected_number = total_rgb_frames - len(aria_hands_mps)
    start_idx = mps_not_detected_number
    num_frames = len(aria_hands_mps)
    end_idx = start_idx + num_frames - 1
    
    # get aria_cam and tss
    aria_cam_generator = AriaCamGenerator(mps_path, vrs_provider, mps_provider)
    aria_cam = aria_cam_generator.get_aria_cam(start_idx, end_idx)
    tss = aria_cam.tss

    # get aria_stereo
    aria_stereo_generator = AriaStereoGenerator(mps_path, vrs_provider)
    aria_stereo = aria_stereo_generator.get_aria_stereo(tss)

    # save data
    aria_stereo.save_aria_stereo_json()
    aria_stereo.save_aria_stereo_video_orig()

# conda activate aria
# cd src
# python -m aria.AriaStereo --mps_path "./data/mps_open_cabinet_5_vrs/"