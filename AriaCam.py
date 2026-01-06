# -*- coding: utf-8 -*-
# @FileName: AriaCam.py

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Optional
from decimal import Decimal, ROUND_HALF_UP
import argparse
import subprocess

from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeQueryOptions, TimeDomain
from projectaria_tools.core.image import InterpolationMethod

from utils.utils_media import create_video_from_frames

TARGET_WIDTH = 640
TARGET_HEIGHT = 640

@dataclass
class AriaCamData:
    ts: int = 0             # timestamp in ns
    rgb: np.ndarray = None  # RGB 图像数据 (H, W, 3)
    h: int = 0              # 图像高度
    w: int = 0              # 图像宽度
    k: np.ndarray = None    # 内参矩阵 (3, 3)
    d: np.ndarray = None    # 畸变系数
    c2w: np.ndarray = None  # 相机到世界的变换矩阵 (4, 4) Pose
    c2d: np.ndarray = None  # 相机到device的变换矩阵 (4, 4) Pose
    d2w: np.ndarray = None  # device到世界的变换矩阵 (4, 4) Pose
    # c2w=d2w@c2d

@dataclass
class AriaCam:
    tss: List[int] = field(default_factory=list)
    cam: List[AriaCamData] = field(default_factory=list)
    
    fps: int = 0
    first_ts: int = 0
    h: int = 0
    w: int = 0
    k: np.ndarray = None
    d: np.ndarray = None
    
    mps_path: str = None

    def __len__(self):
        return len(self.tss)
    
    @staticmethod
    def _safe_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else arr
    
    def save_aria_cam_json(self):
        for idx in range(len(self.tss)):
            frame_dir = os.path.join(self.mps_path, "aria", "all_data", f"{idx:05d}")
            os.makedirs(frame_dir, exist_ok=True)
            rgb_path = os.path.join(frame_dir, "rgb.png")

            cam = self.cam[idx]
            cv2.imwrite(rgb_path, cam.rgb)
        
            json_data = {
                "idx": idx, 
                "ts": cam.ts,
                "h": cam.h, 
                "w": cam.w,
                "k": self._safe_list(cam.k), 
                "d": self._safe_list(cam.d), 
                "c2w": self._safe_list(cam.c2w),
                "c2d": self._safe_list(cam.c2d),
                "d2w": self._safe_list(cam.d2w),
                "rgb_path": os.path.join("aria", "all_data", f"{idx:05d}", "rgb.png"),
                "fps": self.fps
            }
            with open(os.path.join(frame_dir, "aria_cam.json"), 'w') as f:
                json.dump(json_data, f, indent=4)

        self._save_aria_cam_config_json()

    def _save_aria_cam_config_json(self):
        save_path = os.path.join(self.mps_path, "aria", "aria_cam_config.json")
        summary_data = {
            "total_frames": len(self),
            "fps": self.fps,
            "first_ts": self.first_ts,
            "h": self.h,
            "w": self.w,
            "k": self._safe_list(self.k),
            "d": self._safe_list(self.d)
        }
        with open(save_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"aria_cam_config.json saved to: {save_path}")

    def save_aria_cam_video_orig(self):
        rgbs = []
        for idx in range(len(self.tss)):
            cam = self.cam[idx]
            rgbs.append(cam.rgb)

        create_video_from_frames(
            frames=rgbs, 
            save_path=os.path.join(self.mps_path, "aria", "aria_cam_video_orig.mp4"),
            fps=self.fps
        )

class AriaCamGenerator():
    def __init__(self, mps_path: str, vrs_provider: VrsDataProvider, mps_provider: MpsDataProvider):
        self.mps_path = mps_path
        self.vrs_provider = vrs_provider
        self.mps_provider = mps_provider
        self.device_calib = self.vrs_provider.get_device_calibration()
        self.rgb_stream_id = self.vrs_provider.get_stream_id_from_label("camera-rgb")
        self.device_calib.set_devignetting_mask_folder_path(self._get_devignetting_mask_folder_path())
        self.devignetting_mask = self.device_calib.load_devignetting_mask("camera-rgb")
        self.rgb_calib_ref = self.device_calib.get_camera_calib("camera-rgb")
    
    def get_aria_cam(self, start_idx: int, end_idx: int) -> Optional[AriaCam]:
        aria_cam = AriaCam()
        for idx in tqdm(range(start_idx, end_idx + 1), desc="Processing Aria Cam"):
            aria_cam_data = self._get_aria_cam_data(idx)
            aria_cam.tss.append(aria_cam_data.ts)
            aria_cam.cam.append(aria_cam_data)
        aria_cam.fps = self._calculate_fps(aria_cam.tss)
        aria_cam.h = aria_cam.cam[0].h
        aria_cam.w = aria_cam.cam[0].w
        aria_cam.k = aria_cam.cam[0].k
        aria_cam.d = aria_cam.cam[0].d
        aria_cam.first_ts = aria_cam.cam[0].ts
        aria_cam.mps_path = self.mps_path
        return aria_cam
    
    def _get_devignetting_mask_folder_path(self) -> str:
        devignetting_mask_folder_path = os.path.join(os.path.dirname(__file__), "aria_devignetting_masks")
        if not os.path.exists(devignetting_mask_folder_path):
            print(f"Directory {devignetting_mask_folder_path} not found. Downloading devignetting masks...")
            try:
                download_cmd = 'curl -L -o devignetting_masks.zip "https://www.projectaria.com/async/sample/download/?bucket=core&filename=devignetting_masks_bin.zip"'
                subprocess.run(download_cmd, shell=True, check=True)
                unzip_cmd = f'unzip -o ./devignetting_masks.zip -d {devignetting_mask_folder_path}'
                subprocess.run(unzip_cmd, shell=True, check=True)
                os.remove("./devignetting_masks.zip")
                print("Successfully downloaded and extracted devignetting masks.")
            except Exception as e:
                print(f"Error downloading masks: {e}")
        return devignetting_mask_folder_path
    
    def _calculate_fps(self, tss: List[int]) -> int:
        if len(tss) < 2:
            return 10
        first_ts = tss[0]
        last_ts = tss[-1]
        duration_sec = (last_ts - first_ts) / 1e9
        fps = (len(tss) - 1) / duration_sec
        fps = int(Decimal(str(fps)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        return fps
            
    def _get_aria_cam_data(self, idx: int) -> Optional[AriaCamData]:
        rgb_img, rgb_ts = self._get_orig_rgb(idx)
        rgb_img = self._devignette_img(rgb_img)
        rgb_img, rgb_calib, rgb_linear_calib, fx_orig, fy_orig, cx_orig, cy_orig, h_orig, w_orig = self._rectificate_img(rgb_img, rgb_ts)
        rgb_img, fx_rot, fy_rot, cx_rot, cy_rot, h_rot, w_rot = self._rotate_img(rgb_img, fx_orig, fy_orig, cx_orig, cy_orig, h_orig, w_orig)
        rgb_img, fx_rsz, fy_rsz, cx_rsz, cy_rsz, h_rsz, w_rsz = self._resize_img(rgb_img, fx_rot, fy_rot, cx_rot, cy_rot, h_rot, w_rot)
        
        ts = rgb_ts
        rgb = rgb_img
        h, w = h_rsz, w_rsz
        k, d = self._get_intrinsics(fx_rsz, fy_rsz, cx_rsz, cy_rsz)
        c2w, c2d, d2w = self._get_extrinsics(rgb_ts, rgb_calib)
        return AriaCamData(ts=ts,rgb=rgb,h=h,w=w,k=k,d=d,c2w=c2w,c2d=c2d,d2w=d2w)

    def _get_orig_rgb(self, idx: int):
        # get rgb_img
        rgb_data = self.vrs_provider.get_image_data_by_index(self.rgb_stream_id, idx)
        rgb_img = np.copy(rgb_data[0].to_numpy_array())
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # get rgb_ts
        rgb_ts = rgb_data[1].capture_timestamp_ns
        rgb_ts = self.mps_provider.get_rgb_corrected_timestamp_ns(rgb_ts, TimeQueryOptions.CLOSEST)
        return rgb_img, rgb_ts

    def _devignette_img(self, rgb_img):
        rgb_img = calibration.devignetting(rgb_img, self.devignetting_mask).astype(np.uint8)
        return rgb_img
    
    def _rectificate_img(self, rgb_img, rgb_ts):
        rgb_calib = self.mps_provider.get_online_calibration(rgb_ts, TimeQueryOptions.CLOSEST).get_camera_calib("camera-rgb")
        # Fisheye（鱼眼）转换为 Pinhole（针孔）模型
        rgb_linear_calib = calibration.get_linear_camera_calibration(
            int(rgb_calib.get_image_size()[0]), int(rgb_calib.get_image_size()[1]),
            rgb_calib.get_focal_lengths()[0], "camera-rgb", rgb_calib.get_transform_device_camera(),
        )
        rgb_img = calibration.distort_by_calibration(rgb_img, rgb_linear_calib, rgb_calib, InterpolationMethod.BILINEAR)

        fx_orig, fy_orig = rgb_linear_calib.get_focal_lengths()
        cx_orig, cy_orig = rgb_linear_calib.get_principal_point()
        h_orig, w_orig = rgb_img.shape[:2]

        return rgb_img, rgb_calib, rgb_linear_calib, fx_orig, fy_orig, cx_orig, cy_orig, h_orig, w_orig
    
    def _rotate_img(self, rgb_img, fx_orig, fy_orig, cx_orig, cy_orig, h_orig, w_orig):
        rgb_img = np.rot90(rgb_img, k=-1)
        rgb_img = np.ascontiguousarray(rgb_img)

        fx_rot, fy_rot = fy_orig, fx_orig
        cx_rot, cy_rot = h_orig - 1 - cy_orig, cx_orig
        h_rot, w_rot = w_orig, h_orig

        return rgb_img, fx_rot, fy_rot, cx_rot, cy_rot, h_rot, w_rot

    def _resize_img(self, rgb_img, fx_rot, fy_rot, cx_rot, cy_rot, h_rot, w_rot):
        rgb_img = cv2.resize(rgb_img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

        scale_h = TARGET_HEIGHT / h_rot
        scale_w = TARGET_WIDTH / w_rot

        fx_rsz, fy_rsz = fx_rot*scale_w, fy_rot*scale_h
        cx_rsz, cy_rsz = cx_rot*scale_w, cy_rot*scale_h
        h_rsz, w_rsz = TARGET_HEIGHT, TARGET_WIDTH

        return rgb_img, fx_rsz, fy_rsz, cx_rsz, cy_rsz, h_rsz, w_rsz

    def _get_intrinsics(self, fx, fy, cx, cy):
        k = np.array([
            [fx,           0,            cx         ], 
            [0,            fy,           cy         ], 
            [0,            0,            1          ]
        ])
        d = np.zeros(5)

        return k, d
    
    def _get_extrinsics(self, rgb_ts, rgb_calib):
        pose = self.mps_provider.get_closed_loop_pose(rgb_ts, TimeQueryOptions.CLOSEST)
        if not pose:
            return np.eye(4), np.eye(4), np.eye(4)
        
        # T_d2w
        T_d2w = pose.transform_world_device.to_matrix()
        
        # T_c2d
        T_c2d = rgb_calib.get_transform_device_camera().to_matrix()
        R_z_90 = np.array([
            [0, 1,  0,  0],
            [-1,  0,  0,  0],
            [0,  0,  1,  0],
            [0,  0,  0,  1]
        ])
        T_c2d = T_c2d @ R_z_90

        # T_c2w
        T_c2w = T_d2w @ T_c2d
        return T_c2w, T_c2d, T_d2w
    
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
    
    # get aria_cam
    aria_cam_generator = AriaCamGenerator(mps_path, vrs_provider, mps_provider)
    aria_cam = aria_cam_generator.get_aria_cam(start_idx, end_idx)
    tss = aria_cam.tss

    # save data
    aria_cam.save_aria_cam_json()
    aria_cam.save_aria_cam_video_orig()

# conda activate aria
# cd src
# python -m aria.AriaCam --mps_path "./data/open_cabinet_0/mps_open_cabinet_0_005_vrs/"