# -*- coding: utf-8 -*-
# @Time    : 2025/12/03
# @Author  : Leo
# @Project : Aria
# @FileName: aria.py.py
"""
aria.py (EMMA Ultimate Edition V7.2: Stereo Resized & Rotated)

本脚本实现了对 Meta Project Aria 数据 (VRS 和 MPS) 的全流程处理流水线。
旨在提取用于训练 EMMA (Egocentric Mobile Manipulation) 模型所需的多模态数据。

---------------------------------------------------------------------------
版本更新日志 (V7.2):
1. [Stereo Upgrade] SLAM 立体图像现在会强制缩放以匹配 RGB 图像的分辨率。
2. [Stereo Rotation] SLAM 图像现在会顺时针旋转 90 度，与 RGB 图像方向完全一致。
3. [Bug Fix] 修复了 get_index_by_time_ns 缺少的 TimeDomain 参数。
---------------------------------------------------------------------------
"""

import os
import cv2
import json
import numpy as np
import argparse
import open3d as o3d
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
from torch.utils.data import Dataset, get_worker_info
from scipy.optimize import minimize
from scipy.signal import medfilt

# Project Aria Tools 核心依赖库
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.mps.utils import filter_points_from_confidence
from projectaria_tools.core.sensor_data import TimeQueryOptions, TimeDomain
from projectaria_tools.core.vrs import extract_audio_track
from projectaria_tools.core.sophus import SE3, SO3

# 自定义工具库
from utils import utils_media

# -------------------------------------------------------------------------
# 全局配置参数 (Configuration)
# -------------------------------------------------------------------------

# Aria 去暗角 mask 文件路径
DEVIGNETTING_MASKS_PATH = os.path.join(os.path.dirname(__file__), "aria_devignetting_masks")

# [EgoZero] 抓取判定阈值 (单位: 米)
GRASP_THRESHOLD = 0.05 

# [EMMA Phase] 阶段判定阈值
PHASE_VEL_THRESH = 0.15 

# 推门修正窗口 (单位: 帧)
PHASE_PUSH_EXTEND_FRAMES = 30 

# [Stereo] 立体匹配视场角 (FOV)
# 我们保持 85 度 FOV，但像素分辨率将动态跟随 RGB 相机
STEREO_FOV_DEG = 85  

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
            target_shape: (width, height) 这里的宽高指 RGB 原始(旋转前)的宽高。
                          因为我们要在 rectification 之后才旋转，所以虚拟相机设为 landscape。
        """
        self.provider = provider
        self.device_calib = device_calib
        
        # 这里的 target_w/h 是为了匹配 RGB 图像的大小
        # 比如 RGB 是 1408(W) x 1408(H)，虚拟相机就设为这个大小。
        # 之后旋转完变成 1408(H) x 1408(W)，依然匹配。
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
        # 注意：这里使用 self.out_w (RGB的宽)，这样生成的图分辨率高
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

# =========================================================================
# [EMMA Core] 导航轨迹优化器 (Navigation Optimization)
# =========================================================================

class EmmaNavOptimizer:
    def __init__(self, dt: float = 0.1, 
                 lambda_pos: float = 32.0, 
                 lambda_yaw: float = 2.0, 
                 lambda_smooth: float = 1.0):
        self.dt = dt
        self.weights = {'pos': lambda_pos, 'yaw': lambda_yaw, 'smooth': lambda_smooth}
        self.v_limit = 1.2
        self.w_limit = 1.5

    def _wrap_angle(self, theta: float) -> float:
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def _kinematics_step(self, state: np.ndarray, v: float, w: float) -> np.ndarray:
        x, y, theta = state
        new_x = x + v * np.cos(theta) * self.dt
        new_y = y + v * np.sin(theta) * self.dt
        new_theta = self._wrap_angle(theta + w * self.dt)
        return np.array([new_x, new_y, new_theta])

    def optimize_trajectory(self, human_target_poses_2d: np.ndarray) -> Dict[str, np.ndarray]:
        N = len(human_target_poses_2d)
        if N < 2:
            return {'optimized_poses': human_target_poses_2d, 'velocities': np.zeros((N, 2))}
        
        print(f"[EMMA Nav] 开始优化导航轨迹，长度: {N} 帧...")
        x_init = np.zeros(2 * N)
        bounds = []
        for _ in range(N):
            bounds.append((-self.v_limit, self.v_limit))
            bounds.append((-self.w_limit, self.w_limit))

        def objective(z):
            cost = 0.0
            current_state = human_target_poses_2d[0].copy()
            vs = z[0::2]
            ws = z[1::2]
            for k in range(N):
                next_state = self._kinematics_step(current_state, vs[k], ws[k])
                target = human_target_poses_2d[k]
                dist_sq = np.sum((current_state[:2] - target[:2])**2)
                yaw_diff = self._wrap_angle(current_state[2] - target[2])
                yaw_sq = yaw_diff**2
                cost += self.weights['pos'] * dist_sq + self.weights['yaw'] * yaw_sq
                if k > 0:
                    dv = vs[k] - vs[k-1]
                    dw = ws[k] - ws[k-1]
                    cost += self.weights['smooth'] * (dv**2 + dw**2)
                current_state = next_state
            return cost

        res = minimize(objective, x_init, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'disp': False})
        
        optimized_vs = res.x[0::2]
        optimized_ws = res.x[1::2]
        optimized_path = []
        curr = human_target_poses_2d[0].copy()
        for k in range(N):
            optimized_path.append(curr)
            curr = self._kinematics_step(curr, optimized_vs[k], optimized_ws[k])
            
        return {
            'optimized_poses': np.array(optimized_path),
            'velocities': np.stack([optimized_vs, optimized_ws], axis=1)
        }

# =========================================================================
# [Quality Control] 导航质量评估
# =========================================================================

class NavigationEvaluator:
    @staticmethod
    def evaluate_and_plot(human_poses: np.ndarray, robot_poses: np.ndarray, 
                          robot_vels: np.ndarray, save_path: str, dt: float):
        print("[Nav Eval] 正在生成导航质量评估报告...")
        N = len(human_poses)
        time_axis = np.arange(N) * dt
        pos_error = np.linalg.norm(human_poses[:, :2] - robot_poses[:, :2], axis=1)
        mean_err = np.mean(pos_error)
        rmse = np.sqrt(np.mean(pos_error**2))
        
        print(f"  > 平均位置误差: {mean_err:.4f} m")
        print(f"  > RMSE: {rmse:.4f} m")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"EMMA Navigation Retargeting Evaluation\nRMSE: {rmse:.3f}m", fontsize=16)
        
        ax1 = axes[0, 0]
        ax1.plot(human_poses[:, 0], human_poses[:, 1], 'r--', label='Human (Raw)', alpha=0.5)
        ax1.plot(robot_poses[:, 0], robot_poses[:, 1], 'b-', label='Robot (Optimized)', linewidth=2)
        ax1.set_title("2D Trajectory (Top-Down)")
        ax1.legend(); ax1.grid(True); ax1.axis('equal')

        ax2 = axes[0, 1]
        ax2.plot(time_axis, robot_vels[:, 0], 'g-', label='Linear V (m/s)')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(time_axis, robot_vels[:, 1], 'orange', label='Angular W (rad/s)', alpha=0.7)
        ax2.set_title("Robot Velocity Commands")
        ax2.grid(True)

        ax3 = axes[1, 0]
        ax3.plot(time_axis, pos_error, 'k-', label='Pos Error (m)')
        ax3.set_title("Position Tracking Error")
        ax3.grid(True)

        ax4 = axes[1, 1]
        ax4.plot(time_axis, human_poses[:, 2], 'r--', label='Human Yaw')
        ax4.plot(time_axis, robot_poses[:, 2], 'b-', label='Robot Yaw', alpha=0.7)
        ax4.set_title("Yaw Angle Comparison")
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

# =========================================================================
# [Logic Core] 自动阶段分割 (Auto Phase Segmentation)
# =========================================================================

class AriaPhaseSegmenter:
    @staticmethod
    def segment_phases(structs: List['AriaStruct']) -> List[int]:
        if not structs: return []
        n = len(structs)
        phases = np.zeros(n, dtype=int)
        raw_grasp = np.array([max([h.grasp_state for h in s.hands] + [0]) for s in structs])
        smooth_grasp = medfilt(raw_grasp, kernel_size=11).astype(int)
        velocities = np.array([abs(s.nav.robot_v) for s in structs])
        
        grasp_indices = np.where(smooth_grasp == 1)[0]
        if len(grasp_indices) == 0:
            return phases.tolist()
            
        t_start = grasp_indices[0]
        t_end = grasp_indices[-1]
        phases[:t_start] = 0
        phases[t_end:] = 3
        
        for i in range(t_start, t_end + 1):
            if velocities[i] > PHASE_VEL_THRESH:
                phases[i] = 2 
            else:
                phases[i] = 1 
                
        extend_limit = min(t_end + PHASE_PUSH_EXTEND_FRAMES, n)
        for i in range(t_end + 1, extend_limit):
            if velocities[i] > PHASE_VEL_THRESH:
                phases[i] = 2
            else:
                break 
        
        phases = medfilt(phases, kernel_size=15).astype(int)
        return phases.tolist()

# =========================================================================
# [Data Structures] 数据结构定义
# =========================================================================

@dataclass
class AriaHand:
    is_right: bool
    confidence: float
    wrist_pose: np.ndarray = None
    palm_pose: np.ndarray = None
    hand_keypoints_3d: np.ndarray = None
    hand_keypoints_2d: np.ndarray = None
    grasp_state: int = 0

@dataclass
class AriaNav:
    agent_pos_3d: np.ndarray = None
    agent_pos_2d: np.ndarray = None
    robot_pos_2d: np.ndarray = None
    robot_v: float = 0.0
    robot_w: float = 0.0

@dataclass
class AriaCam:
    rgb: np.ndarray = None
    h: int = 0
    w: int = 0
    k: np.ndarray = None
    d: np.ndarray = None
    c2w: np.ndarray = None

@dataclass
class AriaStereoPair:
    left_img: np.ndarray = None
    right_img: np.ndarray = None
    valid: bool = False

@dataclass
class AriaStruct:
    idx: int
    ts: float
    cam: AriaCam = None
    stereo: AriaStereoPair = None
    nav: AriaNav = None
    hands: List[AriaHand] = field(default_factory=list)
    phase: int = 0 

# =========================================================================
# [Main Class] Aria Dataset Pipeline
# =========================================================================

class AriaDataset(Dataset):
    def __init__(self, mps_path: str, vrs_path: str, hand_tracking_results_path: str, save_path: str):
        self.mps_path = mps_path
        self.vrs_path = vrs_path
        self.hand_tracking_results_path = hand_tracking_results_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.mps_data_provider, self.provider = self._init_providers()
        self.rgb_stream_id = self.provider.get_stream_id_from_label("camera-rgb")

        print("正在加载 MPS 手部追踪结果...")
        self.hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(
            self.hand_tracking_results_path
        )

        self.rgb_timestamps_ns = self.provider.get_timestamps_ns(self.rgb_stream_id, TimeDomain.DEVICE_TIME)
        self.mps_not_detected_number = len(self.rgb_timestamps_ns) - len(self.hand_tracking_results)
        self.num_frames = min(len(self.rgb_timestamps_ns), len(self.hand_tracking_results))
        
        self.device_calib = self.provider.get_device_calibration()
        self.device_calib.set_devignetting_mask_folder_path(DEVIGNETTING_MASKS_PATH)
        self.devignetting_mask = self.device_calib.load_devignetting_mask("camera-rgb")
        
        # [Stereo Setup]
        # 获取 RGB 相机原始尺寸 (Landscape)
        rgb_calib_ref = self.device_calib.get_camera_calib("camera-rgb")
        rgb_w = int(rgb_calib_ref.get_image_size()[0])
        rgb_h = int(rgb_calib_ref.get_image_size()[1])
        
        print("初始化立体校正生成器 (Stereo Rectification) - 匹配 RGB 尺寸...")
        # 传入 RGB 原始尺寸，StereoGenerator 会创建同样大小的虚拟相机
        self.stereo_gen = AriaStereoGenerator(self.provider, self.device_calib, target_shape=(rgb_w, rgb_h))

        self.has_online_calibration = (
            self.mps_data_provider is not None
            and self.mps_data_provider.has_semidense_point_cloud()
        )

        self.fps = self._calculate_fps()
        print(f"Aria 数据加载成功: {self.num_frames} 帧有效数据, FPS: {self.fps:.2f}")

        self.point_cloud = None          
        self.trajectory_points_3d = []   
        self.nav_data_map = {} 
        self.nav_optimizer = EmmaNavOptimizer(dt=1.0/self.fps)

        if self.has_online_calibration:
            self._load_point_cloud()
            self._process_full_navigation_trajectory()

        self.all_aria_structs: List[AriaStruct] = []

    def _init_providers(self) -> Tuple[Optional[MpsDataProvider], VrsDataProvider]:
        try:
            mps_provider = MpsDataProvider(
                MpsDataPathsProvider(self.mps_path).get_data_paths()
            )
        except Exception as e:
            print(f"警告: 无法加载 MPS Data Provider ({e})，仅使用 VRS。")
            mps_provider = None
        vrs_provider = data_provider.create_vrs_data_provider(self.vrs_path)
        return mps_provider, vrs_provider

    def _calculate_fps(self) -> float:
        if self.num_frames < 2: return 30.0
        first_ts = self.rgb_timestamps_ns[self.mps_not_detected_number]
        last_ts = self.rgb_timestamps_ns[self.mps_not_detected_number + self.num_frames - 1]
        
        if self.has_online_calibration:
             first_ts = self.mps_data_provider.get_rgb_corrected_timestamp_ns(first_ts, TimeQueryOptions.CLOSEST)
             last_ts = self.mps_data_provider.get_rgb_corrected_timestamp_ns(last_ts, TimeQueryOptions.CLOSEST)
             
        duration_sec = (last_ts - first_ts) / 1e9
        return (self.num_frames - 1) / duration_sec if duration_sec > 0 else 30.0

    def _load_point_cloud(self):
        print("加载环境点云...")
        pc = self.mps_data_provider.get_semidense_point_cloud()
        pc = filter_points_from_confidence(pc, 0.001, 0.15)
        self.point_cloud = np.stack([it.position_world for it in pc])

    def _rotation_matrix_to_yaw(self, R):
        forward_vector = R[:, 0] 
        yaw = math.atan2(forward_vector[1], forward_vector[0])
        return yaw

    def _process_full_navigation_trajectory(self):
        print("正在处理导航轨迹 (Extraction & Retargeting)...")
        raw_timestamps = []
        raw_poses_2d = [] 
        raw_pos_3d_list = [] 
        start_idx = self.mps_not_detected_number
        end_idx = start_idx + self.num_frames

        for i in range(start_idx, end_idx):
            ts = self.rgb_timestamps_ns[i]
            corrected_ts = self.mps_data_provider.get_rgb_corrected_timestamp_ns(ts, TimeQueryOptions.CLOSEST)
            pose_obj = self.mps_data_provider.get_closed_loop_pose(corrected_ts, TimeQueryOptions.CLOSEST)
            if hasattr(pose_obj, "transform_world_device"):
                matrix = pose_obj.transform_world_device.to_matrix()
            else:
                matrix = pose_obj.to_matrix()
            
            pos_3d = matrix[:3, 3]
            raw_pos_3d_list.append(pos_3d)
            yaw = self._rotation_matrix_to_yaw(matrix[:3, :3])
            pose_2d = np.array([pos_3d[0], pos_3d[1], yaw])
            raw_timestamps.append(corrected_ts)
            raw_poses_2d.append(pose_2d)

        self.trajectory_points_3d = np.array(raw_pos_3d_list, dtype=np.float64)
        human_traj_np = np.array(raw_poses_2d)
        opt_result = self.nav_optimizer.optimize_trajectory(human_traj_np)
        
        robot_traj_np = opt_result['optimized_poses']
        robot_vels_np = opt_result['velocities']

        eval_plot_path = os.path.join(self.save_path, "nav_evaluation_report.png")
        NavigationEvaluator.evaluate_and_plot(
            human_traj_np, robot_traj_np, robot_vels_np, 
            eval_plot_path, dt=1.0/self.fps
        )

        for k, ts in enumerate(raw_timestamps):
            nav_obj = AriaNav(
                agent_pos_3d=raw_pos_3d_list[k],
                agent_pos_2d=human_traj_np[k],
                robot_pos_2d=robot_traj_np[k],
                robot_v=robot_vels_np[k][0],
                robot_w=robot_vels_np[k][1]
            )
            self.nav_data_map[ts] = nav_obj
        print(f"导航轨迹处理完毕。")

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> AriaStruct:
        if get_worker_info() is None:
            mps_dp, provider = self.mps_data_provider, self.provider
        else:
            mps_dp, provider = self._init_providers()

        rgb_idx = idx + self.mps_not_detected_number
        rgb_data = provider.get_image_data_by_index(self.rgb_stream_id, rgb_idx)
        rgb_img = np.copy(rgb_data[0].to_numpy_array())
        capture_timestamp_ns = rgb_data[1].capture_timestamp_ns
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        rgb_pose = np.eye(4)
        aria_nav = AriaNav(np.zeros(3), np.zeros(3), np.zeros(3), 0.0, 0.0)

        if self.has_online_calibration:
            corrected_ts = mps_dp.get_rgb_corrected_timestamp_ns(capture_timestamp_ns, TimeQueryOptions.CLOSEST)
            rgb_pose_obj = mps_dp.get_rgb_corrected_closed_loop_pose(corrected_ts, TimeQueryOptions.CLOSEST)
            if hasattr(rgb_pose_obj, "transform_world_device"):
                rgb_pose = rgb_pose_obj.transform_world_device.to_matrix()
            else:
                rgb_pose = rgb_pose_obj.to_matrix()

            if corrected_ts in self.nav_data_map:
                aria_nav = self.nav_data_map[corrected_ts]
            
            rgb_calib = mps_dp.get_online_calibration(corrected_ts, TimeQueryOptions.CLOSEST).get_camera_calib("camera-rgb")
        else:
            rgb_calib = provider.get_device_calibration().get_camera_calib("camera-rgb")

        rgb_linear_calib = calibration.get_linear_camera_calibration(
            int(rgb_calib.get_image_size()[0]), int(rgb_calib.get_image_size()[1]),
            rgb_calib.get_focal_lengths()[0], "camera-rgb", rgb_calib.get_transform_device_camera(),
        )

        rgb_img = calibration.devignetting(rgb_img, self.devignetting_mask).astype(np.uint8)
        rgb_img = calibration.distort_by_calibration(rgb_img, rgb_linear_calib, rgb_calib, InterpolationMethod.BILINEAR)
        
        # RGB 旋转 90 度
        h_orig, w_orig = rgb_img.shape[:2]
        rgb_img = np.rot90(rgb_img, k=-1)
        rgb_img = np.ascontiguousarray(rgb_img)
        h, w = rgb_img.shape[:2]

        fx, fy = rgb_linear_calib.get_focal_lengths()
        cx, cy = rgb_linear_calib.get_principal_point()
        k_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        T_camera_to_device = rgb_calib.get_transform_device_camera().inverse().to_matrix()

        aria_cam = AriaCam(rgb=rgb_img, h=h, w=w, k=k_matrix, d=np.zeros(5), c2w=rgb_pose)
        
        # [Stereo] 获取校正图像 (已自动旋转和缩放)
        img_rect_l, img_rect_r = self.stereo_gen.get_rectified_pair(capture_timestamp_ns)
        stereo_pair = AriaStereoPair()
        if img_rect_l is not None and img_rect_r is not None:
            stereo_pair.left_img = img_rect_l
            stereo_pair.right_img = img_rect_r
            stereo_pair.valid = True

        hands_result = self.hand_tracking_results[idx]
        aria_hands = []

        left_hand = self._process_single_hand(
            hands_result.left_hand, T_camera_to_device, k_matrix, h_orig, w_orig, False
        )
        if left_hand: aria_hands.append(left_hand)

        right_hand = self._process_single_hand(
            hands_result.right_hand, T_camera_to_device, k_matrix, h_orig, w_orig, True
        )
        if right_hand: aria_hands.append(right_hand)

        return AriaStruct(
            idx=idx, ts=capture_timestamp_ns,
            cam=aria_cam, stereo=stereo_pair, nav=aria_nav, hands=aria_hands
        )

    def _process_single_hand(self, hand_mps, T_camera_to_device, k_matrix, h_orig, w_orig, is_right) -> Optional[AriaHand]:
        if hand_mps is None: return None
        confidence = hand_mps.confidence
        if confidence <= 0.1: return None
        
        palm_normal = hand_mps.wrist_and_palm_normal_device.palm_normal_device
        palm_position = hand_mps.landmark_positions_device[int(mps.hand_tracking.HandLandmark.PALM_CENTER)]
        wrist_normal = hand_mps.wrist_and_palm_normal_device.wrist_normal_device
        wrist_position = hand_mps.landmark_positions_device[int(mps.hand_tracking.HandLandmark.WRIST)]

        thumb_tip = np.array(hand_mps.landmark_positions_device[int(mps.hand_tracking.HandLandmark.THUMB_FINGERTIP)])
        index_tip = np.array(hand_mps.landmark_positions_device[int(mps.hand_tracking.HandLandmark.INDEX_FINGERTIP)])
        
        distance = np.linalg.norm(thumb_tip - index_tip)
        grasp_state = 1 if distance < GRASP_THRESHOLD else 0

        palm_pose, wrist_pose = self._compute_wrist_and_palm_pose(
            palm_normal, palm_position, wrist_normal, wrist_position, confidence, T_camera_to_device
        )
        keypoints_3d_cam = self._transform_keypoints_to_camera(hand_mps, T_camera_to_device)
        keypoints_2d, _ = self._project_points_rotated(keypoints_3d_cam, k_matrix, h_orig, w_orig)

        return AriaHand(
            is_right=is_right, confidence=confidence,
            wrist_pose=wrist_pose, palm_pose=palm_pose,
            hand_keypoints_3d=keypoints_3d_cam, hand_keypoints_2d=keypoints_2d,
            grasp_state=grasp_state
        )

    def _compute_wrist_and_palm_pose(self, palm_n, palm_p, wrist_n, wrist_p, conf, T_cam_dev, threshold=0.5):
        if np.linalg.norm(wrist_n) == 0 or np.linalg.norm(palm_n) == 0: return None, None
        T_opencv_to_aria = np.eye(4)
        T_opencv_to_aria[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        palm_wrist_vec = palm_p - wrist_p
        palm_wrist_vec /= (np.linalg.norm(palm_wrist_vec) + 1e-6)
        wrist_n = wrist_n / (np.linalg.norm(wrist_n) + 1e-6)
        palm_n = palm_n / (np.linalg.norm(palm_n) + 1e-6)

        def build_rotation_matrix(pos, normal, forward_vec):
            z_axis = normal
            y_axis = forward_vec
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= (np.linalg.norm(x_axis) + 1e-6)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= (np.linalg.norm(y_axis) + 1e-6)
            mat = np.eye(4)
            mat[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
            mat[:3, 3] = pos
            return mat

        palm_mat = build_rotation_matrix(palm_p, palm_n, palm_wrist_vec)
        wrist_mat = build_rotation_matrix(wrist_p, wrist_n, palm_wrist_vec)
        return T_opencv_to_aria @ T_cam_dev @ palm_mat, T_opencv_to_aria @ T_cam_dev @ wrist_mat

    def _transform_keypoints_to_camera(self, hand, T_cam_dev):
        R = T_cam_dev[:3, :3]
        t = T_cam_dev[:3, 3]
        kpts_dev = np.array(hand.landmark_positions_device)
        kpts_cam = np.einsum("ij, nj -> ni", R, kpts_dev) + t
        return kpts_cam

    def _project_points_rotated(self, points_cam, k, h_orig, w_orig):
        z = points_cam[:, 2]
        valid_mask = z > 1e-3
        if not np.any(valid_mask):
            return np.zeros((len(points_cam), 2)), np.zeros(len(points_cam), dtype=bool)

        homo_pix = (k @ points_cam.T).T
        u = homo_pix[:, 0] / homo_pix[:, 2]
        v = homo_pix[:, 1] / homo_pix[:, 2]
        
        u_new = h_orig - 1 - v
        v_new = u
        projected = np.stack((u_new, v_new), axis=-1)
        
        h_new, w_new = w_orig, h_orig
        in_bounds = (0 <= u_new) & (u_new < h_new) & (0 <= v_new) & (v_new < w_new)
        return projected, valid_mask & in_bounds

    def process_and_save_all(self):
        self.save_merged_point_cloud()
        self.visualize_scene_with_trajectory()
        self.save_point_cloud_ply(visualize=False)
        
        audio_path = os.path.join(self.save_path, "original_audio.wav")
        extract_audio_track(self.vrs_path, audio_path)

        print("开始处理所有 Aria 帧...")
        temp_structs = []
        for i in tqdm(range(len(self)), desc="读取原始数据"):
            temp_structs.append(self[i])
            
        print("正在进行自动阶段分割 (Auto Phase Segmentation)...")
        phases = AriaPhaseSegmenter.segment_phases(temp_structs)
        
        self.all_aria_structs = temp_structs
        for i, s in enumerate(self.all_aria_structs):
            s.phase = phases[i]
            self.save_struct_to_json(s)

        print("数据处理完毕，开始生成视频...")
        self.create_visualization_videos(audio_path)
        self.save_navigation_trajectory() 
        print("所有任务完成！")

    def save_merged_point_cloud(self):
        if self.point_cloud is None or len(self.trajectory_points_3d) == 0: return
        pcd_scene = o3d.geometry.PointCloud()
        pcd_scene.points = o3d.utility.Vector3dVector(self.point_cloud)
        pcd_scene.paint_uniform_color([0.6, 0.6, 0.6])
        pcd_traj = o3d.geometry.PointCloud()
        pcd_traj.points = o3d.utility.Vector3dVector(self.trajectory_points_3d.astype(np.float64))
        pcd_traj.paint_uniform_color([0.0, 1.0, 0.0])
        pcd_merged = pcd_scene + pcd_traj
        out_path = os.path.join(self.save_path, "pc_scene_and_head_traj.ply")
        o3d.io.write_point_cloud(out_path, pcd_merged)
        print(f"融合点云已保存至: {out_path}")

    def visualize_scene_with_trajectory(self):
        if self.point_cloud is None: return
        geometries = []
        pcd_scene = o3d.geometry.PointCloud()
        pcd_scene.points = o3d.utility.Vector3dVector(self.point_cloud)
        pcd_scene.paint_uniform_color([0.6, 0.6, 0.6]) 
        geometries.append(pcd_scene)
        if len(self.trajectory_points_3d) > 0:
            pcd_traj = o3d.geometry.PointCloud()
            pcd_traj.points = o3d.utility.Vector3dVector(self.trajectory_points_3d.astype(np.float64))
            pcd_traj.paint_uniform_color([0, 1, 0]) 
            geometries.append(pcd_traj)
            o3d.io.write_point_cloud(os.path.join(self.save_path, "pc_head_traj.ply"), pcd_traj)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        geometries.append(axes)
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Aria EMMA Trajectory", width=1280, height=720)
            for geom in geometries: vis.add_geometry(geom)
            opt = vis.get_render_option()
            if opt is None: vis.destroy_window(); return
            opt.point_size = 5.0
            opt.background_color = np.asarray([0.1, 0.1, 0.1])
            vis.run()
            vis.destroy_window()
        except Exception: pass

    def save_point_cloud_ply(self, visualize: bool = False):
        if self.point_cloud is None: return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        o3d.io.write_point_cloud(os.path.join(self.save_path, "pc_scene.ply"), pcd)

    def save_struct_to_json(self, s: AriaStruct):
        frame_dir = os.path.join(self.save_path, "all_data", f"{s.idx:05d}")
        os.makedirs(frame_dir, exist_ok=True)
        rgb_path = os.path.join(frame_dir, "rgb.png")
        cv2.imwrite(rgb_path, s.cam.rgb)

        stereo_l_path = None
        stereo_r_path = None
        if s.stereo.valid:
            stereo_l_path = os.path.join(frame_dir, "stereo_left_rect.png")
            stereo_r_path = os.path.join(frame_dir, "stereo_right_rect.png")
            cv2.imwrite(stereo_l_path, s.stereo.left_img)
            cv2.imwrite(stereo_r_path, s.stereo.right_img)

        def safe_list(arr): return arr.tolist() if isinstance(arr, np.ndarray) else arr

        data = {
            "idx": s.idx, "ts": s.ts,
            "phase": int(s.phase),
            "cam": {
                "h": s.cam.h, "w": s.cam.w,
                "k": safe_list(s.cam.k), "d": safe_list(s.cam.d), "c2w": safe_list(s.cam.c2w),
                "rgb_path": rgb_path
            },
            "stereo": {
                "valid": s.stereo.valid,
                "left_path": stereo_l_path,
                "right_path": stereo_r_path
            },
            "rgb_path": rgb_path, "hands": [],
            "nav": {
                "agent_pos_3d": safe_list(s.nav.agent_pos_3d),
                "agent_pos_2d": safe_list(s.nav.agent_pos_2d),
                "robot_pos_2d": safe_list(s.nav.robot_pos_2d),
                "robot_v": float(s.nav.robot_v),
                "robot_w": float(s.nav.robot_w)
            }
        }
        for hand in s.hands:
            data["hands"].append({
                "is_right": hand.is_right, "confidence": hand.confidence,
                "wrist_pose": safe_list(hand.wrist_pose), "palm_pose": safe_list(hand.palm_pose),
                "keypoints_3d": safe_list(hand.hand_keypoints_3d), "keypoints_2d": safe_list(hand.hand_keypoints_2d),
                "grasp_state": hand.grasp_state
            })
        with open(os.path.join(frame_dir, "data.json"), 'w') as f:
            json.dump(data, f, indent=4)

    def create_visualization_videos(self, audio_path: str):
        if not self.all_aria_structs: return
        frames_all = []
        
        PHASE_COLORS = {
            0: (255, 200, 0),   
            1: (0, 165, 255),   
            2: (0, 0, 255),     
            3: (0, 255, 0)      
        }
        PHASE_NAMES = {0: "0:APPROACH", 1: "1:UNLOCK", 2: "2:OPEN", 3: "3:TRAVERSE"}

        for s in tqdm(self.all_aria_structs, desc="渲染视频帧"):
            img = s.cam.rgb.copy()
            for hand in s.hands:
                img = self._draw_axis(img, hand.palm_pose, s.cam.k, s.cam.d)
                img = self._draw_hand_skeleton(img, hand)
            
            cv2.rectangle(img, (0, 0), (s.cam.w, 200), (0, 0, 0), -1)
            
            cv2.putText(img, f"V: {s.nav.robot_v:.2f} m/s", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, f"W: {s.nav.robot_w:.2f} rad/s", (350, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            phase_color = PHASE_COLORS.get(s.phase, (255, 255, 255))
            cv2.putText(img, f"PHASE: {PHASE_NAMES.get(s.phase, 'UNKNOWN')}", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, phase_color, 3)
            
            x_pos_base = 20
            for hand in s.hands:
                side = "R" if hand.is_right else "L"
                state = "CLOSED" if hand.grasp_state == 1 else "OPEN"
                col = (0, 0, 255) if hand.grasp_state == 1 else (255, 255, 0)
                x_pos = 350 if hand.is_right else 20
                cv2.putText(img, f"{side}: {state}", (x_pos, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                
            frames_all.append(img)
            
        video_args = {"fps": self.fps, "audio_path": audio_path if os.path.exists(audio_path) else None}
        utils_media.create_video_from_frames(frames_all, os.path.join(self.save_path, "video_all.mp4"), **video_args)

    def _draw_axis(self, img, pose, k, d):
        if pose is None: return img
        r_vec, _ = cv2.Rodrigues(pose[:3, :3])
        t_vec = pose[:3, 3]
        img_pts, _ = cv2.projectPoints(np.float32([[0.05,0,0],[0,0.05,0],[0,0,0.05],[0,0,0]]), r_vec, t_vec, k, d)
        img_pts = [tuple(pt.ravel().astype(int)) for pt in img_pts]
        origin = img_pts[3]
        if 0 <= origin[0] < img.shape[1] and 0 <= origin[1] < img.shape[0]:
            cv2.line(img, origin, img_pts[0], (0, 0, 255), 3)
            cv2.line(img, origin, img_pts[1], (0, 255, 0), 3)
            cv2.line(img, origin, img_pts[2], (255, 0, 0), 3)
        return img

    def _draw_hand_skeleton(self, img, hand: AriaHand):
        kpts_2d = hand.hand_keypoints_2d
        if kpts_2d is None: return img
        is_grasp = (hand.grasp_state == 1)
        main_color = (0, 0, 255) if is_grasp else (255, 255, 255)
        for conn in mps.hand_tracking.kHandJointConnections:
            pt1 = tuple(kpts_2d[int(conn[0])].astype(int))
            pt2 = tuple(kpts_2d[int(conn[1])].astype(int))
            if pt1 != (0,0) and pt2 != (0,0): cv2.line(img, pt1, pt2, main_color, 2)
        for i, pt in enumerate(kpts_2d):
            pt_tuple = tuple(pt.astype(int))
            if pt_tuple == (0,0): continue
            pt_col = main_color if is_grasp else ((255,255,0) if i>0 else (150,150,150))
            cv2.circle(img, pt_tuple, 5, pt_col, -1)
        return img
    
    def save_navigation_trajectory(self):
        if not self.all_aria_structs: return
        out_csv_path = os.path.join(self.save_path, "nav_trajectory.csv")
        rows = []
        for s in self.all_aria_structs:
            x, y, yaw = s.nav.robot_pos_2d
            v, w = s.nav.robot_v, s.nav.robot_w
            rows.append(f"{s.idx},{s.ts},{x:.6f},{y:.6f},{yaw:.6f},{v:.6f},{w:.6f},{s.phase}")
        with open(out_csv_path, "w") as f:
            f.write("idx,timestamp_ns,robot_x,robot_y,robot_yaw,v,w,phase\n")
            f.write("\n".join(rows))
        print("导航轨迹保存完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True)
    args = parser.parse_args()
    
    mps_root = args.mps_path
    vrs_file = os.path.join(mps_root, "sample.vrs")
    if not os.path.exists(vrs_file):
        files = [f for f in os.listdir(mps_root) if f.endswith('.vrs')]
        vrs_file = os.path.join(mps_root, files[0]) if files else ""
    
    AriaDataset(mps_root, vrs_file, os.path.join(mps_root, "hand_tracking/hand_tracking_results.csv"), os.path.join(mps_root, "aria")).process_and_save_all()

    # python aria.py --mps_path "./test_data/mps_TEST_vrs/"