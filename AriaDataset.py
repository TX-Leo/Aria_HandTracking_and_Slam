# -*- coding: utf-8 -*-
# @FileName: AriaDataset.py
# @Description: Aria 数据集核心处理类，负责整合各个子模块(Cam, Nav, Hand, Stereo, PointCloud)并执行全流程处理。

import os
import cv2
import json
import math
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, get_worker_info

# Project Aria Tools 核心依赖库
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.mps.utils import filter_points_from_confidence
from projectaria_tools.core.sensor_data import TimeQueryOptions, TimeDomain
from projectaria_tools.core.vrs import extract_audio_track

# 自定义模块 imports
from utils import utils_media
from aria.AriaCam import AriaCam
from aria.AriaNav import AriaNav, AriaNavOptimizer, AriaNavEvaluator
from aria.AriaStereo import AriaStereo, AriaStereoGenerator
# from aria.AriaPhaseSegmenter import AriaPhaseSegmenter
from aria.AriaPhaseSegmenter_v2 import AriaPhaseSegmenter
from aria.AriaPointCloud import AriaPointCloudOps          
from aria.AriaHand import AriaHand, AriaHandOps                      

# -------------------------------------------------------------------------
# 全局配置参数 (Configuration)
# -------------------------------------------------------------------------

# Aria 去暗角 mask 文件路径
DEVIGNETTING_MASKS_PATH = os.path.join(os.path.dirname(__file__), "aria_devignetting_masks")

# =========================================================================
# [Data Structures] 复合数据结构定义
# =========================================================================

@dataclass
class AriaStruct:
    """
    [核心数据结构] 定义单帧的所有多模态数据。
    聚合了 Cam(RGB), Stereo(双目), Nav(导航), Hands(手部), Phase(阶段)。
    """
    idx: int
    ts: float
    cam: AriaCam = None
    stereo: AriaStereo = None  # Renamed from AriaStereoPair
    nav: AriaNav = None
    hands: List[AriaHand] = field(default_factory=list)
    phase: int = 0 

# =========================================================================
# [Main Class] Aria Dataset Pipeline
# =========================================================================

class AriaDataset(Dataset):
    """
    [数据集类] 处理 Meta Project Aria 数据 (VRS + MPS) 的主入口。
    实现了 PyTorch Dataset 接口，支持随机访问和批量导出。
    """
    def __init__(self, mps_path: str, vrs_path: str, hand_tracking_results_path: str, save_path: str):
        """
        初始化数据集，加载 Provider 和标定信息。
        """
        self.mps_path = mps_path
        self.vrs_path = vrs_path
        self.hand_tracking_results_path = hand_tracking_results_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        # 初始化数据提供者 (MPS & VRS)
        self.mps_data_provider, self.provider = self._init_providers()
        self.rgb_stream_id = self.provider.get_stream_id_from_label("camera-rgb")

        print("正在加载 MPS 手部追踪结果...")
        self.hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(
            self.hand_tracking_results_path
        )

        # 获取时间戳并对齐数据长度
        self.rgb_timestamps_ns = self.provider.get_timestamps_ns(self.rgb_stream_id, TimeDomain.DEVICE_TIME)
        # MPS 数据通常比 VRS 滞后或少一些，计算偏移量
        self.mps_not_detected_number = len(self.rgb_timestamps_ns) - len(self.hand_tracking_results)
        self.num_frames = min(len(self.rgb_timestamps_ns), len(self.hand_tracking_results))
        
        # 加载设备标定和去暗角 Mask
        self.device_calib = self.provider.get_device_calibration()
        self.device_calib.set_devignetting_mask_folder_path(DEVIGNETTING_MASKS_PATH)
        self.devignetting_mask = self.device_calib.load_devignetting_mask("camera-rgb")
        
        # [Stereo Setup] 初始化立体图像生成器
        # 获取 RGB 相机原始尺寸 (Landscape) 用于对齐分辨率
        rgb_calib_ref = self.device_calib.get_camera_calib("camera-rgb")
        rgb_w = int(rgb_calib_ref.get_image_size()[0])
        rgb_h = int(rgb_calib_ref.get_image_size()[1])
        
        print("初始化立体校正生成器 (Stereo Rectification) - 匹配 RGB 尺寸...")
        self.stereo_gen = AriaStereoGenerator(self.provider, self.device_calib, target_shape=(rgb_w, rgb_h))

        # 检查是否存在 Online Calibration (MPS 生成的更高精度标定)
        self.has_online_calibration = (
            self.mps_data_provider is not None
            and self.mps_data_provider.has_semidense_point_cloud()
        )

        # 计算帧率
        self.fps = self._calculate_fps()
        print(f"Aria 数据加载成功: {self.num_frames} 帧有效数据, FPS: {self.fps:.2f}")

        # 初始化数据容器
        self.point_cloud = None          
        self.trajectory_points_3d = []   
        self.nav_data_map = {} 
        self.nav_optimizer = AriaNavOptimizer(dt=1.0/self.fps)

        # 如果有高精度 MPS 数据，预加载点云和轨迹
        if self.has_online_calibration:
            self._load_point_cloud()
            # self._process_full_navigation_trajectory() # 可按需在 process_all 中调用

        self.all_aria_structs: List[AriaStruct] = []

    def _init_providers(self) -> Tuple[Optional[MpsDataProvider], VrsDataProvider]:
        """初始化 VRS 和 MPS 数据提供者"""
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
        """计算视频的平均帧率"""
        if self.num_frames < 2: return 30.0
        first_ts = self.rgb_timestamps_ns[self.mps_not_detected_number]
        last_ts = self.rgb_timestamps_ns[self.mps_not_detected_number + self.num_frames - 1]
        
        if self.has_online_calibration:
             first_ts = self.mps_data_provider.get_rgb_corrected_timestamp_ns(first_ts, TimeQueryOptions.CLOSEST)
             last_ts = self.mps_data_provider.get_rgb_corrected_timestamp_ns(last_ts, TimeQueryOptions.CLOSEST)
             
        duration_sec = (last_ts - first_ts) / 1e9
        return (self.num_frames - 1) / duration_sec if duration_sec > 0 else 30.0

    def _load_point_cloud(self):
        """加载 MPS 半稠密点云并进行置信度过滤"""
        print("加载环境点云...")
        pc = self.mps_data_provider.get_semidense_point_cloud()
        pc = filter_points_from_confidence(pc, 0.001, 0.15)
        self.point_cloud = np.stack([it.position_world for it in pc])

    def _rotation_matrix_to_yaw(self, R):
        """从旋转矩阵提取 Yaw 角 (假设 Z 轴向上)"""
        forward_vector = R[:, 0] 
        yaw = math.atan2(forward_vector[1], forward_vector[0])
        return yaw

    def _process_full_navigation_trajectory(self):
        """
        处理完整的导航轨迹：
        1. 从 MPS 提取原始 6D Pose。
        2. 投影到 2D 平面 (x, y, yaw)。
        3. 使用优化器平滑并生成机器人运动学轨迹 (v, w)。
        """
        print("正在处理导航轨迹 (Extraction & Retargeting)...")
        raw_timestamps = []
        raw_poses_2d = [] 
        raw_pos_3d_list = [] 
        start_idx = self.mps_not_detected_number
        end_idx = start_idx + self.num_frames

        for i in range(start_idx, end_idx):
            ts = self.rgb_timestamps_ns[i]
            # 获取矫正后的时间戳和 Pose
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
        
        # 调用优化器
        opt_result = self.nav_optimizer.optimize_trajectory(human_traj_np)
        
        robot_traj_np = opt_result['optimized_poses']
        robot_vels_np = opt_result['velocities']

        # 生成评估报告图表
        eval_plot_path = os.path.join(self.save_path, "nav_evaluation_report.png")
        AriaNavEvaluator.evaluate_and_plot(
            human_traj_np, robot_traj_np, robot_vels_np, 
            eval_plot_path, dt=1.0/self.fps
        )

        # 缓存结果
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

    def _save_navigation_trajectory(self):
        """保存导航轨迹为 CSV"""
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

    def _save_struct_to_json(self, s: AriaStruct):
        """保存单帧数据结构为 JSON，并保存图像"""
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

    def _extract_original_video(self, audio_path: str):
        """导出不带可视化的原始视频"""
        if not self.all_aria_structs: return
        frames_all = []
        for s in tqdm(self.all_aria_structs, desc="原视频帧"):
            img = s.cam.rgb.copy()
            frames_all.append(img)
        video_args = {"fps": self.fps, "audio_path": audio_path if os.path.exists(audio_path) else None}
        utils_media.create_video_from_frames(frames_all, os.path.join(self.save_path, "video_original.mp4"), **video_args)

    def _draw_trajectory_trail(self, img, current_idx, future_len=30, step=5, ground_offset=1.5):
        """
        在图像上绘制离散的未来轨迹点 (Waypoints)。
        
        Args:
            img: 当前帧图像
            current_idx: 当前帧索引
            future_len: 向未来预测多少帧 (默认30)
            step: 每隔多少帧画一个点 (默认5)
            ground_offset: 头部到地面的高度 (米)
        """
        if len(self.trajectory_points_3d) == 0: return img
        
        # 1. 获取当前相机参数
        s = self.all_aria_structs[current_idx]
        K = s.cam.k
        T_c2w = s.cam.c2w
        T_w2c = np.linalg.inv(T_c2w) # World -> Camera
        
        h_vis, w_vis = img.shape[:2]
        h_raw, w_raw = w_vis, h_vis 

        # 2. 确定未来窗口: [current, current + future_len]
        start_idx = current_idx
        end_idx = min(len(self.trajectory_points_3d), current_idx + future_len)
        
        if start_idx >= end_idx: return img

        # 3. 提取并采样点
        # 我们取 slice，然后用 [::step] 进行采样
        # 例如: indices = 0, 5, 10, 15, 20, 25
        points_world_subset = self.trajectory_points_3d[start_idx:end_idx:step].copy()
        
        if len(points_world_subset) == 0: return img

        # 4. 映射到地面
        points_world_subset[:, 2] -= ground_offset

        # 5. 投影: World -> Camera -> Pixel
        R = T_w2c[:3, :3]
        t = T_w2c[:3, 3]
        points_cam = (R @ points_world_subset.T).T + t
        
        # Camera -> Pixel (Raw Landscape)
        uv_homo = (K @ points_cam.T).T
        z_safe = uv_homo[:, 2]
        
        # 过滤掉相机背后的点
        valid_mask = z_safe > 0.1
        if not np.any(valid_mask): return img
        
        u_raw = uv_homo[:, 0] / (z_safe + 1e-6)
        v_raw = uv_homo[:, 1] / (z_safe + 1e-6)
        
        # 6. 旋转: Raw -> Vis
        u_vis = h_raw - 1 - v_raw
        v_vis = u_raw
        
        points_2d = np.stack([u_vis, v_vis], axis=1).astype(np.int32)
        
        # 7. 绘制圆点
        # 使用 HSV 彩虹色：近处(Red/Warm) -> 远处(Blue/Cold)
        num_pts = len(points_2d)
        
        for i in range(num_pts):
            if not valid_mask[i]: continue
            
            pt = tuple(points_2d[i])
            
            # 边界检查
            if not (0 <= pt[0] < w_vis and 0 <= pt[1] < h_vis): continue
            
            # 颜色计算: 0.0 (Near) -> 1.0 (Far)
            progress = i / max(1, num_pts - 1)
            
            # Hue: 0(Red) -> 120(Green) -> 160(Blue)
            # 让人感觉近处是热点(目标)，远处是冷色
            hue = int(160 * progress) 
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
            
            # 大小渐变: 近大远小
            radius = int(10 - 4 * progress) # 10 -> 6
            
            # 画实心圆
            cv2.circle(img, pt, radius, color_bgr, -1)
            # 画白色轮廓，增加对比度
            # cv2.circle(img, pt, radius + 1, (255, 255, 255), 2)

        return img
    
    def _create_visualization_videos(self, audio_path: str):
        """生成带骨架、速度、阶段信息的可视化视频"""
        if not self.all_aria_structs: return
        frames_all = []
        
        # For v1:
        # PHASE_COLORS = { 0: (255, 200, 0), 1: (0, 165, 255), 2: (0, 0, 255), 3: (0, 255, 0) }
        # PHASE_NAMES = {0: "0:APPROACH", 1: "1:UNLOCK", 2: "2:OPEN", 3: "3:TRAVERSE"}
        # For v2:
        PHASE_COLORS = { 0: (0, 255, 0), 1: (0, 0, 255) }
        PHASE_NAMES = { 0: "0:NAVIGATION", 1: "1:MANIPULATION"}

        for s in tqdm(self.all_aria_structs, desc="渲染视频帧"):
            img = s.cam.rgb.copy()

            # --- [新增] 绘制地面轨迹 ---
            # [修改] 绘制未来离散航点
            # future_len=30 (未来30帧), step=5 (每5帧取一个点), 共画 6 个点
            img = self._draw_trajectory_trail(img, s.idx, future_len=90, step=3, ground_offset=1.7)
            # -------------------------

            for hand in s.hands:
                # 调用AriaHandOps 进行绘制
                img = AriaHandOps.draw_axis(img, hand.palm_pose, s.cam.k, s.cam.d)
                img = AriaHandOps.draw_skeleton(img, hand)
            
            # 绘制底部黑色信息栏
            cv2.rectangle(img, (0, 0), (s.cam.w, 200), (0, 0, 0), -1)
            
            # 绘制速度信息
            cv2.putText(img, f"V: {s.nav.robot_v:.2f} m/s", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, f"W: {s.nav.robot_w:.2f} rad/s", (350, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 绘制阶段信息
            phase_id = s.phase
            phase_name = PHASE_NAMES.get(phase_id, "UNKNOWN")
            phase_color = PHASE_COLORS.get(phase_id, (255, 255, 255))
            cv2.putText(img, f"PHASE: {phase_name}", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, phase_color, 3)
            
            # 绘制手部状态
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

   
    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> AriaStruct:
        """
        获取单帧数据，包含图像处理、坐标变换和手部/导航数据组装。
        """
        if get_worker_info() is None:
            mps_dp, provider = self.mps_data_provider, self.provider
        else:
            mps_dp, provider = self._init_providers()

        # 1. 获取原始 RGB 图像
        rgb_idx = idx + self.mps_not_detected_number
        rgb_data = provider.get_image_data_by_index(self.rgb_stream_id, rgb_idx)
        rgb_img = np.copy(rgb_data[0].to_numpy_array())
        capture_timestamp_ns = rgb_data[1].capture_timestamp_ns
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # 2. 获取 Pose 和 Calibration
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

        # 3. 图像去畸变 (Rectification)
        rgb_linear_calib = calibration.get_linear_camera_calibration(
            int(rgb_calib.get_image_size()[0]), int(rgb_calib.get_image_size()[1]),
            rgb_calib.get_focal_lengths()[0], "camera-rgb", rgb_calib.get_transform_device_camera(),
        )

        rgb_img = calibration.devignetting(rgb_img, self.devignetting_mask).astype(np.uint8)
        rgb_img = calibration.distort_by_calibration(rgb_img, rgb_linear_calib, rgb_calib, InterpolationMethod.BILINEAR)
        
        # 4. 图像旋转 (Landscape -> Portrait)
        # RGB 顺时针旋转 90 度 (k=-1)
        h_orig, w_orig = rgb_img.shape[:2]
        rgb_img = np.rot90(rgb_img, k=-1)
        rgb_img = np.ascontiguousarray(rgb_img)
        h, w = rgb_img.shape[:2]

        # 5. 构建 AriaCam 对象
        fx, fy = rgb_linear_calib.get_focal_lengths()
        cx, cy = rgb_linear_calib.get_principal_point()
        k_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # 注意: 这里的 Pose 是 T_device_world，而 Cam 需要 c2w
        # 但通常 device_calib 中有 T_device_camera，需要组合
        # 这里简化处理，直接使用 rgb_pose (设备 Pose)
        
        aria_cam = AriaCam(rgb=rgb_img, h=h, w=w, k=k_matrix, d=np.zeros(5), c2w=rgb_pose)
        T_camera_to_device = rgb_calib.get_transform_device_camera().inverse().to_matrix()
        
        # 6. 获取校正后的立体图像 (Stereo)
        # Generator 会自动处理尺寸匹配和旋转
        img_rect_l, img_rect_r = self.stereo_gen.get_rectified_pair(capture_timestamp_ns)
        stereo_pair = AriaStereo() # Renamed from AriaStereoPair
        if img_rect_l is not None and img_rect_r is not None:
            stereo_pair.left_img = img_rect_l
            stereo_pair.right_img = img_rect_r
            stereo_pair.valid = True

        # 7. 处理手部数据 (使用AriaHandOps 类静态方法)
        hands_result = self.hand_tracking_results[idx]
        aria_hands = []

        # 处理左手
        left_hand =AriaHandOps.process_single_hand(
            hands_result.left_hand, T_camera_to_device, k_matrix, h_orig, w_orig, False
        )
        if left_hand: aria_hands.append(left_hand)

        # 处理右手
        right_hand =AriaHandOps.process_single_hand(
            hands_result.right_hand, T_camera_to_device, k_matrix, h_orig, w_orig, True
        )
        if right_hand: aria_hands.append(right_hand)

        return AriaStruct(
            idx=idx, ts=capture_timestamp_ns,
            cam=aria_cam, stereo=stereo_pair, nav=aria_nav, hands=aria_hands
        )

    def process_and_save_all(self):
        """
        执行完整的数据处理流水线并保存所有结果。
        包含：导航优化、点云融合、视频生成、阶段分割、数据导出。
        """
        # [Step 1] 计算导航轨迹
        if self.has_online_calibration and not self.nav_data_map:
            self._process_full_navigation_trajectory()

        # [Step 2] 保存点云数据 (调用 AriaPointCloudOps)
        # 保存融合点云 (环境 + 轨迹)
        AriaPointCloudOps.save_merged_point_cloud(
            self.point_cloud, self.trajectory_points_3d,
            os.path.join(self.save_path, "pc_scene_and_head_traj.ply")
        )
        # 仅保存环境点云
        AriaPointCloudOps.save_point_cloud_ply(
            self.point_cloud, os.path.join(self.save_path, "pc_scene.ply")
        )
        # 可视化 (可选，默认注释掉以支持 Headless 模式)
        # AriaPointCloudOps.visualize_scene_with_trajectory(self.point_cloud, self.trajectory_points_3d)
        
        # [Step 3] 提取原始音频
        audio_path = os.path.join(self.save_path, "original_audio.wav")
        extract_audio_track(self.vrs_path, audio_path)

        print("开始处理所有 Aria 帧...")
        temp_structs = []
        # 遍历所有帧，触发 __getitem__ 处理
        for i in tqdm(range(len(self)), desc="读取原始数据"):
            temp_structs.append(self[i])
            
        # [Step 4] 自动阶段分割
        print("正在进行自动阶段分割 (Auto Phase Segmentation)...")
        phases = AriaPhaseSegmenter.segment_phases(temp_structs)
        
        # [Step 5] 生成视频和导出数据
        print("数据处理完毕，开始生成视频...")
        self.all_aria_structs = temp_structs
        for i, s in enumerate(self.all_aria_structs):
            s.phase = phases[i]
            self._save_struct_to_json(s)
        self._save_navigation_trajectory() 
        self._extract_original_video(audio_path)
        self._create_visualization_videos(audio_path)
        
        print("所有任务完成！")