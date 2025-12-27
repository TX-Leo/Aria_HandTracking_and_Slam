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
from typing import List, Tuple, Optional, Dict
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
from aria.AriaHand import AriaJointAngle, AriaHand, AriaHandOps                      

# -------------------------------------------------------------------------
# 全局配置参数 (Configuration)
# -------------------------------------------------------------------------
# 目标图像尺寸: (宽, 高)
TARGET_WIDTH = 640
TARGET_HEIGHT = 640

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
            corrected_ts = self.mps_data_provider.get_rgb_corrected_timestamp_ns(ts, TimeQueryOptions.CLOSEST)
            T_d2w = self.mps_data_provider.get_closed_loop_pose(corrected_ts, TimeQueryOptions.CLOSEST).transform_world_device.to_matrix()
            pos_3d = T_d2w[:3, 3]
            raw_pos_3d_list.append(pos_3d)
            yaw = self._rotation_matrix_to_yaw(T_d2w[:3, :3])
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
        eval_plot_path = os.path.join(self.save_path, "aria_nav_eval.png")
        AriaNavEvaluator.evaluate_and_plot(
            human_traj_np, robot_traj_np, robot_vels_np, 
            eval_plot_path, dt=1.0/self.fps
        )
        
        # 手动修改robot nav所有结果的符号，让其数值更加易读
        robot_traj_np = -robot_traj_np  # 修正 x, y, theta
        robot_vels_np = -robot_vels_np  # 修正 v, w
        start_pose = robot_traj_np[0] # 获取第一帧作为基准
        # 缓存结果
        for k, ts in enumerate(raw_timestamps):
            # 计算增量 Delta
            curr_pose = robot_traj_np[k]
            # 计算绝对角度 (弧度 -> 度)
            abs_theta_deg = math.degrees(curr_pose[2])

            # 计算增量角度 (必须在弧度下做归一化，防止 359度到 1度跳变)
            dt_raw_rad = curr_pose[2] - start_pose[2]
            dt_norm_rad = math.atan2(math.sin(dt_raw_rad), math.cos(dt_raw_rad))
            delta_theta_deg = math.degrees(dt_norm_rad)

            dx = curr_pose[0] - start_pose[0]
            dy = curr_pose[1] - start_pose[1]

            nav_obj = AriaNav(
                agent_pos_3d = raw_pos_3d_list[k],
                agent_pos_2d = human_traj_np[k],
                robot_pos_2d = np.array([curr_pose[0], curr_pose[1], abs_theta_deg]),# 存入增量 (单位：m, m, deg)
                robot_v = robot_vels_np[k][0],
                robot_w = robot_vels_np[k][1],
                delta_x = dx,
                delta_y = dy,
                delta_theta = delta_theta_deg
            )
            self.nav_data_map[ts] = nav_obj
        print(f"导航轨迹处理完毕。")

    def _save_navigation_trajectory(self):
        """保存导航轨迹为 CSV"""
        if not self.all_aria_structs: return
        out_csv_path = os.path.join(self.save_path, "aria_nav_traj.csv")
        rows = []
        for s in self.all_aria_structs:
            x, y, yaw = s.nav.robot_pos_2d
            v, w = s.nav.robot_v, s.nav.robot_w
            dx, dy, dt = s.nav.delta_x, s.nav.delta_y, s.nav.delta_theta
            rows.append(f"{s.idx},{s.ts},{x:.6f},{y:.6f},{yaw:.6f},{v:.6f},{w:.6f},"
                        f"{dx:.6f},{dy:.6f},{dt:.6f},{s.phase}")
        
        with open(out_csv_path, "w") as f:
            f.write("idx,ts_ns,x_m,y_m,theta_deg,v_ms,w_rs,dx_m,dy_m,dt_deg,phase\n")
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
            stereo_l_path = os.path.join(frame_dir, "slam_l.png")
            stereo_r_path = os.path.join(frame_dir, "slam_r.png")
            cv2.imwrite(stereo_l_path, s.stereo.left_img)
            cv2.imwrite(stereo_r_path, s.stereo.right_img)

        def safe_list(arr): return arr.tolist() if isinstance(arr, np.ndarray) else arr

        data = {
            "idx": s.idx, 
            "ts": s.ts,
            "phase": int(s.phase),
            "cam": {
                "h": s.cam.h, 
                "w": s.cam.w,
                "k": safe_list(s.cam.k), 
                "d": safe_list(s.cam.d), 
                "c2w": safe_list(s.cam.c2w),
                "c2d": safe_list(s.cam.c2d),
                "d2w": safe_list(s.cam.d2w),
                "rgb_path": rgb_path
            },
            "stereo": {
                "valid": s.stereo.valid,
                "slam_l_path": stereo_l_path,
                "slam_r_path": stereo_r_path
            },
            "nav": {
                "agent_pos_3d": safe_list(s.nav.agent_pos_3d),
                "agent_pos_2d": safe_list(s.nav.agent_pos_2d),
                "robot_pos_2d": safe_list(s.nav.robot_pos_2d),
                "robot_v": float(s.nav.robot_v),
                "robot_w": float(s.nav.robot_w),
                "delta_x": float(s.nav.delta_x),
                "delta_y": float(s.nav.delta_y),
                "delta_theta": float(s.nav.delta_theta)
            },
            "hands": []
        }
        for hand in s.hands:
            data["hands"].append({
                "is_right": hand.is_right, 
                "confidence": hand.confidence,
                "wrist_pose": safe_list(hand.wrist_pose), 
                "palm_pose": safe_list(hand.palm_pose),
                "keypoints_3d": safe_list(hand.hand_keypoints_3d), 
                "keypoints_2d": safe_list(hand.hand_keypoints_2d),
                "grasp_state": hand.grasp_state,
                "joint_angles": hand.joint_angles.data if hand.joint_angles else {}
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
        utils_media.create_video_from_frames(frames_all, os.path.join(self.save_path, "aria_video_orig.mp4"), **video_args)

    def _draw_trajectory_trail(self, img, current_idx, future_len=60, step=3, ground_offset=1.7):
        if len(self.trajectory_points_3d) == 0: return img
        
        # 1. 环境准备 (保持不变)
        s = self.all_aria_structs[current_idx]
        K = s.cam.k
        T_w2c = np.linalg.inv(s.cam.c2w) 
        h_vis, w_vis = img.shape[:2]
        h_raw, w_raw = w_vis, h_vis 

        # 2. 提取并“抽稀”点位
        start_idx = current_idx
        end_idx = min(len(self.trajectory_points_3d), current_idx + future_len)
        raw_pts = self.trajectory_points_3d[start_idx:end_idx:step].copy()
        if len(raw_pts) < 2: return img
        
        # --- [关键修改：距离过滤逻辑] ---
        # 确保用于生成路面的点之间至少有 10cm 的距离，防止抖动和重叠
        filtered_pts = [raw_pts[0]]
        for i in range(1, len(raw_pts)):
            dist = np.linalg.norm(raw_pts[i] - filtered_pts[-1])
            if dist > 0.10: # 10厘米阈值
                filtered_pts.append(raw_pts[i])
        
        points_world = np.array(filtered_pts)
        
        # 如果过滤后点太少，说明位移不足以支撑路面渲染
        should_draw_road = len(points_world) >= 2
        # -----------------------------

        points_world[:, 2] -= ground_offset # 映射到地面

        # 3. 计算路径边界 (仅在位移足够时执行)
        left_edges, right_edges = [], []
        if should_draw_road:
            ROAD_WIDTH = 0.35 
            for i in range(len(points_world)):
                if i < len(points_world) - 1:
                    direction = points_world[i+1] - points_world[i]
                else:
                    direction = points_world[i] - points_world[i-1]
                
                norm = np.array([-direction[1], direction[0], 0])
                norm_len = np.linalg.norm(norm)
                if norm_len < 1e-6: continue
                norm = (norm / norm_len) * ROAD_WIDTH
                left_edges.append(points_world[i] + norm)
                right_edges.append(points_world[i] - norm)

        # 4. 投影函数 (定义同前)
        def project_pts(pts3d):
            if len(pts3d) == 0: return np.array([]), np.array([])
            pts3d = np.array(pts3d)
            R, t = T_w2c[:3, :3], T_w2c[:3, 3]
            p_cam = (R @ pts3d.T).T + t
            
            mask = p_cam[:, 2] > 0.1 
            # 使用已经修正过的 K 进行投影
            uv_h = (K @ p_cam.T).T
            z = uv_h[:, 2] + 1e-6
            u_final = uv_h[:, 0] / z
            v_final = uv_h[:, 1] / z
            
            return np.stack([u_final, v_final], axis=1).astype(np.int32), mask

        # 5. 渲染底层路面 (仅在位移足够时渲染)
        if should_draw_road:
            l_2d, l_mask = project_pts(left_edges)
            r_2d, r_mask = project_pts(right_edges)
            ROAD_BASE_COLOR = (255, 200, 0)
            
            num_seg = len(l_2d) - 1
            for i in range(num_seg):
                if not (l_mask[i] and l_mask[i+1] and r_mask[i] and r_mask[i+1]): continue
                
                progress = i / num_seg
                # 额外的平滑度处理：如果机器人走得很慢，透明度进一步降低
                speed_factor = min(1.0, np.linalg.norm(points_world[-1] - points_world[0]) / 1.0)
                alpha = max(0, 0.4 * (1.0 - progress) * speed_factor)
                
                if alpha > 0.01:
                    sub_overlay = img.copy()
                    poly_pts = np.array([l_2d[i], l_2d[i+1], r_2d[i+1], r_2d[i]], np.int32)
                    cv2.fillPoly(sub_overlay, [poly_pts], ROAD_BASE_COLOR, lineType=cv2.LINE_AA)
                    img = cv2.addWeighted(sub_overlay, alpha, img, 1 - alpha, 0)

        # 6. 渲染航向点 (始终渲染原始采样点，确保视觉连贯性)
        # 重新投影原始采样点
        c_2d, c_mask = project_pts(raw_pts - np.array([0,0,ground_offset]))
        num_pts = len(c_2d)
        for i in range(num_pts):
            if not c_mask[i]: continue
            pt = tuple(c_2d[i])
            if not (0 <= pt[0] < w_vis and 0 <= pt[1] < h_vis): continue
            
            progress = i / max(1, num_pts - 1)
            hue = int(160 * progress) 
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
            radius = int(5 - 2 * progress) 
            cv2.circle(img, pt, radius, color_bgr, -1, cv2.LINE_AA)

        return img
    
    def _create_visualization_videos(self, audio_path: str):
        """生成带精美仪表盘、居中标题和清晰骨架的视频"""
        if not self.all_aria_structs: return
        frames_all = []

        # 记录初始位姿用于计算 Delta
        start_nav = self.all_aria_structs[0].nav.robot_pos_2d # [x0, y0, t0]
        
        # 配色与字体设置
        PHASE_COLORS = { 0: (0, 255, 127), 1: (255, 127, 0) } 
        PHASE_NAMES = { 0: "NAVIGATION", 1: "MANIPULATION"}
        TEXT_COLOR = (240, 240, 240)
        VAL_COLOR = (0, 255, 255) 
        LABEL_COLOR = (200, 200, 200)
        DELTA_COLOR = (255, 150, 50)
        # 使用 DUPLEX 字体看起来更硬朗、更帅
        FONT = cv2.FONT_HERSHEY_DUPLEX 
        THICK = 1

        def draw_glass_rect(img, pt1, pt2, alpha=0.5):
            """绘制更薄、更透明的灰色矩形框"""
            overlay = img.copy()
            cv2.rectangle(overlay, pt1, pt2, (25, 25, 25), -1)
            return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        def draw_arc_gauge(img, center, val, max_val, label, color_pos=(0, 255, 127), color_neg=(0, 0, 255)):
            """ 绘制汽车风格圆弧仪表盘 """
            cx, cy = center; r = 28
            # 背景弧
            cv2.ellipse(img, (cx, cy), (r, r), 0, 135, 405, (60, 60, 60), 4, cv2.LINE_AA)
            # 根据正负选择颜色
            color = color_pos if val >= 0 else color_neg
            # 计算弧度角度 (仪表盘范围 270 度: 从 135度 到 405度)
            angle = 270 * (abs(val) / max_val)
            angle = min(angle, 270)
            cv2.ellipse(img, (cx, cy), (r, r), 0, 135, 135 + int(angle), color, 4, cv2.LINE_AA)
            # 中间写简称
            cv2.putText(img, label, (cx-8, cy+5), FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        for s in tqdm(self.all_aria_structs, desc="渲染 UI"):
            img = s.cam.rgb.copy()

            # --- 基础 3D 绘制 ---
            img = self._draw_trajectory_trail(img, s.idx, future_len=90, step=3, ground_offset=1.7)
            for hand in s.hands:
                # img = AriaHandOps.draw_axis(img, hand.palm_pose, s.cam.k, s.cam.d)
                img = AriaHandOps.draw_axis(img, hand.wrist_pose, s.cam.k, s.cam.d)
                img = AriaHandOps.draw_skeleton(img, hand)

            # --- UI 布局：顶部 Phase (完全居中) ---
            phase_name = f"PHASE: {PHASE_NAMES.get(s.phase, 'UNKNOWN')}"
            p_color = PHASE_COLORS.get(s.phase, (255, 255, 255))
            
            # 计算文字宽度以实现真正居中
            (t_w, t_h), _ = cv2.getTextSize(phase_name, FONT, 0.6, 1)
            center_x = (640 - t_w) // 2
            # 绘制顶部半透明背景
            # img = draw_glass_rect(img, (center_x - 20, 10), (center_x + t_w + 20, 45), alpha=0.6)
            cv2.putText(img, phase_name, (center_x, 30), FONT, 0.6, p_color, 1, cv2.LINE_AA)

            # --- UI 布局：左上角 Nav Panel ---
            img = draw_glass_rect(img, (10, 40), (240, 360), alpha=0.6)
            
            # 计算增量 Delta
            dx = s.nav.delta_x
            dy = s.nav.delta_y
            dt = s.nav.delta_theta

            # 1. V 仪表盘 (圆弧) + 文字
            draw_arc_gauge(img, (45, 105), s.nav.robot_v, 1.0, "V")
            cv2.putText(img, f"LINEAR SPEED", (90, 95), FONT, 0.4, LABEL_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, f"{s.nav.robot_v:>5.2f} m/s", (90, 115), FONT, 0.5, VAL_COLOR, 1, cv2.LINE_AA)

            # 2. W 仪表盘 (圆弧) + 文字
            draw_arc_gauge(img, (45, 175), s.nav.robot_w, 0.75, "W")
            cv2.putText(img, f"ANGULAR SPEED", (90, 165), FONT, 0.4, LABEL_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, f"{s.nav.robot_w:>5.2f} r/s", (90, 185), FONT, 0.5, VAL_COLOR, 1, cv2.LINE_AA)

            # 3. X/Y 坐标 (文字区分 Abs 和 Delta)
            # 略微放大雷达框到 50x50 像素
            bx1, by1, bx2, by2 = 20, 220, 70, 270
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            
            # 绘制雷达背景框和十字基准线（增加科技感）
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (80, 80, 80), 1, cv2.LINE_AA)
            cv2.line(img, (cx, by1), (cx, by2), (50, 50, 50), 1) # 垂直基准线
            cv2.line(img, (bx1, cy), (bx2, cy), (50, 50, 50), 1) # 水平基准线

            # [关键修改] 提高灵敏度：量程从 5.0m 缩小到 2.0m
            # 这意味着机器人移动 2 米，点就会到达方框边缘
            # 你可以根据实际行走范围调整这个值，例如 0.5 (极灵敏) 或 2.0 (适中)
            SENSITIVITY_RANGE = 2.0
            
            half_width = (bx2 - bx1) // 2
            # 计算映射像素 (dx 和 dy 使用之前计算好的 Delta)
            map_px = int(cx + (dx / SENSITIVITY_RANGE) * half_width)
            map_py = int(cy - (dy / SENSITIVITY_RANGE) * half_width) # 图像 Y 轴向下，所以用减

            # 边缘约束：防止点跑出方框
            map_px = max(bx1 + 2, min(map_px, bx2 - 2))
            map_py = max(by1 + 2, min(map_py, by2 - 2))
            
            # 绘制当前位置：核心点 + 外圈发光效果
            cv2.circle(img, (map_px, map_py), 3, VAL_COLOR, -1, cv2.LINE_AA)
            cv2.circle(img, (map_px, map_py), 5, VAL_COLOR, 1, cv2.LINE_AA) # 外圈发光
            
            # 右侧文字信息
            cv2.putText(img, "POSITION (X, Y)", (90, 230), FONT, 0.4, LABEL_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, f"Abs: {s.nav.robot_pos_2d[0]:>5.2f}, {s.nav.robot_pos_2d[1]:>5.2f}m", (90, 250), FONT, 0.4, VAL_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, f"Dlt: {dx:>5.2f}, {dy:>5.2f}m", (90, 268), FONT, 0.4, DELTA_COLOR, 1, cv2.LINE_AA)

            # 4. Theta 罗盘仪表盘 + 文字
            c_cx, c_cy = 45, 315; c_r = 20
            
            # [核心修改] 获取已经是度的 Delta Theta
            dt_deg = s.nav.delta_theta
            
            # 绘图逻辑需要弧度：
            dt_rad = math.radians(dt_deg) 
            
            # 绘制背景
            cv2.circle(img, (c_cx, c_cy), c_r, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.line(img, (c_cx, c_cy - c_r), (c_cx, c_cy - c_r + 5), (200, 200, 200), 1, cv2.LINE_AA)
            
            # 绘制彩色填充弧
            # color_theta 逻辑
            color_theta = (0, 255, 127) if dt_deg >= 0 else (200, 0, 255)
            # cv2.ellipse 的角度参数本来就是 Degree，直接用 dt_deg
            cv2.ellipse(img, (c_cx, c_cy), (c_r, c_r), 0, -90, -90 + int(dt_deg), color_theta, 2, cv2.LINE_AA)

            # 绘制指针 (这里需要弧度)
            p_rad = dt_rad - math.pi/2 
            px = int(c_cx + c_r * math.cos(p_rad))
            py = int(c_cy + c_r * math.sin(p_rad))
            cv2.line(img, (c_cx, c_cy), (px, py), color_theta, 2, cv2.LINE_AA)
            cv2.circle(img, (c_cx, c_cy), 2, (255, 255, 255), -1)
            
            # 文字展示 (由于 s.nav.robot_pos_2d[2] 已经是度了，不需要再 math.degrees)
            cv2.putText(img, "ORIENTATION (T)", (90, 305), FONT, 0.4, LABEL_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, f"Abs: {s.nav.robot_pos_2d[2]:>5.1f} deg", (90, 325), FONT, 0.4, VAL_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, f"Dlt: {s.nav.delta_theta:>+5.1f} deg", (90, 343), FONT, 0.4, DELTA_COLOR, 1, cv2.LINE_AA)

            # --- UI 布局：底部手部面板 (全状态+数据对齐版) ---
            l_hand = next((h for h in s.hands if not h.is_right), None)
            r_hand = next((h for h in s.hands if h.is_right), None)
            is_blink = (s.idx // 5) % 2 == 0 

            def draw_hand_panel(img, hand, is_right):
                x = 430 if is_right else 10
                y = 475 # 稍微调高一点，为多出的行留出空间
                w, h = 200, 155 # 高度增加到 155
                
                # --- 1. 数据对齐映射表 ---
                mapping = {
                    "Thu": (["Thumb_CMC_Flex", "Thumb_MCP_Flex", "Thumb_IP_Flex"], "Thumb_CMC_Abd"),
                    "Ind": (["Index_MCP_Flex", "Index_PIP_Flex", "Index_DIP_Flex"], "Index_MCP_Abd"),
                    "Mid": (["Middle_MCP_Flex", "Middle_PIP_Flex", "Middle_DIP_Flex"], "Middle_MCP_Abd"),
                    "Rin": (["Ring_MCP_Flex", "Ring_PIP_Flex", "Ring_DIP_Flex"], "Ring_MCP_Abd"),
                    "Pin": (["Pinky_MCP_Flex", "Pinky_PIP_Flex", "Pinky_DIP_Flex"], "Pinky_MCP_Abd")
                }

                # --- 2. 状态逻辑 ---
                if hand is not None:
                    is_grasp = (hand.grasp_state == 1)
                    conf = hand.confidence
                    state_str = "CLOSED" if is_grasp else "OPEN"
                    col = (0, 0, 255) if is_grasp else (0, 191, 255)
                    angle_data = hand.joint_angles.data if hand.joint_angles else {}
                else:
                    is_grasp = False; conf = 0.0; state_str = "NOT DETECTED"
                    col = (120, 120, 120); angle_data = {}

                # --- 3. 绘制底框与特效 ---
                img = draw_glass_rect(img, (x, y), (x + w, y + h))
                if is_grasp:
                    if is_blink:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2, cv2.LINE_AA)
                        # [红点移动] 放在左上角空位，避免遮挡
                        cv2.circle(img, (x + 15, y + 18), 4, (0, 0, 255), -1, cv2.LINE_AA)

                # --- 4. 第一行：标题居中 ---
                title = f"{'RIGHT' if is_right else 'LEFT'} HAND"
                (t_w, _), _ = cv2.getTextSize(title, FONT, 0.45, 1)
                t_x = x + (w - t_w) // 2
                cv2.putText(img, title, (t_x, y + 25), FONT, 0.45, col, 1, cv2.LINE_AA)

                # --- 5. 第二行：Confidence 和 Status 全称 ---
                conf_txt = f"CONFIDENCE: {conf:.2f}"
                stat_txt = f"STATUS: {state_str}"
                # 较细小的字体展示全称
                cv2.putText(img, conf_txt, (x + 10, y + 45), FONT, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(img, stat_txt, (x + 105, y + 45), FONT, 0.32, col, 1, cv2.LINE_AA)

                # --- 6. 绘制 5 指动态 Bar ---
                finger_names = ["Thu", "Ind", "Mid", "Rin", "Pin"]
                for i, ui_name in enumerate(finger_names):
                    y_row = y + 65 + i * 18 # 从 y+65 开始绘制 Bar
                    cv2.putText(img, ui_name, (x + 10, y_row + 8), FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # 获取角度数据
                    flex_keys, abd_key = mapping[ui_name]
                    flex_sum = sum([angle_data.get(k, 0) for k in flex_keys])
                    abd_val = angle_data.get(abd_key, 0)
                    
                    f_col = (0, 0, 255) if is_grasp else (200, 130, 60)
                    f_w = int(min(flex_sum / 180.0, 1.0) * 125) 
                    a_w = int(min(abd_val / 45.0, 1.0) * 125)
                    
                    # 背景槽
                    cv2.rectangle(img, (x + 55, y_row + 1), (x + 180, y_row + 4), (50, 50, 50), -1)
                    cv2.rectangle(img, (x + 55, y_row + 7), (x + 180, y_row + 10), (50, 50, 50), -1)
                    
                    # 填充 Bar
                    if f_w > 0:
                        cv2.rectangle(img, (x + 55, y_row + 1), (x + 55 + f_w, y_row + 4), f_col, -1, cv2.LINE_AA)
                    if a_w > 0:
                        cv2.rectangle(img, (x + 55, y_row + 7), (x + 55 + a_w, y_row + 10), (100, 200, 255), -1, cv2.LINE_AA)
                return img

            # --- 每一帧都绘制面板 (不管手在不在) ---
            img = draw_hand_panel(img, l_hand, False)
            img = draw_hand_panel(img, r_hand, True)

            frames_all.append(img)
            
        video_args = {"fps": self.fps, "audio_path": audio_path if os.path.exists(audio_path) else None}
        utils_media.create_video_from_frames(frames_all, os.path.join(self.save_path, "aria_video_vis.mp4"), **video_args)

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
        T_world_device = np.eye(4)
        aria_nav = AriaNav(np.zeros(3), np.zeros(3), np.zeros(3), 0.0, 0.0)

        if self.has_online_calibration:
            corrected_ts = mps_dp.get_rgb_corrected_timestamp_ns(capture_timestamp_ns, TimeQueryOptions.CLOSEST)
            T_world_device = mps_dp.get_closed_loop_pose(corrected_ts, TimeQueryOptions.CLOSEST).transform_world_device.to_matrix()
            # these two lines are equal to： T_world_device = mps_dp.get_rgb_corrected_closed_loop_pose(capture_timestamp_ns, TimeQueryOptions.CLOSEST).transform_world_device.to_matrix()
           
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
        
        # 4. 获取原始内参（landscape）
        fx_orig, fy_orig = rgb_linear_calib.get_focal_lengths()
        cx_orig, cy_orig = rgb_linear_calib.get_principal_point()
        h_orig, w_orig = rgb_img.shape[:2]

        # 5. 图像旋转 (Landscape -> Portrait)
        # RGB 顺时针旋转 90 度 (k=-1) 之后需要修改两个东西：K和c2w
        rgb_img = np.rot90(rgb_img, k=-1)
        rgb_img = np.ascontiguousarray(rgb_img)

        # # 6. 计算旋转 90 度(顺时针)后的内参
        # # 旋转后：新轴 x = 旧轴 (h - 1 - y), 新轴 y = 旧轴 x
        fx_rot, fy_rot = fy_orig, fx_orig
        cx_rot = h_orig - 1 - cy_orig
        cy_rot = cx_orig

        # @@@@@@@@@DEBUG@@@@@@@@@
        # fx_rot, fy_rot = fx_orig, fy_orig
        # cx_rot = cx_orig
        # cy_rot = cy_orig
        # @@@@@@@@@@@@@@@@@@

        # 7. 缩放图像
        rgb_img = cv2.resize(rgb_img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

        # 8. 计算缩放参数
        # 旋转后 w = h_orig_land, h = w_orig_land
        scale_w = TARGET_WIDTH / h_orig
        scale_h = TARGET_HEIGHT / w_orig

        # 9. 最后的h和w
        h, w = TARGET_HEIGHT, TARGET_WIDTH
        
        # 最后的fx,fy,cx,cy
        fx = fx_rot*scale_w
        fy = fy_rot*scale_h
        cx = cx_rot*scale_w
        cy = cy_rot*scale_h

        # 10. 最后的K (after rot and scale)
        k_matrix = np.array([
            [fx,           0,            cx         ], 
            [0,            fy,           cy         ], 
            [0,            0,            1          ]
        ])

        # 11. 计算T_world_camera
        # 公式：T_world_camera = T_world_device * T_device_camera
        T_device_camera = rgb_calib.get_transform_device_camera().to_matrix()
        
        # 定义绕 Z 轴顺时针旋转 90 度的旋转矩阵 R_z
        # theta = -90 度 (-pi/2 弧度)
        # R = [[cos(theta), -sin(theta), 0],
        #      [sin(theta),  cos(theta), 0],
        #      [0,           0,          1]]
        
        R_z_90 = np.array([
            [0, 1,  0,  0],
            [-1,  0,  0,  0],
            [0,  0,  1,  0],
            [0,  0,  0,  1]
        ])
        T_device_camera_rotated = T_device_camera @ R_z_90

        T_world_camera = T_world_device @ T_device_camera_rotated

        aria_cam = AriaCam(rgb=rgb_img, h=h, w=w, k=k_matrix, d=np.zeros(5), c2w=T_world_camera, c2d=T_device_camera_rotated, d2w=T_world_device)
        
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
            hands_result.left_hand, np.linalg.inv(T_device_camera_rotated), k_matrix, h, w, False
        )
        if left_hand: aria_hands.append(left_hand)

        # 处理右手
        right_hand =AriaHandOps.process_single_hand(
            hands_result.right_hand, np.linalg.inv(T_device_camera_rotated), k_matrix, h, w, True
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

        print("开始处理所有 Aria 帧...")
    
        # 1. 【修复】：先计算全局导航轨迹，填充 nav_data_map
        if self.has_online_calibration:
            self._process_full_navigation_trajectory()
        else:
            print("警告: 缺失在线标定数据，导航可视化将保持为零。")

        # 2. 现在再读取数据，__getitem__ 就能正确找到 nav_data_map 里的值了
        temp_structs = []
        for i in tqdm(range(len(self)), desc="读取并处理数据帧"):
            temp_structs.append(self[i])

        # [Step 2] 保存点云数据 (调用 AriaPointCloudOps)
        # 保存融合点云 (环境 + 轨迹)
        AriaPointCloudOps.save_merged_point_cloud(
            self.point_cloud, self.trajectory_points_3d,
            os.path.join(self.save_path, "aria_pc_scene_and_head_traj.ply")
        )
        # 仅保存环境点云
        AriaPointCloudOps.save_point_cloud_ply(
            self.point_cloud, os.path.join(self.save_path, "aria_pc_scene.ply")
        )
        # 可视化 (可选，默认注释掉以支持 Headless 模式)
        # AriaPointCloudOps.visualize_scene_with_trajectory(self.point_cloud, self.trajectory_points_3d)
        
        # [Step 3] 提取原始音频
        audio_path = os.path.join(self.save_path, "aria_audio_orig.wav")
        extract_audio_track(self.vrs_path, audio_path)
            
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

        