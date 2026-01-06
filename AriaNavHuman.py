# TODO: vis human traj：路不是渐变的，点也不是很多颜色的
# -*- coding: utf-8 -*-
# @FileName: AriaNavHuman.py

import numpy as np
import cv2
import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import argparse
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeQueryOptions, TimeDomain

from utils.utils_media import create_video_from_frames
from aria.AriaCam import AriaCamGenerator

NAV_SAMPLE_N = 10

@dataclass
class AriaNavHumanData:
    ts: int
    T_world: np.ndarray  # (4, 4) 原始世界坐标系位姿 (T_d2w)
    T_local: np.ndarray  # (4, 4) PCA对齐后的局部坐标系位姿 
    theta_start_rad: float = 0.0  # 记录该轨迹第一帧的绝对弧度（local系下）

    @property
    def t_world(self) -> np.ndarray: 
        return self.T_world[0:3, 3]
    @property
    def R_world(self) -> np.ndarray: 
        return self.T_world[0:3, 0:3]

    @property
    def t_local(self) -> np.ndarray: 
        return self.T_local[0:3, 3]
    @property
    def R_local(self) -> np.ndarray: 
        return self.T_local[0:3, 0:3]

    # project huamn traj to ground (x,y,theta) (以 Aria 设备坐标系 Z 轴为前方，投影到水平面计算 Yaw)
    @property
    def x_world(self) -> float:
        return self.t_world[0]
    @property
    def y_world(self) -> float:
        return self.t_world[1]
    @property
    def theta_world(self) -> float:
        forward_vector = self.R_world[:, 2]
        theta_rad = math.atan2(forward_vector[1], forward_vector[0])
        theta_deg = math.degrees(theta_rad)
        return theta_deg

    # --- Local 属性 (PCA 对齐系) ---
    @property
    def x_local(self) -> float:
        return self.t_local[0]
    x = x_local
    @property
    def y_local(self) -> float:
        return self.t_local[1]
    y = y_local
    @property
    def theta_local(self) -> float: # 相对于第一帧朝向的角度增量 (Degree)
        forward_vector = self.R_local[:, 2]
        theta_rad = math.atan2(forward_vector[1], forward_vector[0])
        diff_rad = theta_rad - self.theta_start_rad
        norm_rad = math.atan2(math.sin(diff_rad), math.cos(diff_rad)) # 角度归一化到 [-pi, pi]
        theta_deg = math.degrees(norm_rad)
        return theta_deg
    theta = theta_local

@dataclass
class AriaNavHuman:
    tss: List[int] = field(default_factory=list)
    traj: List[AriaNavHumanData] = field(default_factory=list)
    T_w2l: np.ndarray = field(default_factory=lambda: np.eye(4))
    walk_range: Optional[Tuple[int, int]] = None # 记录行走片段的起始/结束索引
    
    tss_aug: List[int] = field(default_factory=list)
    traj_aug: List[AriaNavHumanData] = field(default_factory=list)
    walk_range_aug: Optional[Tuple[int, int]] = None # 记录行走片段的起始/结束索引

    mps_path: str = None

    def __len__(self):
        return len(self.tss)
    
    @staticmethod
    def _safe_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else arr
    
    def save_aria_nav_human_json(self):
        for i in range(len(self.tss)):
            frame_dir = os.path.join(self.mps_path, "aria", "all_data", f"{i:05d}")
            os.makedirs(frame_dir, exist_ok=True)

            # 获取当前原始帧的数据
            current_data = self.traj[i]

            seg_start = i * NAV_SAMPLE_N
            seg_end = (i + 1) * NAV_SAMPLE_N
            segment_data = self.traj_aug[seg_start:seg_end]

            # 构造采样点列表
            traj_aug_samples = []
            for s_data in segment_data:
                traj_aug_samples.append({
                    "ts": s_data.ts,
                    "T_world": self._safe_list(s_data.T_world),
                    "T_local": self._safe_list(s_data.T_local)
                })

            json_data = {
                "idx": i,
                "ts": current_data.ts,
                "T_world": self._safe_list(current_data.T_world),
                "T_local": self._safe_list(current_data.T_local),
                "x": current_data.x,
                "y": current_data.y,
                "theta": current_data.theta,

                "T_w2l": self._safe_list(self.T_w2l),
                "walk_range": self.walk_range,
                "walk_range_aug": self.walk_range_aug,
                
                "traj_aug_samples": traj_aug_samples  # 这里存入了 NAV_SAMPLE_N 个细分点
            }

            with open(os.path.join(frame_dir, "aria_nav_human.json"), 'w') as f:
                json.dump(json_data, f, indent=4)

class AriaNavHumanGenerator:
    def __init__(self, mps_path: str, mps_provider: MpsDataProvider, tss: List[int]):
        self.mps_path = mps_path
        self.mps_provider = mps_provider
        self.tss = tss
    
    def get_aria_nav_human(self, run_aug=False) -> Optional[AriaNavHuman]:
        aria_nav_human = AriaNavHuman()
        aria_nav_human.mps_path = self.mps_path

        # 1. 统一只算一次 PCA，以 Orig 轨迹为基准
        T_w2l, T_world_list, walk_range = self._compute_pca_alignment(self.tss)
        aria_nav_human.T_w2l = T_w2l
        aria_nav_human.walk_range = walk_range
        
        # 2. 获取并转换 Orig 轨迹
        aria_nav_human.traj = self._get_aria_nav_human_traj(self.tss, T_w2l, T_world_list, "Processing Orig")
        aria_nav_human.tss = [d.ts for d in aria_nav_human.traj]
        
        # 3. 绘图
        os.makedirs(os.path.join(self.mps_path, "aria"), exist_ok=True)
        
        plot_path = os.path.join(self.mps_path, "aria", "aria_nav_human_analysis_orig.png")
        AriaNavHumanOps.save_trajectory_plot(aria_nav_human.traj, walk_range, plot_path, subtitle="Original")

        if run_aug:
            tss_aug = self._get_tss_aug()

            T_world_list_aug = []
            for ts in tss_aug:
                T_world_list_aug.append(self.mps_provider.get_closed_loop_pose(ts, TimeQueryOptions.CLOSEST).transform_world_device.to_matrix())

            aria_nav_human.traj_aug = self._get_aria_nav_human_traj(tss_aug, T_w2l, T_world_list_aug, "Processing Aug")
            aria_nav_human.tss_aug = [d.ts for d in aria_nav_human.traj_aug]
            aria_nav_human.walk_range_aug = (walk_range[0] * NAV_SAMPLE_N, walk_range[1] * NAV_SAMPLE_N)
            
            plot_path_aug = os.path.join(self.mps_path, "aria", "aria_nav_human_analysis_aug.png")
            AriaNavHumanOps.save_trajectory_plot(aria_nav_human.traj_aug, aria_nav_human.walk_range_aug, plot_path_aug, subtitle="Augmented (High Frequency)")

        AriaNavHumanOps.print_summary_and_eval(aria_nav_human)

        return aria_nav_human

    def _get_aria_nav_human_traj(self, tss: List[int], T_w2l, T_world_list, desc):
        
        aria_nav_human_traj = []

        # get theta_start_rad
        T0_local = T_w2l @ T_world_list[0]
        fwd0 = T0_local[:3, 2]
        theta_start_rad = math.atan2(fwd0[1], fwd0[0])

        for ts, T_d2w in tqdm(zip(tss, T_world_list), desc=desc):
            T_d2l = T_w2l @ T_d2w 
            aria_nav_human_traj.append(AriaNavHumanData(ts, T_d2w, T_d2l, theta_start_rad))
      
        return aria_nav_human_traj
    
    def _compute_pca_alignment(self, tss: List[int]):
        T_world_list = []
        pts_world = []
        for ts in tss:
            T_d2w = self.mps_provider.get_closed_loop_pose(ts, TimeQueryOptions.CLOSEST).transform_world_device.to_matrix()
            T_world_list.append(T_d2w)
            pts_world.append(T_d2w[:3, 3])
        
        # 简单清洗数据
        valid_indices = [i for i, p in enumerate(pts_world) if p is not None]
        pts_clean = np.array([pts_world[i] for i in valid_indices])
        tss_clean = np.array([tss[i] for i in valid_indices])

        # 运动分割
        dt = np.diff(tss_clean) / 1e9
        dr = np.linalg.norm(np.diff(pts_clean, axis=0), axis=1)
        vel = np.convolve(dr/(dt+1e-9), np.ones(5)/5, mode='same')
        walking_mask = vel > 0.2
        walk_idx = np.where(walking_mask)[0]
        
        walking_pts = pts_clean[walk_idx] if len(walk_idx) > 5 else pts_clean
        walk_range = (int(walk_idx[0]), int(walk_idx[-1])) if len(walk_idx) > 5 else (0, len(pts_clean)-1)

        # PCA 拟合
        t0 = pts_clean[0]
        xy_pts = walking_pts[:, :2] - t0[:2]
        _, _, vh = np.linalg.svd(xy_pts)
        x_dir_2d = vh[0]
        if np.dot(x_dir_2d, pts_clean[-1, :2] - t0[:2]) < 0: x_dir_2d = -x_dir_2d
        
        R_w2l = np.stack([
            np.array([x_dir_2d[0], x_dir_2d[1], 0.0]),
            np.array([-x_dir_2d[1], x_dir_2d[0], 0.0]),
            np.array([0.0, 0.0, 1.0])
        ], axis=0)

        T_w2l = np.eye(4)
        T_w2l[:3, :3] = R_w2l
        T_w2l[:3, 3] = -R_w2l @ t0
        return T_w2l, T_world_list, walk_range
    
    def _get_tss_aug(self) -> List[int]:
        tss_aug = []
        for i in range(len(self.tss) - 1):
            segment = np.linspace(self.tss[i], self.tss[i+1], NAV_SAMPLE_N, endpoint=False)
            tss_aug.extend(segment)
        tss_aug.append(self.tss[-1]) # add the last point
        tss_aug = np.array(tss_aug, dtype=int).tolist()
        return tss_aug

class AriaNavHumanOps:
    @staticmethod
    def save_trajectory_plot(traj_data: List[AriaNavHumanData], walk_range, save_path: str, subtitle="Original"):
        if not traj_data: return
        
        # 数据提取
        pts_w = np.array([d.t_world for d in traj_data])
        pts_l = np.array([d.t_local for d in traj_data])
        theta_w = np.array([d.theta_world for d in traj_data])
        theta_l = np.array([d.theta_local for d in traj_data])
        idx = np.arange(len(pts_w))

        # 设置高级绘图风格
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axs = plt.subplots(4, 2, figsize=(18, 16), sharex='col')
        
        # 颜色与标签定义 (保持专业色系)
        colors = ['#E74C3C', '#27AE60', '#2980B9', '#8E44AD'] # X, Y, Z, Theta
        labels = ['X (Forward Path)', 'Y (Lateral Drift)', 'Z (Altitude)', 'Theta (Orientation)']
        units = ['meters', 'meters', 'meters', 'degrees']
        
        data_left = [pts_w[:, 0], pts_w[:, 1], pts_w[:, 2], theta_w]
        data_right = [pts_l[:, 0], pts_l[:, 1], pts_l[:, 2], theta_l]

        for i in range(4):
            # --- 左列: Raw World (原始 MPS 轨迹) ---
            axs[i, 0].plot(idx, data_left[i], color=colors[i], alpha=0.6, linewidth=1.5)
            axs[i, 0].set_ylabel(f"{labels[i]}\n({units[i]})", fontweight='bold')
            if i == 0: axs[i, 0].set_title("Raw MPS World Coordinate", fontsize=14, fontweight='bold', pad=15)
            
            # --- 右列: Aligned Local (PCA 对齐后的轨迹) ---
            axs[i, 1].plot(idx, data_right[i], color=colors[i], linewidth=2.5)
            # 为 Y, Z, Theta 增加 0 基准线，方便观察波动
            if i > 0: axs[i, 1].axhline(y=0, color='black', linestyle=':', alpha=0.5)
            
            if i == 0: axs[i, 1].set_title("PCA Aligned Local Frame", fontsize=14, fontweight='bold', pad=15)

            # 统一标记行走区间
            if walk_range:
                for col in [0, 1]:
                    axs[i, col].axvspan(walk_range[0], walk_range[1], color='#BDC3C7', alpha=0.2, label='Walking Phase')
                    # 仅在第一行添加图例
                    if i == 0 and col == 1: axs[i, col].legend(loc='upper right', frameon=True)

            # 优化右侧 Local 列的显示范围 (Y, Z 聚焦细节)
            if i == 1 or i == 2: # Y, Z
                axs[i, 1].set_ylim(-0.4, 0.4)
            elif i == 3: # Theta
                cur_max = np.max(np.abs(data_right[i]))
                limit = max(30, cur_max * 1.5)
                axs[i, 1].set_ylim(-limit, limit)

            # 增加网格细节
            for col in [0, 1]:
                axs[i, col].grid(True, which='major', linestyle='--', alpha=0.7)

        # 标注底部 X 轴
        axs[3, 0].set_xlabel("Frame Index (Time Sequence)", fontweight='bold')
        axs[3, 1].set_xlabel("Frame Index (Time Sequence)", fontweight='bold')
        
        # 总标题
        plt.suptitle(f"Aria Trajectory Refinement: MPS World vs. PCA Local ({subtitle})\n"
                     f"Automatic Alignment Optimization for Robot Kinematics", fontsize=20, fontweight='bold', y=0.97)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"高级轨迹对比图已保存至: {save_path}")

    @staticmethod
    def print_summary_and_eval(nav_human: AriaNavHuman):
        if not nav_human.traj:
            print("[Error] No trajectory data to summarize.")
            return

        # 基础数据计算
        w_start, w_end = nav_human.walk_range
        p_start = nav_human.traj[w_start].t_local
        p_end = nav_human.traj[w_end].t_local
        diff = p_end - p_start
        abs_dist = np.linalg.norm(diff)
        
        # 定量评价指标
        walking_pts_local = np.array([d.t_local for d in nav_human.traj[w_start:w_end+1]])
        std_y = np.std(walking_pts_local[:, 1])
        var_all = np.var(walking_pts_local, axis=0)
        x_energy_ratio = var_all[0] / (np.sum(var_all) + 1e-9) * 100
        
        # 终端美化输出
        print("\n" + "╔" + "═" * 60 + "╗")
        print(f"║{'ARIA NAV HUMAN TRAJECTORY REPORT':^60}║")
        print("╠" + "═" * 60 + "╣")
        
        # 1. 轨迹元数据
        print(f"║ [DATA INFO] {'':<47}║")
        print(f"║  - Total Points (Orig) : {len(nav_human.traj):<34}║")
        print(f"║  - Total Points (Aug)  : {len(nav_human.traj_aug):<34}║")
        print(f"║  - Walking Range (Idx) : {f'{w_start} to {w_end}':<34}║")
        
        print(f"║ {'':<58} ║")
        
        # 2. 步行统计
        print(f"║ [WALKING STATISTICS] {'':<37}║")
        print(f"║  - Forward Dist (ΔX)   : {diff[0]:>8.4f} m {'':<23}║")
        print(f"║  - Lateral Drift(ΔY)   : {diff[1]:>8.4f} m {'':<23}║")
        print(f"║  - Altitude Δ (ΔZ)     : {diff[2]:>8.4f} m {'':<23}║")
        print(f"║  - Total Displacement  : {abs_dist:>8.4f} m {'':<23}║")
        
        print(f"║ {'':<58} ║")

        # 3. PCA 对齐质量评价
        print(f"║ [PCA ALIGNMENT QUALITY] {'':<34}║")
        print(f"║  - Y-Axis StdDev (Drift): {std_y:>8.4f} m {'':<23}║")
        print(f"║  - X-Axis Energy Ratio : {x_energy_ratio:>8.2f} % {'':<23}║")
        
        # 自动等级评定
        quality_str = "EXCELLENT" if std_y < 0.05 else "GOOD" if std_y < 0.15 else "AVERAGE"
        print(f"║  - Overall Alignment   : {quality_str:<34}║")
        
        print("╚" + "═" * 60 + "╝\n")

    @staticmethod
    def draw_aria_nav_human_traj(img: np.ndarray, idx: int, aria_nav_human: AriaNavHuman, 
                                 K: np.ndarray, T_c2w: np.ndarray, 
                                 future_len=90, step=3, ground_offset=1.7):
        """
        在图像上绘制彩虹色的人类运动轨迹轨迹。
        Args:
            img: 待绘制的图像
            idx: 当前帧在 aria_nav_human 中的索引
            aria_nav_human: 完整的人类轨迹对象
            K: 相机内参
            T_c2w: 相机坐标系到世界坐标系的变换矩阵
            future_len: 向前预测展示的长度
            step: 采样步长
            ground_offset: 地面高度偏移
        """
        if len(aria_nav_human) == 0 or idx >= len(aria_nav_human):
            return img
        
        h_vis, w_vis = img.shape[:2]
        T_w2c = np.linalg.inv(T_c2w)
        
        # 1. 提取未来一段时间内的 3D 点位
        start_idx = idx
        end_idx = min(len(aria_nav_human), idx + future_len)
        
        # 获取原始 3D 点
        raw_pts_3d = []
        for i in range(start_idx, end_idx, step):
            raw_pts_3d.append(aria_nav_human.traj[i].t_world.copy())
            
        if len(raw_pts_3d) < 2:
            return img

        points_world = np.array(raw_pts_3d)
        points_world[:, 2] -= ground_offset  # 映射到地面

        # 2. 投影函数
        def project_pts(pts3d):
            if len(pts3d) == 0: return np.array([]), np.array([])
            pts3d = np.array(pts3d)
            R_w2c, t_w2c = T_w2c[:3, :3], T_w2c[:3, 3]
            p_cam = (R_w2c @ pts3d.T).T + t_w2c
            
            mask = p_cam[:, 2] > 0.1  # 深度过滤
            uv_h = (K @ p_cam.T).T
            z = uv_h[:, 2] + 1e-6
            u_final = uv_h[:, 0] / z
            v_final = uv_h[:, 1] / z
            uv = np.stack([u_final, v_final], axis=1).astype(np.int32)
            return uv, mask

        # 3. 计算路面边界
        ROAD_WIDTH = 0.35
        left_edges, right_edges = [], []
        for i in range(len(points_world)):
            if i < len(points_world) - 1:
                direction = points_world[i+1] - points_world[i]
            else:
                direction = points_world[i] - points_world[i-1]
            
            # 计算水平面内的法向量
            norm = np.array([-direction[1], direction[0], 0])
            norm_len = np.linalg.norm(norm)
            if norm_len > 1e-6:
                norm = (norm / norm_len) * ROAD_WIDTH
                left_edges.append(points_world[i] + norm)
                right_edges.append(points_world[i] - norm)

        # 4. 渲染路面 (优化性能：使用单次 overlay)
        if len(left_edges) >= 2:
            l_2d, l_mask = project_pts(np.array(left_edges))
            r_2d, r_mask = project_pts(np.array(right_edges))

            # 创建一个用于绘制路面的透明层
            overlay = img.copy()
            ROAD_BASE_COLOR = (255, 200, 0) # 亮蓝色
            
            for i in range(len(l_2d) - 1):
                # 只有当这一段的四个顶点都在相机前方时才绘制
                if not (l_mask[i] and l_mask[i+1] and r_mask[i] and r_mask[i+1]): continue
                
                # 随着距离增加，路面越来越透明
                alpha = max(0, 0.4 * (1.0 - i/len(l_2d)))
                poly_pts = np.array([l_2d[i], l_2d[i+1], r_2d[i+1], r_2d[i]], np.int32)
                
                # 在普通原图和 overlay 之间混合 (为了解决段与段之间的重叠暗线)
                cv2.fillPoly(overlay, [poly_pts], ROAD_BASE_COLOR, cv2.LINE_AA)

            # 最终混合
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        # 5. 渲染彩虹航向点
        c_2d, c_mask = project_pts(points_world)
        for i in range(len(c_2d)):
            if not c_mask[i]: continue
            pt = tuple(c_2d[i])
            if not (0 <= pt[0] < w_vis and 0 <= pt[1] < h_vis): continue
            
            # 颜色计算：从黄色向绿色/紫色过渡 (HSV 空间)
            progress = i / max(1, len(c_2d) - 1)
            hue = int(160 * progress)  # 0-160
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
            # 远处的点变小
            radius = max(1, int(5 - 3 * progress))
            cv2.circle(img, pt, radius, color_bgr, -1, cv2.LINE_AA)

        return img

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

    # get aria_nav_human
    aria_nav_human_generator = AriaNavHumanGenerator(mps_path, mps_provider, tss)
    aria_nav_human = aria_nav_human_generator.get_aria_nav_human(run_aug=True)

    # save data
    aria_nav_human.save_aria_nav_human_json()

    # visualize
    def visualize_all():
        frames_all = []
        for idx, ts in enumerate(tqdm(tss, desc="Visualization")):
            img = aria_cam.cam[idx].rgb.copy()

            ########## orig data #########
            # draw nav huamn traj
            img = AriaNavHumanOps.draw_aria_nav_human_traj(img, idx, aria_nav_human, aria_cam.cam[idx].k, aria_cam.cam[idx].c2w)

            frames_all.append(img)
        
        save_path = os.path.join(mps_path, "aria", "aria_video_vis.mp4")
        create_video_from_frames(
            frames=frames_all, 
            save_path=save_path,
            fps=aria_cam.fps)
    
    visualize_all()

# conda activate aria
# cd src
# python -m aria.AriaNavHuman --mps_path "./data/mps_open_cabinet_5_vrs/"