# -*- coding: utf-8 -*-
# @FileName: AriaHand.py

import os
import json
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional,Dict, List
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeDomain

from utils.utils_media import create_video_from_frames
from utils.utils_vis import draw_glass_rect
from aria.AriaCam import AriaCam, AriaCamGenerator

# 抓取判定阈值 (单位: 米)
GRASP_THRESHOLD = 0.05 

@dataclass
class AriaJointAngle:
    """
    存储手部 20 个关节角度数据 (基于 emg2pose 定义)
    单位：度 (Degrees)
    """
    # 存储原始字典数据以便快速访问
    data: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def from_keypoints_3d(cls, kpts: np.ndarray):
        """
        核心计算逻辑：严格适配 Aria 21点序。
        5-Wrist, 20-Palm, 0-4 Tips, 6-19 Joints
        """
        if kpts is None or len(kpts) < 21:
            return cls(data={})
        
        angles = {}

        def get_angle(v1, v2):
            v1_n = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_n = v2 / (np.linalg.norm(v2) + 1e-6)
            return np.degrees(np.arccos(np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)))

        def get_abduction(bone_vec, ref_vec, plane_normal):
            def project(v): return v - np.dot(v, plane_normal) * plane_normal
            return get_angle(project(bone_vec), project(ref_vec))

        # 1. 建立掌心平面 (使用 Wrist-5, IndexMCP-8, MiddleMCP-11)
        v_w_m = kpts[11] - kpts[5] 
        v_w_i = kpts[8] - kpts[5]
        palm_normal = np.cross(v_w_i, v_w_m)
        palm_normal /= (np.linalg.norm(palm_normal) + 1e-6)
        
        # 2. 中指近节作为外展参考向量 (Middle MCP-11 to PIP-12)
        v_mid_prox_ref = kpts[12] - kpts[11]

        # 3. 四指计算 [MCP, PIP, DIP, Tip]
        fingers_map = {
            'Index':  [8, 9, 10, 1],
            'Middle': [11, 12, 13, 2],
            'Ring':   [14, 15, 16, 3],
            'Pinky':  [17, 18, 19, 4]
        }

        for name, idxs in fingers_map.items():
            mcp, pip, dip, tip = idxs
            v_metacarpal = kpts[mcp] - kpts[5]  # Wrist to MCP
            v_prox       = kpts[pip] - kpts[mcp] # MCP to PIP
            v_inter      = kpts[dip] - kpts[pip] # PIP to DIP
            v_dist       = kpts[tip] - kpts[dip] # DIP to Tip
            
            # Flexion (弯曲角度)
            angles[f'{name}_MCP_Flex'] = get_angle(v_metacarpal, v_prox)
            angles[f'{name}_PIP_Flex'] = get_angle(v_prox, v_inter)
            angles[f'{name}_DIP_Flex'] = get_angle(v_inter, v_dist)
            
            # Abduction (外展角度)
            if name == 'Middle':
                angles[f'{name}_MCP_Abd'] = 0.0
            else:
                angles[f'{name}_MCP_Abd'] = get_abduction(v_prox, v_mid_prox_ref, palm_normal)

        # 4. 拇指计算 [5-Wrist, 6-MCP, 7-PIP, 0-Tip]
        v_thu_metacarpal = kpts[6] - kpts[5]
        v_thu_prox       = kpts[7] - kpts[6]
        v_thu_dist       = kpts[0] - kpts[7]
        
        angles['Thumb_CMC_Flex'] = get_angle(kpts[11]-kpts[5], v_thu_metacarpal) # 简化定义
        angles['Thumb_CMC_Abd']  = get_abduction(v_thu_metacarpal, v_mid_prox_ref, palm_normal)
        angles['Thumb_MCP_Flex'] = get_angle(v_thu_metacarpal, v_thu_prox)
        angles['Thumb_IP_Flex']  = get_angle(v_thu_prox, v_thu_dist)

        return cls(data=angles)
    
@dataclass
class AriaHandData:
    is_right: bool
    confidence: float
    wrist_pose: Optional[np.ndarray] = None       # (4, 4)
    palm_pose: Optional[np.ndarray] = None        # (4, 4)
    hand_keypoints_3d: Optional[np.ndarray] = None # (21, 3)
    hand_keypoints_2d: Optional[np.ndarray] = None # (21, 2)
    grasp_state: int = 0                        # 0: Open, 1: Closed
    joint_angles: Optional[AriaJointAngle] = None
    wrist_vel_world: float = 0.0 # 手腕在世界坐标系下的速度 (m/s)

@dataclass
class AriaHandsData:
    ts: int = 0               
    hand_r: Optional[AriaHandData] = None
    hand_l: Optional[AriaHandData] = None

@dataclass
class AriaHands:
    tss: List[int] = field(default_factory=list)
    hands: List[AriaHandsData] = field(default_factory=list)
    mps_path: str = None
    
    def __len__(self):
        return len(self.tss)

    @staticmethod
    def _safe_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else arr
    
    def save_aria_hands_json(self):
        """将手部数据分帧存入 json"""
        for i in range(len(self.tss)):
            frame_dir = os.path.join(self.mps_path, "aria", "all_data", f"{i:05d}")
            os.makedirs(frame_dir, exist_ok=True)

            data = self.hands[i]
            
            def pack_hand(h: Optional[AriaHandData]):
                if h is None: return None
                return {
                    "confidence": h.confidence,
                    "grasp_state": h.grasp_state,
                    "wrist_vel_world": h.wrist_vel_world,
                    "wrist_pose": self._safe_list(h.wrist_pose),
                    "palm_pose": self._safe_list(h.palm_pose),
                    "kpts_3d": self._safe_list(h.hand_keypoints_3d),
                    "kpts_2d": self._safe_list(h.hand_keypoints_2d),
                    "joint_angles": h.joint_angles.data if h.joint_angles else {}
                }

            json_data = {
                "idx": i,
                "ts": data.ts,
                "hand_r": pack_hand(data.hand_r),
                "hand_l": pack_hand(data.hand_l)
            }

            with open(os.path.join(frame_dir, "aria_hands.json"), 'w') as f:
                json.dump(json_data, f, indent=4)

class AriaHandsGenerator():
    def __init__(self, mps_path: str, aria_hands_mps, aria_cam: AriaCam):
        self.mps_path = mps_path
        self.aria_hands_mps = aria_hands_mps
        self.aria_cam = aria_cam
        # 用于计算速度的缓存
        self.prev_wrist_pos_w_r = None
        self.prev_wrist_pos_w_l = None

    def get_aria_hands(self) -> Optional[AriaHands]:
        aria_hands = AriaHands(mps_path=self.mps_path)
        dt = 1.0 / self.aria_cam.fps

        for hands_mps, cam_data in tqdm(zip(self.aria_hands_mps, self.aria_cam.cam), total=len(self.aria_cam), desc="Processing Aria Hands"):
            d2c = np.linalg.inv(cam_data.c2d)
            aria_hands_data = self._get_aria_hands_data(hands_mps, d2c, cam_data.k, cam_data.h, cam_data.w, cam_data.ts)

            self._compute_and_assign_vel(aria_hands_data, cam_data.c2w, dt)

            aria_hands.hands.append(aria_hands_data)
            aria_hands.tss.append(cam_data.ts)
        
        # 渲染高级分析图与报告
        os.makedirs(os.path.join(self.mps_path, "aria"), exist_ok=True)
        plot_path = os.path.join(self.mps_path, "aria", "aria_hands_analysis.png")
        AriaHandsOps.save_hands_analysis_plot(aria_hands, plot_path)
        AriaHandsOps.print_summary_and_eval(aria_hands)

        return aria_hands

    def _compute_and_assign_vel(self, hands_data, c2w, dt):
        """计算手部在世界坐标系下的瞬时速度"""
        for is_right in [True, False]:
            h_data = hands_data.hand_r if is_right else hands_data.hand_l
            prev_pos_w = self.prev_wrist_pos_w_r if is_right else self.prev_wrist_pos_w_l
            
            if h_data and h_data.wrist_pose is not None:
                # 获取世界坐标: T_c2w @ P_wrist_cam
                p_cam = h_data.wrist_pose[:3, 3]
                p_world = (c2w[:3, :3] @ p_cam) + c2w[:3, 3]
                
                if prev_pos_w is not None:
                    vel = np.linalg.norm(p_world - prev_pos_w) / dt
                    h_data.wrist_vel_world = float(vel)
                else:
                    h_data.wrist_vel_world = 0.0
                
                # 更新缓存
                if is_right: self.prev_wrist_pos_w_r = p_world
                else: self.prev_wrist_pos_w_l = p_world
            else:
                # 手消失了，重置缓存
                if is_right: self.prev_wrist_pos_w_r = None
                else: self.prev_wrist_pos_w_l = None

    def _get_aria_hands_data(self, hands_mps, d2c, k, h, w, ts) -> AriaHandsData:
        hand_r = self._get_aria_hand_data(hands_mps.right_hand, d2c, k, h, w, is_right=True)
        hand_l = self._get_aria_hand_data(hands_mps.left_hand, d2c, k, h, w, is_right=False)
        aria_hands_data = AriaHandsData(ts, hand_r, hand_l)
        return aria_hands_data

    def _get_aria_hand_data(self, hand_mps, d2c, k, h, w, is_right) -> Optional[AriaHandData]:
        if hand_mps is None or not hand_mps.confidence: 
            return None
        
        confidence = hand_mps.confidence
        if confidence <= 0.1: 
            return None
        
        # 1. 提取法向量和关键位置
        palm_normal = hand_mps.wrist_and_palm_normal_device.palm_normal_device
        palm_position = hand_mps.landmark_positions_device[int(mps.hand_tracking.HandLandmark.PALM_CENTER)]
        wrist_normal = hand_mps.wrist_and_palm_normal_device.wrist_normal_device
        wrist_position = hand_mps.landmark_positions_device[int(mps.hand_tracking.HandLandmark.WRIST)]

        # 2. 计算抓取状态 (Thumb Tip <-> Index Tip 距离)
        thumb_tip = np.array(hand_mps.landmark_positions_device[int(mps.hand_tracking.HandLandmark.THUMB_FINGERTIP)])
        index_tip = np.array(hand_mps.landmark_positions_device[int(mps.hand_tracking.HandLandmark.INDEX_FINGERTIP)])
        distance = np.linalg.norm(thumb_tip - index_tip)
        grasp_state = 1 if distance < GRASP_THRESHOLD else 0

        # 3. 计算位姿矩阵
        palm_pose, wrist_pose = AriaHandsGenerator._compute_wrist_and_palm_pose(palm_normal, palm_position, wrist_normal, wrist_position, confidence, d2c)
        
        # 4. 坐标变换与投影
        keypoints_3d = AriaHandsGenerator._transform_keypoints_to_camera(hand_mps, d2c)
        keypoints_2d, _ = AriaHandsGenerator._project_points_rotated(keypoints_3d, k, h, w)

        # 5. joint angle
        joint_angles = AriaJointAngle.from_keypoints_3d(keypoints_3d)

        return AriaHandData(
            is_right=is_right,
            confidence=confidence,
            wrist_pose=wrist_pose,
            palm_pose=palm_pose,
            hand_keypoints_3d=keypoints_3d, 
            hand_keypoints_2d=keypoints_2d,
            grasp_state=grasp_state,
            joint_angles=joint_angles
        )

    @staticmethod
    def _compute_wrist_and_palm_pose(palm_n, palm_p, wrist_n, wrist_p, conf, T_cam_dev):
        """计算手腕和手掌的 6D Pose 矩阵"""
        if np.linalg.norm(wrist_n) == 0 or np.linalg.norm(palm_n) == 0: 
            return None, None
        
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
        
        # 变换到相机坐标系
        return  T_cam_dev @ palm_mat, T_cam_dev @ wrist_mat

    @staticmethod
    def _transform_keypoints_to_camera(hand, T_cam_dev):
        """将关键点从 Device 坐标系变换到 Camera 坐标系"""
        R = T_cam_dev[:3, :3]
        t = T_cam_dev[:3, 3]
        kpts_dev = np.array(hand.landmark_positions_device)
        kpts_cam = np.einsum("ij, nj -> ni", R, kpts_dev) + t
        return kpts_cam

    @staticmethod
    def _project_points_rotated(points_cam, k, h_target, w_target):
        """
        标准投影函数。由于内参 K 已经包含了旋转和缩放，这里直接投影即可。
        """
        z = points_cam[:, 2]
        valid_mask = z > 1e-3
        if not np.any(valid_mask):
            return np.zeros((len(points_cam), 2)), np.zeros(len(points_cam), dtype=bool)

        # 标准内参投影公式: uv = K * [X, Y, Z]^T / Z
        homo_pix = (k @ points_cam.T).T
        u = homo_pix[:, 0] / (homo_pix[:, 2] + 1e-6)
        v = homo_pix[:, 1] / (homo_pix[:, 2] + 1e-6)
        
        projected = np.stack((u, v), axis=-1)
        
        # 边界检查 (此时 h_target, w_target 已经是 640 了)
        in_bounds = (0 <= u) & (u < w_target) & (0 <= v) & (v < h_target)
        return projected, valid_mask & in_bounds
    
class AriaHandsOps:
    @staticmethod
    def save_hands_analysis_plot(aria_hands, save_path):
        """绘制双手的速度、置信度、抓取状态随时间变化的图表"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        n = len(aria_hands)
        time = np.arange(n)
        
        # 准备数据
        def get_series(hand_attr):
            r_vals, l_vals = [], []
            for h in aria_hands.hands:
                r_vals.append(getattr(h.hand_r, hand_attr) if h.hand_r else 0)
                l_vals.append(getattr(h.hand_l, hand_attr) if h.hand_l else 0)
            return np.array(r_vals), np.array(l_vals)

        conf_r, conf_l = get_series('confidence')
        vel_r, vel_l = get_series('wrist_vel_world')
        grasp_r, grasp_l = get_series('grasp_state')

        # 子图 1: 置信度
        axs[0].plot(time, conf_r, label='Right Confidence', color='#E74C3C', alpha=0.8)
        axs[0].plot(time, conf_l, label='Left Confidence', color='#3498DB', alpha=0.8)
        axs[0].set_title("Hand Tracking Confidence", fontsize=14, fontweight='bold')
        axs[0].set_ylim(0, 1.1); axs[0].legend()

        # 子图 2: 世界速度 (最重要的特征)
        axs[1].plot(time, vel_r, label='Right Wrist Vel (World)', color='#C0392B', linewidth=2)
        axs[1].plot(time, vel_l, label='Left Wrist Vel (World)', color='#2980B9', linewidth=2)
        axs[1].set_title("Hand Movement Velocity in World Frame (m/s)", fontsize=14, fontweight='bold')
        axs[1].set_ylabel("m/s"); axs[1].legend()

        # 子图 3: 抓取状态 (0/1)
        axs[2].fill_between(time, grasp_r, color='#E74C3C', alpha=0.3, label='Right Grasp')
        axs[2].fill_between(time, grasp_l, color='#3498DB', alpha=0.3, label='Left Grasp')
        axs[2].set_title("Grasp State (Closed=1, Open=0)", fontsize=14, fontweight='bold')
        axs[2].set_yticks([0, 1]); axs[2].set_xlabel("Frames"); axs[2].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Hand analysis plot saved to: {save_path}")

    @staticmethod
    def print_summary_and_eval(aria_hands):
        """专业的手部交互统计报告 (已修正双向速度统计)"""
        total_frames = len(aria_hands)
        r_frames = sum(1 for h in aria_hands.hands if h.hand_r)
        l_frames = sum(1 for h in aria_hands.hands if h.hand_l)
        
        # 1. 提取所有有效帧的速度进行统计
        all_vels = []
        for h in aria_hands.hands:
            if h.hand_r: all_vels.append(h.hand_r.wrist_vel_world)
            if h.hand_l: all_vels.append(h.hand_l.wrist_vel_world)
        
        max_v = max(all_vels) if all_vels else 0.0
        avg_v = sum(all_vels) / len(all_vels) if all_vels else 0.0

        # 2. 计算抓取次数 (通过信号跳变)
        def count_grasps(hand_list_attr):
            # 提取指定手的抓取序列
            states = []
            for h_data in aria_hands.hands:
                hand = getattr(h_data, hand_list_attr)
                states.append(hand.grasp_state if hand else 0)
            return sum(1 for i in range(1, len(states)) if states[i] == 1 and states[i-1] == 0)

        r_grasps = count_grasps('hand_r')
        l_grasps = count_grasps('hand_l')
        
        # 3. 打印报告
        print("\n" + "╔" + "═" * 60 + "╗")
        print(f"║{'ARIA HAND INTERACTION REPORT':^60}║")
        print("╠" + "═" * 60 + "╣")
        print(f"║  - Total Frames       : {total_frames:<38} ║")
        print(f"║  - Right Hand Presence: {r_frames:<10} ({r_frames/total_frames*100:>5.1f}%) {'':<15} ║")
        print(f"║  - Left Hand Presence : {l_frames:<10} ({l_frames/total_frames*100:>5.1f}%) {'':<15} ║")
        print(f"║ {'':<58} ║")
        print(f"║ [ACTIVITY & KINEMATICS] {'':<35} ║")
        print(f"║  - Grasp Events (R/L) : {f'{r_grasps} / {l_grasps}':<38} ║")
        print(f"║  - Peak Wrist Speed   : {max_v:>8.3f} m/s {'':<28} ║")
        print(f"║  - Mean Wrist Speed   : {avg_v:>8.3f} m/s {'':<28} ║")
        
        # 数据质量评定
        if total_frames > 0:
            quality_score = (r_frames + l_frames) / total_frames
            quality = "EXCELLENT" if quality_score > 0.8 else "GOOD" if quality_score > 0.5 else "PARTIAL"
        else:
            quality = "EMPTY"
            
        print(f"║  - Overall Data Qual  : {quality:<38} ║")
        print("╚" + "═" * 60 + "╝\n")

    @staticmethod
    def _draw_skeleton(img, hand):
        """ 优雅的手部骨架绘制：严格适配 Aria 官方点序 (5-Wrist, 20-Palm) """
        if hand is None or hand.hand_keypoints_2d is None: 
            return img
        
        pts = hand.hand_keypoints_2d.astype(np.int32)
        is_grasp = (hand.grasp_state == 1)

        if len(pts) < 21: return img
        
        # --- 1. 定义手指颜色映射 (BGR) ---
        # 按照 [大拇指, 食指, 中指, 无名指, 小指] 顺序
        if not is_grasp:
            # --- 高对比清新色系 (BGR) ---
            # 增加了饱和度，降低了白色占比，确保五指一眼就能分清
            finger_colors = [
                (160, 160, 255), # 大拇指 - 珊瑚粉 (带红感)
                (160, 255, 160), # 食指   - 嫩草绿 (带绿感)
                (255, 210, 160), # 中指   - 晴空蓝 (带蓝感)
                (150, 250, 250), # 无名指 - 明亮黄 (带黄感)
                (250, 160, 250)  # 小指   - 薰衣草紫 (带紫感)
            ]
        else:
            # 深色系 (保持高强度对比)
            finger_colors = [
                (0, 0, 255),     # 大拇指 - 纯红
                (0, 200, 0),     # 食指   - 深绿
                (255, 100, 0),   # 中指   - 深蓝
                (0, 215, 255),   # 无名指 - 亮金
                (200, 0, 200)    # 小指   - 浓紫
            ]

        # --- 2. 绘制骨骼连线 (极细浅灰色) ---
        # 使用官方定义的连接逻辑
        for conn in mps.hand_tracking.kHandJointConnections:
            idx1, idx2 = int(conn[0]), int(conn[1])
            p1, p2 = tuple(pts[idx1]), tuple(pts[idx2])
            
            # 过滤掉无效点并绘制连线
            if p1 != (0,0) and p2 != (0,0):
                # 连线统一使用极细、浅灰色
                cv2.line(img, p1, p2, (210, 210, 210), 1, cv2.LINE_AA)

        # --- 3. 绘制关键点 (小巧精美) ---
        # 定义点所属的手指 ID 用于分配颜色
        # 根据图示：0,6,7 属大拇指；1,8,9,10 属食指... 等等
        finger_map = {
            0:0, 6:0, 7:0,                # Thumb
            1:1, 8:1, 9:1, 10:1,          # Index
            2:2, 11:2, 12:2, 13:2,        # Middle
            3:3, 14:3, 15:3, 16:3,        # Ring
            4:4, 17:4, 18:4, 19:4         # Pinky
        }

        for i, pt in enumerate(pts):
            p = tuple(pt)
            if p == (0, 0): continue
            
            # 颜色逻辑
            if i == 5:
                color = (255, 255, 255) # 5号点：手腕白色
            elif i == 20:
                color = (0, 0, 0)       # 20号点：掌心黑色
            else:
                f_idx = finger_map.get(i, 0)
                color = finger_colors[f_idx]
            
            # 绘制关节点 (半径 2)
            cv2.circle(img, p, 2, color, -1, cv2.LINE_AA)
            
            # 抓取状态下的额外视觉增强：画一个外圈发光
            if is_grasp and i != 5 and i != 20:
                cv2.circle(img, p, 3, color, 1, cv2.LINE_AA)
            
            # 为黑色或白色的关键点画一个细微轮廓防止在类似背景中消失
            if i == 5 or i == 20:
                cv2.circle(img, p, 2, (150, 150, 150), 1, cv2.LINE_AA)

        return img

    @staticmethod
    def _draw_axis(img, pose, k, d):
        """[酷炫版] 绘制 3D 坐标轴，增加基盘和激光指向线"""
        if pose is None: return img
        
        # 1. 提取旋转向量和平移向量
        r_vec, _ = cv2.Rodrigues(pose[:3, :3])
        t_vec = pose[:3, 3]
        
        # 2. 定义 3D 空间中的点
        axis_len = 0.04   # 轴长 4cm
        pointer_len = 0.12 # 指向线长度 12cm
        base_r = 0.025     # 基盘半径 2.5cm
        
        # 定义：X, Y, Z轴端点, 原点, 指向线端点
        pts_3d = np.float32([
            [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len], # 0,1,2: XYZ轴端点
            [0, 0, 0],                                           # 3: 原点
            [0, 0, pointer_len]                                  # 4: 指向激光
        ])
        
        # 定义基盘圆周上的点 (XY 平面上的一个圈)
        num_circle_pts = 16
        circle_pts_3d = []
        for i in range(num_circle_pts):
            angle = 2 * np.pi * i / num_circle_pts
            circle_pts_3d.append([base_r * np.cos(angle), base_r * np.sin(angle), 0])
        pts_3d = np.append(pts_3d, np.float32(circle_pts_3d), axis=0)

        # 3. 投影到 2D 像素
        img_pts, _ = cv2.projectPoints(pts_3d, r_vec, t_vec, k, d)
        img_pts = [tuple(pt.ravel().astype(int)) for pt in img_pts]
        
        origin = img_pts[3]
        h, w = img.shape[:2]
        if not (0 <= origin[0] < w and 0 <= origin[1] < h):
            return img

        # --- 绘制开始 ---
        overlay = img.copy()

        # A. 绘制基盘 (XY平面半透明圆环)
        for i in range(num_circle_pts):
            p1 = img_pts[5 + i]
            p2 = img_pts[5 + (i + 1) % num_circle_pts]
            cv2.line(overlay, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)
        
        # B. 绘制指向激光 (Z方向长延伸)
        laser_end = img_pts[4]
        # 使用淡蓝色发光效果
        cv2.line(overlay, origin, laser_end, (255, 255, 0), 1, cv2.LINE_AA) 
        cv2.circle(overlay, laser_end, 2, (255, 255, 0), -1, cv2.LINE_AA)

        # C. 绘制标准 XYZ 轴 (带箭头)
        # X - Red, Y - Green, Z - Blue
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in range(3):
            # 使用带箭头的线，并开启抗锯齿
            cv2.arrowedLine(overlay, origin, img_pts[i], colors[i], 2, tipLength=0.3, line_type=cv2.LINE_AA)

        # D. 在原点画一个白色小核心
        cv2.circle(overlay, origin, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # 混合透明层 (令基盘和激光有虚幻感)
        return cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    @staticmethod
    def _draw_aria_hand_panel(img, idx, hand, is_right):
        """ 绘制手部 UI 信息面板 """
        is_blink = (idx // 5) % 2 == 0 

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
            angle_data = hand.joint_angles.data if (hand.joint_angles and hand.joint_angles.data) else {}
        else:
            is_grasp = False; conf = 0.0; state_str = "N/A"
            col = (120, 120, 120); angle_data = {}

        # --- 3. 绘制底框与特效 ---
        img = draw_glass_rect(img, (x, y), (x + w, y + h))
        if is_grasp and is_blink:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2, cv2.LINE_AA)
                # [红点移动] 放在左上角空位，避免遮挡
                cv2.circle(img, (x + 15, y + 18), 4, (0, 0, 255), -1, cv2.LINE_AA)

        # --- 4. 第一行：标题居中 ---
        title = f"{'RIGHT' if is_right else 'LEFT'} HAND"
        (t_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        t_x = x + (w - t_w) // 2
        cv2.putText(img, title, (t_x, y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.45, col, 1, cv2.LINE_AA)

        # --- 5. 第二行：Confidence 和 Status 全称 ---
        conf_txt = f"CONFIDENCE: {conf:.2f}"
        stat_txt = f"STATUS: {state_str}"
        # 较细小的字体展示全称
        cv2.putText(img, conf_txt, (x + 10, y + 45), cv2.FONT_HERSHEY_DUPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, stat_txt, (x + 105, y + 45), cv2.FONT_HERSHEY_DUPLEX, 0.32, col, 1, cv2.LINE_AA)

        # --- 6. 绘制 5 指动态 Bar ---
        finger_names = ["Thu", "Ind", "Mid", "Rin", "Pin"]
        for i, ui_name in enumerate(finger_names):
            y_row = y + 65 + i * 18 # 从 y+65 开始绘制 Bar
            cv2.putText(img, ui_name, (x + 10, y_row + 8), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            
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


    @staticmethod
    def draw_aria_hands_skeleton_and_wrist_pose(img, aria_hands_data: AriaHandsData, k, d):
        # draw skeleton
        if aria_hands_data.hand_r is not None:
            img = AriaHandsOps._draw_skeleton(img, aria_hands_data.hand_r)
        if aria_hands_data.hand_l is not None:
            img = AriaHandsOps._draw_skeleton(img, aria_hands_data.hand_l)

        # draw wrise pose
        if aria_hands_data.hand_r is not None:
            img = AriaHandsOps._draw_axis(img, aria_hands_data.hand_r.wrist_pose, k, d)
        if aria_hands_data.hand_l is not None:
            img = AriaHandsOps._draw_axis(img, aria_hands_data.hand_l.wrist_pose, k, d)
        return img

    @staticmethod
    def draw_aria_hands_panel(img, idx: int, aria_hands_data: AriaHandsData):
        img = AriaHandsOps._draw_aria_hand_panel(img, idx, aria_hands_data.hand_r, True)
        img = AriaHandsOps._draw_aria_hand_panel(img, idx, aria_hands_data.hand_l, False)
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

    # get aria_hands
    aria_hands_generator = AriaHandsGenerator(mps_path, aria_hands_mps, aria_cam)
    aria_hands = aria_hands_generator.get_aria_hands()

    # save data
    aria_hands.save_aria_hands_json()

    # visualize
    def visualize_all():
        frames_all = []
        for idx, ts in enumerate(tqdm(tss, desc="Visualization")):
            img = aria_cam.cam[idx].rgb.copy()

            ########## orig data #########
            # draw hands skeleton and wrist pose
            img = AriaHandsOps.draw_aria_hands_skeleton_and_wrist_pose(img, aria_hands.hands[idx], aria_cam.cam[idx].k, aria_cam.cam[idx].d)

            ########### Panels ###########
            # hands panel
            img = AriaHandsOps.draw_aria_hands_panel(img, idx, aria_hands.hands[idx])
            
            frames_all.append(img)
        
        save_path = os.path.join(mps_path, "aria", "aria_video_vis.mp4")
        create_video_from_frames(
            frames=frames_all, 
            save_path=save_path,
            fps=aria_cam.fps)
    
    visualize_all()

# conda activate aria
# cd src
# python -m aria.AriaHands --mps_path "./data/mps_open_cabinet_5_vrs/"