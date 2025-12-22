# TODO
# 1. 添加estimate joint angle的功能和参数

# -*- coding: utf-8 -*-
# @FileName: AriaHand.py

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

# Project Aria Tools 依赖
from projectaria_tools.core import mps

# [EgoZero] 抓取判定阈值 (单位: 米)
GRASP_THRESHOLD = 0.05 

@dataclass
class AriaJointAngle:
    """
    [新增类] 存储手部 20 个关节角度数据 (基于 emg2pose 定义)。
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
class AriaHand:
    """
    [数据结构] 定义单帧中一只手的数据。
    包含置信度、3D/2D 关键点、手腕/手掌位姿以及抓取状态。
    """
    is_right: bool
    confidence: float
    wrist_pose: np.ndarray = None       # 手腕 6D 位姿 (4, 4)
    palm_pose: np.ndarray = None        # 手掌 6D 位姿 (4, 4)
    hand_keypoints_3d: np.ndarray = None # 3D 关键点 (21, 3) (for current camera frame)
    hand_keypoints_2d: np.ndarray = None # 2D 投影点 (21, 2)
    grasp_state: int = 0                # 0: Open, 1: Closed
    joint_angles: Optional[AriaJointAngle] = None

class AriaHandOps:
    """
    [工具类] 提供手部数据的处理、计算和可视化功能。
    """

    @staticmethod
    def process_single_hand(hand_mps, T_camera_to_device, k_matrix, h_orig, w_orig, is_right) -> Optional[AriaHand]:
        """
        处理 MPS 返回的单只手数据，转换为 AriaHand 对象。
        
        Args:
            hand_mps: MPS 手部数据对象
            T_camera_to_device: 相机到设备的变换矩阵
            k_matrix: 相机内参矩阵
            h_orig: 原始图像高度 (旋转前)
            w_orig: 原始图像宽度 (旋转前)
            is_right: 是否为右手
        """
        if hand_mps is None: return None
        confidence = hand_mps.confidence
        if confidence <= 0.1: return None
        
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
        palm_pose, wrist_pose = AriaHandOps._compute_wrist_and_palm_pose(
            palm_normal, palm_position, wrist_normal, wrist_position, confidence, T_camera_to_device
        )
        
        # 4. 坐标变换与投影
        keypoints_3d_cam = AriaHandOps._transform_keypoints_to_camera(hand_mps, T_camera_to_device)
        keypoints_2d, _ = AriaHandOps._project_points_rotated(keypoints_3d_cam, k_matrix, h_orig, w_orig)

        # 5. joint angle
        joint_angles = AriaJointAngle.from_keypoints_3d(keypoints_3d_cam)

        return AriaHand(
            is_right=is_right,
            confidence=confidence,
            wrist_pose=wrist_pose,
            palm_pose=palm_pose,
            hand_keypoints_3d=keypoints_3d_cam, 
            hand_keypoints_2d=keypoints_2d,
            grasp_state=grasp_state,
            joint_angles=joint_angles
        )

    @staticmethod
    def _compute_wrist_and_palm_pose(palm_n, palm_p, wrist_n, wrist_p, conf, T_cam_dev):
        """计算手腕和手掌的 6D Pose 矩阵"""
        if np.linalg.norm(wrist_n) == 0 or np.linalg.norm(palm_n) == 0: return None, None
        
        # OpenCV 坐标系修正 (y down, z forward)
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
        
        # 变换到相机坐标系
        return T_opencv_to_aria @ T_cam_dev @ palm_mat, T_opencv_to_aria @ T_cam_dev @ wrist_mat

    @staticmethod
    def _transform_keypoints_to_camera(hand, T_cam_dev):
        """将关键点从 Device 坐标系变换到 Camera 坐标系"""
        R = T_cam_dev[:3, :3]
        t = T_cam_dev[:3, 3]
        kpts_dev = np.array(hand.landmark_positions_device)
        kpts_cam = np.einsum("ij, nj -> ni", R, kpts_dev) + t
        return kpts_cam

    @staticmethod
    def _project_points_rotated(points_cam, k, h_orig, w_orig):
        """
        将 3D 相机坐标投影到 2D 像素坐标，并处理图像旋转 (90度)。
        输入: points_cam (N, 3), K (3, 3), h_orig (Landscape Height), w_orig (Landscape Width)
        输出: projected (N, 2), valid_mask (N,)
        """
        z = points_cam[:, 2]
        valid_mask = z > 1e-3
        if not np.any(valid_mask):
            return np.zeros((len(points_cam), 2)), np.zeros(len(points_cam), dtype=bool)

        homo_pix = (k @ points_cam.T).T
        u = homo_pix[:, 0] / homo_pix[:, 2]
        v = homo_pix[:, 1] / homo_pix[:, 2]
        
        # 旋转 90 度顺时针的坐标变换公式
        # new_u = h - 1 - v
        # new_v = u
        u_new = h_orig - 1 - v
        v_new = u
        projected = np.stack((u_new, v_new), axis=-1)
        
        # 旋转后的宽高互换
        h_new, w_new = w_orig, h_orig
        in_bounds = (0 <= u_new) & (u_new < h_new) & (0 <= v_new) & (v_new < w_new)
        return projected, valid_mask & in_bounds
    
    @staticmethod
    def draw_skeleton(img, hand):
        """ 优雅的手部骨架绘制：严格适配 Aria 官方点序 (5-Wrist, 20-Palm) """
        pts = hand.hand_keypoints_2d.astype(np.int32)
        is_grasp = (hand.grasp_state == 1)
        
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
    def draw_axis(img, pose, k, d):
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
    
