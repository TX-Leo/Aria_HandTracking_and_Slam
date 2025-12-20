# TODO
# 1. 添加estimate joint angle的功能和参数

# -*- coding: utf-8 -*-
# @FileName: AriaHand.py

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# Project Aria Tools 依赖
from projectaria_tools.core import mps

# [EgoZero] 抓取判定阈值 (单位: 米)
GRASP_THRESHOLD = 0.05 

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
    hand_keypoints_3d: np.ndarray = None # 3D 关键点 (21, 3)
    hand_keypoints_2d: np.ndarray = None # 2D 投影点 (21, 2)
    grasp_state: int = 0                # 0: Open, 1: Closed

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

        return AriaHand(
            is_right=is_right, confidence=confidence,
            wrist_pose=wrist_pose, palm_pose=palm_pose,
            hand_keypoints_3d=keypoints_3d_cam, hand_keypoints_2d=keypoints_2d,
            grasp_state=grasp_state
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
    def draw_skeleton(img, hand: AriaHand):
        """在图像上绘制手部骨架和关键点"""
        kpts_2d = hand.hand_keypoints_2d
        if kpts_2d is None: return img
        
        is_grasp = (hand.grasp_state == 1)
        # 抓取时变为红色，否则白色
        main_color = (0, 0, 255) if is_grasp else (255, 255, 255)
        
        # 绘制骨骼连线
        for conn in mps.hand_tracking.kHandJointConnections:
            pt1 = tuple(kpts_2d[int(conn[0])].astype(int))
            pt2 = tuple(kpts_2d[int(conn[1])].astype(int))
            if pt1 != (0,0) and pt2 != (0,0): 
                cv2.line(img, pt1, pt2, main_color, 2)
        
        # 绘制关键点圆圈
        for i, pt in enumerate(kpts_2d):
            pt_tuple = tuple(pt.astype(int))
            if pt_tuple == (0,0): continue
            # 腕关节(0号点)灰色，其他黄色，抓取时全红
            pt_col = main_color if is_grasp else ((255,255,0) if i>0 else (150,150,150))
            cv2.circle(img, pt_tuple, 5, pt_col, -1)
        return img

    @staticmethod
    def draw_axis(img, pose, k, d):
        """绘制 3D 坐标轴 (RGB = XYZ)"""
        if pose is None: return img
        r_vec, _ = cv2.Rodrigues(pose[:3, :3])
        t_vec = pose[:3, 3]
        
        # 轴长 5cm
        axis_len = 0.05
        axis_pts = np.float32([[axis_len,0,0], [0,axis_len,0], [0,0,axis_len], [0,0,0]])
        
        img_pts, _ = cv2.projectPoints(axis_pts, r_vec, t_vec, k, d)
        img_pts = [tuple(pt.ravel().astype(int)) for pt in img_pts]
        
        origin = img_pts[3]
        # 简单边界检查
        h, w = img.shape[:2]
        if 0 <= origin[0] < w and 0 <= origin[1] < h:
            cv2.line(img, origin, img_pts[0], (0, 0, 255), 3) # X - Red
            cv2.line(img, origin, img_pts[1], (0, 255, 0), 3) # Y - Green
            cv2.line(img, origin, img_pts[2], (255, 0, 0), 3) # Z - Blue
        return img
    
