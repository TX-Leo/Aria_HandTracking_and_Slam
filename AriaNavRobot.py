# -*- coding: utf-8 -*-
# @FileName: AriaNavRobot.py
import os
import json
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from scipy.optimize import minimize
from dataclasses import dataclass, field
import argparse
from tqdm import tqdm

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeDomain

from utils.utils_media import create_video_from_frames
from utils.utils_vis import draw_glass_rect, draw_arc_gauge
from aria.AriaCam import AriaCamGenerator
from aria.AriaNavHuman import AriaNavHuman, AriaNavHumanData, AriaNavHumanOps,AriaNavHumanGenerator, NAV_SAMPLE_N

@dataclass
class AriaNavRobotData:
    ts: int = 0
    x: float = 0.0     #m
    y: float = 0.0 
    theta: float = 0.0 # 单位：degree
    v: float = 0.0     # 线速度 m/s
    w: float = 0.0     # 角速度 rad/s
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_theta: float = 0.0 # 单位：degree

@dataclass
class AriaNavRobot:
    tss: List[int] = field(default_factory=list)
    traj: List[AriaNavRobotData] = field(default_factory=list)
    tss_aug: List[int] = field(default_factory=list)
    traj_aug: List[AriaNavRobotData] = field(default_factory=list)
    mps_path: str = None

    def __len__(self) -> int:
        return len(self.traj)

    @staticmethod
    def _safe_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else arr
    
    def save_aria_nav_robot_json(self):
        for i in range(len(self.tss)):
            frame_dir = os.path.join(self.mps_path, "aria", "all_data", f"{i:05d}")
            os.makedirs(frame_dir, exist_ok=True)

            current_data = self.traj[i]

            # 提取对应的增广采样段 (NAV_SAMPLE_N 个点)
            seg_start = i * NAV_SAMPLE_N
            seg_end = (i + 1) * NAV_SAMPLE_N
            segment_data = self.traj_aug[seg_start:seg_end]

            # 如果最后一帧越界，Python 切片会自动处理，但为了保险可以手动补齐或处理
            nav_samples = []
            for s in segment_data:
                nav_samples.append({
                    "ts": s.ts,
                    "x": s.x, 
                    "y": s.y, 
                    "theta": s.theta,
                    "v": s.v, 
                    "w": s.w,
                    "delta_x": s.delta_x,
                    "delta_y": s.delta_y,
                    "delta_theta": s.delta_theta
                })

            json_data = {
                "idx": i,
                "ts": current_data.ts,
                "x": current_data.x,
                "y": current_data.y,
                "theta": current_data.theta,
                "v": current_data.v,
                "w": current_data.w,
                "delta_x": current_data.delta_x,
                "delta_y": current_data.delta_y,
                "delta_theta": current_data.delta_theta,
                "nav_robot_traj_aug": nav_samples
            }

            with open(os.path.join(frame_dir, "aria_nav_robot.json"), 'w') as f:
                json.dump(json_data, f, indent=4)

class AriaNavRobotGenerator:
    def __init__(self, mps_path: str, aria_nav_human: AriaNavHuman):
        self.mps_path = mps_path
        self.aria_nav_human = aria_nav_human

    def get_aria_nav_robot(self, run_aug: bool = False) -> Optional[AriaNavRobot]:
        print("Processing Aria Nav Robot (Orig)")
        self.human_traj = self.aria_nav_human.traj
        tss, traj = self._human_traj_to_robot_traj(self.human_traj, label="orig")
        
        tss_aug, traj_aug = [], [] # 初始化为空列表
        if run_aug:
            print("Processing Aria Nav Robot (Aug)")
            self.human_traj_aug = self.aria_nav_human.traj_aug
            tss_aug, traj_aug = self._human_traj_to_robot_traj(self.human_traj_aug, label="aug")
        else:
            print("Skipping Aria Nav Robot (Aug) to save time.")

        aria_nav_robot = AriaNavRobot(tss, traj, tss_aug, traj_aug)
        aria_nav_robot.mps_path = self.mps_path
        return aria_nav_robot

    def _human_traj_to_robot_traj(self, human_traj: List[AriaNavHumanData], label: str):
        if not human_traj: return [], []

        human_poses = []
        for data in human_traj:
            human_poses.append([data.x, data.y, math.radians(data.theta)])
        human_poses_np = np.array(human_poses)

        if len(human_traj) > 1:
            dt = (human_traj[1].ts - human_traj[0].ts) / 1e9
        else:
            dt = 0.1

        optimizer = AriaNavHuman2Robot(dt=dt)
        opt_res = optimizer.optimize(human_poses_np)
        
        robot_poses_np = opt_res['poses'] # N x 3
        robot_vels_np = opt_res['vels']   # N x 2
        
        plot_path = os.path.join(self.mps_path, "aria", f"aria_nav_robot_analysis_{label}.png")
        AriaNavHuman2RobotEvaluator.evaluate_and_plot(human_poses_np, robot_poses_np, robot_vels_np, plot_path, dt)

        tss = []
        traj = []

        ########### 为了易读性做的处理 ##############
        # robot_poses_np = -robot_poses_np
        # robot_vels_np = -robot_vels_np
        ##########################################

        start_pose = robot_poses_np[0]

        for k in range(len(human_traj)):
            ts = human_traj[k].ts
            curr_p = robot_poses_np[k]
            dt_raw_rad = curr_p[2] - start_pose[2]
            dt_norm_rad = math.atan2(math.sin(dt_raw_rad), math.cos(dt_raw_rad))
            aria_nav_robot_data = AriaNavRobotData(
                ts = ts,
                x = curr_p[0], 
                y = curr_p[1], 
                theta = math.degrees(curr_p[2]),
                v = robot_vels_np[k][0], 
                w = robot_vels_np[k][1],
                delta_x = curr_p[0] - start_pose[0],
                delta_y = curr_p[1] - start_pose[1],
                delta_theta = math.degrees(dt_norm_rad)
            )
            tss.append(ts)
            traj.append(aria_nav_robot_data)
        return tss, traj

class AriaNavHuman2Robot:
    """
    [算法类] 将人类轨迹平滑并重定向为符合机器人运动学的轨迹。
    
    参数调试指南 (Tuning Guide):
    ---------------------------------------------------------
    1. lambda_pos: [位置权重]
       - 想要机器人“寸步不离”紧跟人类路径？ -> 调大 (如 300+)
       - 想要机器人过滤人类走路时的左右晃动(蛇形位移)？ -> 调小 (如 50-100)
       
    2. lambda_yaw: [航向权重]
       - 想要机器人始终面向眼镜正前方(即人眼看的方向)？ -> 调大 (如 20+)
       - 发现机器人为了对准角度频繁原地打转？ -> 调小 (如 1-2)
       
    3. lambda_smooth: [平滑/加速度权重]
       - 发现机器人速度指令(v, w图)锯齿严重、加减速太猛？ -> 调大 (如 5-10)
       - 发现机器人反应迟钝，拐弯半径过大，跟不上人的节奏？ -> 调小 (如 0.1-0.5)
    ---------------------------------------------------------
    """
    def __init__(self, dt: float, lambda_pos=10.0, lambda_yaw=2.0, lambda_smooth=1.0):
        self.dt = dt
        self.weights = {'pos': lambda_pos, 'yaw': lambda_yaw, 'smooth': lambda_smooth}
        self.v_limit = 1.2 # 最大线速度 m/s
        self.w_limit = 1.5 # 最大角速度 rad/s

    def _wrap_angle(self, theta: float) -> float:
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def _kinematics_step(self, state, v, w):
        x, y, theta = state
        new_x = x + v * np.cos(theta) * self.dt
        new_y = y + v * np.sin(theta) * self.dt
        new_theta = self._wrap_angle(theta + w * self.dt)
        return np.array([new_x, new_y, new_theta])

    def objective(self, z, human_poses_2d, N):
        cost = 0.0
        current_state = human_poses_2d[0].copy()
        vs, ws = z[0::2], z[1::2]
        for k in range(N):
            target = human_poses_2d[k]
            # 1. 位置代价
            cost += self.weights['pos'] * np.sum((current_state[:2] - target[:2])**2)
            # 2. 航向代价 (如果你头部很稳，这个权重一定要高)
            cost += self.weights['yaw'] * (self._wrap_angle(current_state[2] - target[2])**2)
            # 3. 平滑代价 (惩罚加速度)
            if k > 0:
                cost += self.weights['smooth'] * ((vs[k]-vs[k-1])**2 + (ws[k]-ws[k-1])**2)
            else:
                # 初始静止约束：希望第一帧速度接近0
                cost += self.weights['smooth'] * (vs[k]**2 + ws[k]**2)
            current_state = self._kinematics_step(current_state, vs[k], ws[k])
        return cost

    def optimize(self, human_poses_2d: np.ndarray) -> Dict:
        """
        输入 N x 3 (x, y, yaw_rad) 的人类轨迹，返回优化后的状态和速度。
        """
        N = len(human_poses_2d)
        
        # 设置位移初值和bounds
        x_init = np.zeros(2 * N)
        x_init[0::2] = 0.1  # 启发式初值：初始速度给一个很小的值，给优化器一个“向前”的暗示
        # 约束：v 必须在 [0, v_limit] 之间，严禁倒车
        bounds = []
        for i in range(2 * N):
            if i % 2 == 0:
                bounds.append((0.0, self.v_limit)) # 修改点：下限设为0
            else:
                bounds.append((-self.w_limit, self.w_limit))
        res = minimize(self.objective, x_init, args=(human_poses_2d, N),method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-6}) # 增加迭代次数保证收敛
        
        opt_vs, opt_ws = res.x[0::2], res.x[1::2]
        opt_path = []
        curr = human_poses_2d[0].copy()
        for k in range(N):
            opt_path.append(curr)
            curr = self._kinematics_step(curr, opt_vs[k], opt_ws[k])
            
        return {'poses': np.array(opt_path), 'vels': np.stack([opt_vs, opt_ws], axis=1)}
    
class AriaNavHuman2RobotEvaluator:
    @staticmethod
    def evaluate_and_plot(human_poses, robot_poses, robot_vels, save_path, dt):
        """
        绘制高级、专业的可视化图表。
        """
        if not save_path: return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        N = len(human_poses)
        time_axis = np.arange(N) * dt
        pos_error = np.linalg.norm(human_poses[:, :2] - robot_poses[:, :2], axis=1)
        rmse = np.sqrt(np.mean(pos_error**2))
        max_err = np.max(pos_error)

        # 执行终端打印分析
        AriaNavHuman2RobotEvaluator.print_summary_and_eval(rmse, max_err, robot_vels, N, dt)

        # 设置绘图风格
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f"Aria Navigation: Human vs Robot Trajectory Evaluation\nRMSE: {rmse:.4f} m", 
                     fontsize=20, fontweight='bold', y=0.98)

        # 颜色定义
        color_human, color_robot = '#E74C3C', '#3498DB'
        color_v, color_w, color_error = '#2ECC71', '#F1C40F', '#9B59B6'

        # --- 1. Map View (增加了 Start/End 标记) ---
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(human_poses[:, 0], human_poses[:, 1], color=color_human, linestyle='--', alpha=0.4, label='Human Path')
        ax1.plot(robot_poses[:, 0], robot_poses[:, 1], color=color_robot, linewidth=2.5, label='Robot Path')
        
        # 绘制起点和终点 (视觉优化)
        ax1.scatter(robot_poses[0, 0], robot_poses[0, 1], color='green', s=100, edgecolors='black', zorder=5, label='Start')
        ax1.scatter(robot_poses[-1, 0], robot_poses[-1, 1], color='red', s=100, edgecolors='black', zorder=5, label='End')
        ax1.text(robot_poses[0, 0], robot_poses[0, 1]+0.05, "START", color='green', fontweight='bold', ha='center')
        ax1.text(robot_poses[-1, 0], robot_poses[-1, 1]+0.05, "END", color='red', fontweight='bold', ha='center')

        # 抽稀画箭头
        step = max(1, N // 15)
        ax1.quiver(robot_poses[::step, 0], robot_poses[::step, 1], 
                   np.cos(robot_poses[::step, 2]), np.sin(robot_poses[::step, 2]), 
                   color=color_robot, scale=25, width=0.006)
        
        ax1.set_title("Top-Down Map View (Trajectory & Heading)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("X Position (meters)"); ax1.set_ylabel("Y Position (meters)")
        ax1.legend(loc='best'); ax1.axis('equal')

        # --- 2. X Comparison ---
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(time_axis, human_poses[:, 0], color=color_human, linestyle='--', label='Human X')
        ax2.plot(time_axis, robot_poses[:, 0], color=color_robot, label='Robot X')
        ax2.set_title("X Position Comparison", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("X (m)"); ax2.legend()

        # --- 3. Y Comparison ---
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(time_axis, human_poses[:, 1], color=color_human, linestyle='--', label='Human Y')
        ax3.plot(time_axis, robot_poses[:, 1], color=color_robot, label='Robot Y')
        ax3.set_title("Y Position Comparison (Lateral)", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Y (m)"); ax3.legend()

        # --- 4. Heading Comparison ---
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(time_axis, np.degrees(human_poses[:, 2]), color=color_human, linestyle='--', label='Human Theta')
        ax4.plot(time_axis, np.degrees(robot_poses[:, 2]), color=color_robot, label='Robot Theta')
        ax4.set_title("Heading Angle (Orientation)", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Theta (deg)"); ax4.legend()

        # --- 5. Control Commands (v, w) ---
        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(time_axis, robot_vels[:, 0], color=color_v, label='v (m/s)')
        ax5.set_ylabel("v (m/s)", color=color_v)
        ax5_twin = ax5.twinx()
        ax5_twin.plot(time_axis, robot_vels[:, 1], color=color_w, label='w (rad/s)')
        ax5_twin.set_ylabel("w (rad/s)", color=color_w)
        ax5.set_title("Robot Control Commands (v, w)", fontsize=14, fontweight='bold')
        ax5.set_xlabel("Time (s)")
        lines, labels = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines + lines2, labels + labels2, loc='upper right')

        # --- 6. Tracking Error ---
        ax6 = plt.subplot(3, 2, 6)
        ax6.fill_between(time_axis, pos_error, color=color_error, alpha=0.3)
        ax6.plot(time_axis, pos_error, color=color_error, label='L2 Error')
        ax6.axhline(y=rmse, color='black', linestyle=':', label=f'RMSE ({rmse:.3f}m)')
        ax6.set_title("Positioning Tracking Error", fontsize=14, fontweight='bold')
        ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Error (m)"); ax6.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300); plt.close()
        print(f"Professional evaluation plot saved to: {save_path}")

    @staticmethod
    def print_summary_and_eval(rmse, max_err, vels, N, dt):
        """
        打印关键定性分析结果。
        """
        avg_v = np.mean(vels[:, 0])
        max_v = np.max(vels[:, 0])
        avg_w_abs = np.mean(np.abs(vels[:, 1]))
        
        print("\n" + "═"*50)
        print("      ARIA HUMAN-TO-ROBOT OPTIMIZATION SUMMARY")
        print("═"*50)
        print(f"Trajectory Duration : {N*dt:.2f} seconds ({N} points)")
        print(f"Mean Tracking Error : {rmse*100:.2f} cm")
        print(f"Max Tracking Error  : {max_err*100:.2f} cm")
        
        print("\nKinematic Statistics:")
        print(f"  - Average Velocity (v): {avg_v:.3f} m/s")
        print(f"  - Peak Velocity (v_max): {max_v:.3f} m/s")
        print(f"  - Avg Angular Speed|w|: {avg_w_abs:.3f} rad/s")

        print("\nQualitative Evaluation:")
        if rmse < 0.03:
            res = "EXCELLENT: Near-perfect human path replication."
        elif rmse < 0.10:
            res = "GOOD: Smooth robot motion with minor deviations."
        else:
            res = "AVERAGE: Significant filtering, likely due to jitter."
        print(f"  - Path Fidelity: {res}")

        if max_v > 1.1:
            print("  - Alert: Robot reached near-maximum speed limits.")
        
        print("═"*50 + "\n")

class AriaNavRobotOps:
    @staticmethod
    def draw_aria_nav_robot_panel(img, aria_nav_robot_data: AriaNavRobotData):
        LABEL_COLOR = (200, 200, 200)
        DELTA_COLOR = (255, 150, 50)
        VAL_COLOR = (0, 255, 255) 
    
        img = draw_glass_rect(img, (10, 40), (240, 360), alpha=0.6)
        
        # 计算增量 Delta
        x = aria_nav_robot_data.x
        y = aria_nav_robot_data.y
        theta = aria_nav_robot_data.theta
        v = aria_nav_robot_data.v
        w = aria_nav_robot_data.w
        delta_x = aria_nav_robot_data.delta_x
        delta_y = aria_nav_robot_data.delta_y
        delta_theta = aria_nav_robot_data.delta_theta
        
        # 1. V 仪表盘 (圆弧) + 文字
        draw_arc_gauge(img, (45, 105), v, 1.0, "V")
        cv2.putText(img, f"LINEAR SPEED", (90, 95), cv2.FONT_HERSHEY_DUPLEX , 0.4, LABEL_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, f"{v:>5.2f} m/s", (90, 115), cv2.FONT_HERSHEY_DUPLEX , 0.5, VAL_COLOR, 1, cv2.LINE_AA)

        # 2. W 仪表盘 (圆弧) + 文字
        draw_arc_gauge(img, (45, 175), w, 1.0, "W") # 这里的 1.0 是 max_w
        cv2.putText(img, f"ANGULAR SPEED", (90, 165), cv2.FONT_HERSHEY_DUPLEX , 0.4, LABEL_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, f"{w:>5.2f} r/s", (90, 185), cv2.FONT_HERSHEY_DUPLEX , 0.5, VAL_COLOR, 1, cv2.LINE_AA)

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
        # 你可以根据实际行走范围调整这个值，例如 0.5 (极灵敏) 或 2.5 (适中)
        SENSITIVITY_RANGE = 2.5
        
        half_width = (bx2 - bx1) // 2
        # 计算映射像素 (delta_x 和 delta_y 使用之前计算好的 Delta)
        map_px = int(cx + (delta_x / SENSITIVITY_RANGE) * half_width)
        map_py = int(cy - (delta_y / SENSITIVITY_RANGE) * half_width) # 图像 Y 轴向下，所以用减

        # 边缘约束：防止点跑出方框
        map_px = max(bx1 + 2, min(map_px, bx2 - 2))
        map_py = max(by1 + 2, min(map_py, by2 - 2))
        
        # 绘制当前位置：核心点 + 外圈发光效果
        cv2.circle(img, (map_px, map_py), 3, VAL_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, (map_px, map_py), 5, VAL_COLOR, 1, cv2.LINE_AA) # 外圈发光
        
        # 右侧文字信息
        cv2.putText(img, "POSITION (X, Y)", (90, 230), cv2.FONT_HERSHEY_DUPLEX , 0.4, LABEL_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, f"Abs: {x:>5.2f}, {y:>5.2f}m", (90, 250), cv2.FONT_HERSHEY_DUPLEX , 0.4, VAL_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, f"Dlt: {delta_x:>5.2f}, {delta_y:>5.2f}m", (90, 268), cv2.FONT_HERSHEY_DUPLEX , 0.4, DELTA_COLOR, 1, cv2.LINE_AA)

        # 4. Theta 罗盘仪表盘 + 文字
        c_cx, c_cy = 45, 315; c_r = 20
        
        # 绘图逻辑需要弧度：
        
        # 绘制背景
        cv2.circle(img, (c_cx, c_cy), c_r, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.line(img, (c_cx, c_cy - c_r), (c_cx, c_cy - c_r + 5), (200, 200, 200), 1, cv2.LINE_AA)
        
        # 绘制彩色填充弧
        # color_theta 逻辑
        color_theta = (0, 255, 127) if delta_theta >= 0 else (200, 0, 255)
        # cv2.ellipse 的角度参数本来就是 Degree，直接用 delta_theta
        cv2.ellipse(img, (c_cx, c_cy), (c_r, c_r), 0, -90, -90 + int(delta_theta), color_theta, 2, cv2.LINE_AA)

        # 绘制指针 (这里需要弧度)
        p_rad = np.radians(delta_theta) - math.pi/2 
        px = int(c_cx + c_r * math.cos(p_rad))
        py = int(c_cy + c_r * math.sin(p_rad))
        cv2.line(img, (c_cx, c_cy), (px, py), color_theta, 2, cv2.LINE_AA)
        cv2.circle(img, (c_cx, c_cy), 2, (255, 255, 255), -1)
        
        # 文字展示
        cv2.putText(img, "ORIENTATION (T)", (90, 305), cv2.FONT_HERSHEY_DUPLEX , 0.4, LABEL_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, f"Abs: {theta:>5.1f} deg", (90, 325), cv2.FONT_HERSHEY_DUPLEX , 0.4, VAL_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, f"Dlt: {delta_theta:>+5.1f} deg", (90, 343), cv2.FONT_HERSHEY_DUPLEX , 0.4, DELTA_COLOR, 1, cv2.LINE_AA)

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

    # get aria_nav_robot
    aria_nav_robot_generator = AriaNavRobotGenerator(mps_path, aria_nav_human)
    aria_nav_robot = aria_nav_robot_generator.get_aria_nav_robot(run_aug=True)

    # save data
    aria_nav_robot.save_aria_nav_robot_json()

    # visualize
    def visualize_all():
        frames_all = []
        for idx, ts in enumerate(tqdm(tss, desc="Visualization")):
            img = aria_cam.cam[idx].rgb.copy()

            ########## orig data #########
            # draw nav huamn traj
            img = AriaNavHumanOps.draw_aria_nav_human_traj(img, idx, aria_nav_human, aria_cam.cam[idx].k, aria_cam.cam[idx].c2w)

            ########### Panels ###########
            # nav robot panel
            img = AriaNavRobotOps.draw_aria_nav_robot_panel(img, aria_nav_robot.traj[idx])
            
            frames_all.append(img)
        
        save_path = os.path.join(mps_path, "aria", "aria_video_vis.mp4")
        create_video_from_frames(
            frames=frames_all, 
            save_path=save_path,
            fps=aria_cam.fps)
    
    visualize_all()

# conda activate aria
# cd src
# python -m aria.AriaNavRobot --mps_path "./data/mps_open_cabinet_5_vrs/"