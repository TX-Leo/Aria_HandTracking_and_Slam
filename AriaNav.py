# TODO
# 1. 可视化Nav （我现在想加一个功能，那就是把头部3d的waypoints（映射到地面上的）可视化到最终的video_all里，最好是彩虹色的，而且是那种最近20个点（举例）的可视化，像人在沿着这个轨迹行走，请你告诉我应该如何修改代码，而且给我一个测试脚本来运行这个功能，其他的功能先不用跑，因为会很耗时间

# -*- coding: utf-8 -*-
# @FileName: AriaNav.py

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict
from scipy.optimize import minimize

@dataclass
class AriaNav:
    """
    [数据结构] 定义单帧导航相关数据。
    包含人(Agent)的位置和映射后的机器人(Robot)位置及速度。
    """
    agent_pos_3d: np.ndarray = None # 人类头部 3D 位置
    agent_pos_2d: np.ndarray = None # 投影到 2D 平面的位置 (x, y, yaw)
    robot_pos_2d: np.ndarray = None # 优化后的机器人 2D 位置
    robot_v: float = 0.0            # 线速度
    robot_w: float = 0.0            # 角速度
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_theta: float = 0.0

class AriaNavOptimizer:
    def __init__(self, dt: float = 0.1, 
                 lambda_pos: float = 32.0, 
                 lambda_yaw: float = 2.0, 
                 lambda_smooth: float = 1.0):
        """
        初始化导航优化器。
        Args:
            dt: 时间步长 (1/FPS)
            lambda_pos: 位置误差权重
            lambda_yaw: 角度误差权重
            lambda_smooth: 平滑度权重
        """
        self.dt = dt
        self.weights = {'pos': lambda_pos, 'yaw': lambda_yaw, 'smooth': lambda_smooth}
        self.v_limit = 1.2 # 速度限制 m/s
        self.w_limit = 1.5 # 角速度限制 rad/s

    def _wrap_angle(self, theta: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def _kinematics_step(self, state: np.ndarray, v: float, w: float) -> np.ndarray:
        """运动学模型推演一步"""
        x, y, theta = state
        new_x = x + v * np.cos(theta) * self.dt
        new_y = y + v * np.sin(theta) * self.dt
        new_theta = self._wrap_angle(theta + w * self.dt)
        return np.array([new_x, new_y, new_theta])

    def optimize_trajectory(self, human_target_poses_2d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        核心优化函数：将人类轨迹平滑并重定向为符合机器人运动学的轨迹。
        """
        N = len(human_target_poses_2d)
        if N < 2:
            return {'optimized_poses': human_target_poses_2d, 'velocities': np.zeros((N, 2))}
        
        print(f"[Nav] 开始优化导航轨迹，长度: {N} 帧...")
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

        # 使用 L-BFGS-B 求解器
        # 注意: disp 选项在 SciPy 新版中已弃用，此处已移除
        res = minimize(objective, x_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
        
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

class AriaNavEvaluator:
    @staticmethod
    def evaluate_and_plot(human_poses: np.ndarray, robot_poses: np.ndarray, 
                          robot_vels: np.ndarray, save_path: str, dt: float):
        """
        生成导航质量评估报告图表，对比人类原始轨迹和机器人优化轨迹。
        """
        print("[Nav Eval] 正在生成导航质量评估报告...")
        N = len(human_poses)
        time_axis = np.arange(N) * dt
        pos_error = np.linalg.norm(human_poses[:, :2] - robot_poses[:, :2], axis=1)
        mean_err = np.mean(pos_error)
        rmse = np.sqrt(np.mean(pos_error**2))
        
        print(f"  > 平均位置误差: {mean_err:.4f} m")
        print(f"  > RMSE: {rmse:.4f} m")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Navigation Retargeting Evaluation\nRMSE: {rmse:.3f}m", fontsize=16)
        
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