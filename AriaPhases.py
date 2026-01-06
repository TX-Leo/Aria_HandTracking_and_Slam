# -*- coding: utf-8 -*-
# @FileName: AriaPhase.py

import os
import json
import cv2
import numpy as np
from scipy.signal import medfilt
from typing import List, Dict
from tqdm import tqdm
import argparse
from dataclasses import dataclass, field
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeDomain

from utils.utils_media import create_video_from_frames
from utils.utils_vis import draw_glass_rect
from aria.AriaCam import AriaCamGenerator
from aria.AriaNavRobot import AriaNavRobot, AriaNavRobotOps, AriaNavRobotGenerator
from aria.AriaHands import AriaHands, AriaHandsOps, AriaHandsGenerator
from aria.AriaNavHuman import AriaNavHumanOps,AriaNavHumanGenerator

# [Config] 阶段判定参数
V_ROBOT_THRESHOLD = 0.10  # 机器人静止阈值 (m/s)
V_HAND_THRESHOLD = 0.10   # 手部主动运动阈值 (m/s)
GRASP_CONF_THRESHOLD = 0.7 # 抓取信号稳定阈值 (0-1)
# TODO: 从手动设置修改成自动计算成fps+1
SMOOTH_WINDOW = 11        # 信号平滑窗口大小
SMOOTH_KERNEL_SIZE = 11

# 阶段定义
PHASE_NAV = 0    # Navigation
PHASE_GRASP = 1  # Grasp/Reach (手部伸向物体)
PHASE_MANIP = 2  # Manipulation (抓稳并操作)
PHASE_IDLE = 3   # Idle/Static (待剔除数据)

PHASE_NAMES = { 
    PHASE_NAV: "NAVIGATION", 
    PHASE_GRASP: "GRASP/REACH", 
    PHASE_MANIP: "MANIPULATION",
    PHASE_IDLE: "IDLE/STATIC" 
}
PHASE_COLORS_BGR = { 
    PHASE_NAV: (0, 255, 127),    # 嫩绿
    PHASE_GRASP: (0, 215, 255),  # 金黄
    PHASE_MANIP: (0, 127, 255),   # 橙红
    PHASE_IDLE: (150, 150, 150)  # 灰色
}

@dataclass
class AriaPhasesData:
    ts: int = 0
    phase: int = 0
    v_robot: float = 0.0
    v_hand: float = 0.0
    grasp_conf: float = 0.0 # 是grasp_state经过信号平滑窗口之后的结果
    grasp_state: int = 0 

@dataclass
class AriaPhases:
    tss: List[int] = field(default_factory=list)
    phases: List[AriaPhasesData] = field(default_factory=list)
    mps_path: str = None

    windows: Dict[str, List[List[int]]] = field(default_factory=lambda: {"nav_windows": [], "grasp_windows": [], "manip_windows": [], "idle_windows": []})
    key_indices: Dict[str, int] = field(default_factory=lambda: {"start_idx": -1, "end_idx": -1, "nav2grasp_idx": -1, "grasp2manip_idx": -1})

    def __len__(self):
        return len(self.tss)
    
    def save_aria_phases_json(self):
        for i in range(len(self.tss)):
            frame_dir = os.path.join(self.mps_path, "aria", "all_data", f"{i:05d}")
            os.makedirs(frame_dir, exist_ok=True)

            data = self.phases[i]
            json_data = {
                "idx": i,
                "ts": data.ts,
                "phase": data.phase,
                "phase_label": PHASE_NAMES.get(data.phase, "UNKNOWN"),
                "v_robot": data.v_robot,
                "v_hand": data.v_hand,
                "grasp_conf": data.grasp_conf,
                "grasp_state": data.grasp_state
            }

            with open(os.path.join(frame_dir, "aria_phases.json"), 'w') as f:
                json.dump(json_data, f, indent=4)
        
        self._save_summary_json()
        
    def _save_summary_json(self):
        save_path = os.path.join(self.mps_path, "aria", "aria_phases_results.json")
        summary_data = {
            "total_frames": len(self),
            "windows": self.windows,
            "key_indices": self.key_indices
        }
        with open(save_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"Phase summary results saved to: {save_path}")

class AriaPhasesGenerator:
    def __init__(self, mps_path: str, aria_nav_robot: AriaNavRobot, aria_hands: AriaHands):
        self.mps_path = mps_path
        self.aria_nav_robot = aria_nav_robot
        self.aria_hands = aria_hands
    
    def get_aria_phases(self) -> AriaPhases:
        print("Processing Aria Phases")

        # 1. 提取原始特征并对齐
        n = len(self.aria_hands.hands)
        v_robot = np.array([self.aria_nav_robot.traj[i].v if i < len(self.aria_nav_robot.traj) else 0 for i in range(n)])

        # 提取手部速度 (取左右手最大值) 和 抓取状态
        v_hand_raw = []
        grasp_raw = []
        for h in self.aria_hands.hands:
            vh_r = h.hand_r.wrist_vel_world if h.hand_r else 0
            vh_l = h.hand_l.wrist_vel_world if h.hand_l else 0
            v_hand_raw.append(max(vh_r, vh_l))
            
            gr = h.hand_r.grasp_state if h.hand_r else 0
            gl = h.hand_l.grasp_state if h.hand_l else 0
            grasp_raw.append(max(gr, gl))

        # 信号平滑处理
        v_hand = medfilt(np.array(v_hand_raw), kernel_size=5)
        grasp_conf = np.convolve(np.array(grasp_raw), np.ones(SMOOTH_WINDOW)/SMOOTH_WINDOW, mode='same')

        # 2. 三阶段有限状态机 (FSM) 核心逻辑
        raw_phases = np.zeros(n, dtype=int)
        current_state = PHASE_IDLE  # 初始设为 IDLE
        hand_loss_timer = 0 # 容忍手部丢失的计数器

        for i in range(n):
            # 逻辑：每一帧都在状态机中流转
            v_low = v_robot[i] < V_ROBOT_THRESHOLD
            hand_visible = (self.aria_hands.hands[i].hand_r or self.aria_hands.hands[i].hand_l)
            hand_intent = v_hand[i] > V_HAND_THRESHOLD or grasp_conf[i] > 0.1
            
            # --- 状态: IDLE ---
            if current_state == PHASE_IDLE:
                if not v_low: # 机器人开始运动
                    current_state = PHASE_NAV
                # elif hand_visible and hand_intent:
                #     # 机器人没动但手出现了，准备抓取
                #     current_state = PHASE_GRASP

            # --- 状态: NAVIGATION ---
            if current_state == PHASE_NAV:
                if v_low and hand_visible and hand_intent:
                    # 切换到 GRASP 条件：机器人慢下来 + 手部有效出现
                    current_state = PHASE_GRASP
                elif v_low and not hand_visible:
                    # 如果运动停止且没手，回到 IDLE
                    current_state = PHASE_IDLE
            
            # --- 状态: GRASP ---
            elif current_state == PHASE_GRASP:
                if grasp_conf[i] > GRASP_CONF_THRESHOLD:
                    # 切换到 MANIP 条件：抓取信号稳定
                    current_state = PHASE_MANIP
                elif not v_low:
                    # 意外退出条件：机器人突然启动（说明只是路过，没抓）
                    current_state = PHASE_NAV
                elif v_low and not hand_visible:
                     # 增加容忍度
                    hand_loss_timer += 1
                    if hand_loss_timer > 5: 
                        current_state = PHASE_IDLE
                else: 
                    hand_loss_timer = 0

            # --- 状态: MANIPULATION ---
            elif current_state == PHASE_MANIP:
                if not v_low:
                    # 退出条件：机器人重新开始显著运动（说明操作完成，进入下一段导航）
                    current_state = PHASE_NAV
                elif v_low and (not hand_visible or grasp_conf[i] < 0.1):
                     # 松手或手离开
                    current_state = PHASE_IDLE

            raw_phases[i] = current_state
        
        # 3. 后处理：去除毛刺
        hand_visible_mask = np.array([(h.hand_r is not None or h.hand_l is not None) for h in self.aria_hands.hands])
        refined_phases = self._refine_phases(raw_phases, v_robot, hand_visible_mask)

        # 4. 封装结果
        aria_phases = AriaPhases(mps_path=self.mps_path)
        for i in range(n):
            data = AriaPhasesData(
                ts=self.aria_hands.tss[i], 
                phase=int(refined_phases[i]),
                v_robot=float(v_robot[i]),
                v_hand=float(v_hand[i]),
                grasp_conf=float(grasp_conf[i]),
                grasp_state=int(grasp_raw[i]) 
            )
            aria_phases.tss.append(data.ts)
            aria_phases.phases.append(data)

        self._analyze_phase_windows(aria_phases, refined_phases)# 执行窗口提取和关键索引分析

        # 5. 可视化与报告
        plot_path = os.path.join(self.mps_path, "aria", "aria_phases_analysis.png")
        AriaPhasesOps.save_advanced_phase_plot(aria_phases, plot_path)
        AriaPhasesOps.print_phase_summary(aria_phases)

        return aria_phases
    
    def _refine_phases(self, phases: List[int], v_robot: np.ndarray, hand_visible: np.ndarray) -> List[int]:
        """
        [精炼算法]：
        1. 使用中值滤波去除中间的瞬时毛刺。
        2. 强制恢复开头和结尾由于窗口投票丢失的 IDLE 状态。
        """
        n = len(phases)
        # 1. 执行中值滤波平滑 (处理中间跳变)
        refined = medfilt(phases, kernel_size=SMOOTH_KERNEL_SIZE).astype(int).tolist()

        # 2. 强制恢复开头的 IDLE 
        # 逻辑：从第一帧开始，只要满足静止且无手，就必须是 IDLE，直到遇到第一次动作
        for i in range(n):
            # 物理判定条件
            is_idle_physically = v_robot[i] < V_ROBOT_THRESHOLD
            if is_idle_physically:
                refined[i] = PHASE_IDLE
            else:
                # 一旦检测到真正的动作（移动或手部进入），停止恢复，保持滤波结果
                break

        # 3. 强制恢复结尾的 IDLE
        # 逻辑：从最后一帧往回找，只要满足静止且无手，就设为 IDLE
        for i in range(n - 1, -1, -1):
            is_idle_physically = (v_robot[i] < V_ROBOT_THRESHOLD and not hand_visible[i])
            if is_idle_physically:
                refined[i] = PHASE_IDLE
            else:
                # 回溯到最后一个动作点停止
                break

        return refined

    def _analyze_phase_windows(self, aria_phases: AriaPhases, phases: List[int]):
        """
        分析平滑后的序列：
        1. 提取所有阶段的起止窗口。
        2. 记录第一个 start_idx, nav2grasp_idx, grasp2manip_idx, 和结尾前的 end_idx。
        """
        n = len(phases)
        if n == 0: return

        # --- 1. 提取所有连续窗口并捕获【第一个】跳转点 ---
        windows = {PHASE_NAV: [], PHASE_GRASP: [], PHASE_MANIP: [], PHASE_IDLE: []}
        start_idx_current_segment = 0
        curr_val = phases[0]
        
        for i in range(1, n):
            if phases[i] != curr_val:
                # 结束当前段
                windows[curr_val].append([start_idx_current_segment, i - 1])
                
                # --- 核心修改：只记录第一个跳转点 ---
                # 记录第一个 NAV -> GRASP
                if curr_val == PHASE_NAV and phases[i] == PHASE_GRASP:
                    if aria_phases.key_indices["nav2grasp_idx"] == -1:
                        aria_phases.key_indices["nav2grasp_idx"] = i
                
                # 记录第一个 GRASP -> MANIP
                elif curr_val == PHASE_GRASP and phases[i] == PHASE_MANIP:
                    if aria_phases.key_indices["grasp2manip_idx"] == -1:
                        aria_phases.key_indices["grasp2manip_idx"] = i
                
                # 更新状态
                start_idx_current_segment = i
                curr_val = phases[i]
        
        # 记录最后一段
        windows[curr_val].append([start_idx_current_segment, n - 1])

        # 映射窗口
        aria_phases.windows["nav_windows"] = windows[PHASE_NAV]
        aria_phases.windows["grasp_windows"] = windows[PHASE_GRASP]
        aria_phases.windows["manip_windows"] = windows[PHASE_MANIP]
        aria_phases.windows["idle_windows"] = windows[PHASE_IDLE]

        # --- 2. 确定 start_idx (IDLE到第一个NAV) ---
        for i in range(n):
            if phases[i] == PHASE_NAV:
                aria_phases.key_indices["start_idx"] = i
                break
        
        # --- 3. 确定 end_idx (结尾IDLE开始前的最后一帧) ---
        # 逻辑：如果最后一段是 IDLE，end_idx 是 IDLE 开始前的那一帧；否则是最后一帧
        if windows[PHASE_IDLE] and windows[PHASE_IDLE][-1][1] == n - 1:
            last_idle_start = windows[PHASE_IDLE][-1][0]
            aria_phases.key_indices["end_idx"] = max(0, last_idle_start - 1)
        else:
            aria_phases.key_indices["end_idx"] = n - 1
    
class AriaPhasesOps:
    @staticmethod
    def print_phase_summary(aria_phases):
        phases = np.array([p.phase for p in aria_phases.phases])
        n = len(phases)
        
        print("\n" + "╔" + "═" * 60 + "╗")
        print(f"║{'ARIA TASK 3-PHASE MOE SEGMENTATION REPORT':^60}║")
        print("╠" + "═" * 60 + "╣")
        for p_id, p_name in PHASE_NAMES.items():
            count = np.sum(phases == p_id)
            print(f"║  - {p_name:<15} : {count:<10} frames ({count/n*100:>5.1f}%) {'':<11} ║")
        
        switches = np.sum(np.diff(phases) != 0)
        print(f"║  - Total Transitions  : {switches:<39} times  ║")
        print("╚" + "═" * 60 + "╝\n")
    
    @staticmethod
    def save_advanced_phase_plot(aria_phases, save_path):
        """
        绘制全能型分析图：涵盖动力学、交互信号、原始标签及逻辑说明。
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 14), sharex=True, 
                                            gridspec_kw={'height_ratios': [2.5, 2.5, 2]})
        
        n = len(aria_phases.phases)
        time = np.arange(n)
        v_robot = [p.v_robot for p in aria_phases.phases]
        v_hand = [p.v_hand for p in aria_phases.phases]
        grasp_conf = [p.grasp_conf for p in aria_phases.phases]
        grasp_raw = [p.grasp_state for p in aria_phases.phases]
        phases = [p.phase for p in aria_phases.phases]

        # 颜色与图例定义
        MPL_COLORS = {
            PHASE_NAV: '#27AE60', 
            PHASE_GRASP: '#F1C40F', 
            PHASE_MANIP: '#E67E22',
            PHASE_IDLE: '#95A5A6'
        }

        # --- Subplot 1: Robot Motion ---
        ax1.plot(time, v_robot, color='#2C3E50', linewidth=1.5, label='V_robot (m/s)')
        ax1.fill_between(time, v_robot, color='#2C3E50', alpha=0.1)
        ax1.axhline(y=V_ROBOT_THRESHOLD, color='r', linestyle='--', alpha=0.6)
        ax1.set_title("Robot Kinematics & Static Threshold", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Velocity (m/s)")
        ax1.legend(loc='upper right')

        # --- Subplot 2: Hand Reach Motion ---
        ax2.plot(time, v_hand, color='#2980B9', linewidth=2, label='V_hand_world (m/s)')
        ax2.axhline(y=V_HAND_THRESHOLD, color='r', linestyle='--', alpha=0.6)
        ax2.set_title("Hand Reaching Momentum (World Frame)", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Velocity (m/s)")
        ax2.legend(loc='upper right')

        # --- Subplot 3: Grasping Signals & Final Gating ---
        # 背景填充阶段条
        for i in range(n-1):
            ax3.axvspan(i, i+1, color=MPL_COLORS[phases[i]], alpha=0.4)
        
        # 绘制抓取置信度 (平滑曲线)
        ax3.plot(time, grasp_conf, color='#8E44AD', linewidth=2, label='Grasp Confidence (Smoothed)')
        # 绘制原始抓取脉冲 (Scatter 点)
        grasp_raw_indices = [i for i, val in enumerate(grasp_raw) if val == 1]
        ax3.scatter(grasp_raw_indices, [1.05]*len(grasp_raw_indices), color='#C0392B', s=10, 
                    marker='|', label='Raw Grasp Signal (1)', alpha=0.6)
        
        ax3.set_ylim(-0.1, 1.2)
        ax3.set_title("Grasp Interaction & MoE Phase Gating", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Frame Index")
        
        # --- 增加逻辑说明框 (The Aesthetic Logic Box) ---
        logic_text = (
            "Gating Logic Definition:\n"
            "--------------------------\n"
            f"1. NAV: V_robot > {V_ROBOT_THRESHOLD}m/s\n"
            "2. GRASP: V_robot < Static & Hand Visible\n"
            f"3. MANIP: Grasp Conf > {GRASP_CONF_THRESHOLD} & V_robot < Static\n"
            "4. IDLE: V_robot < Static & No Hand Action\n"
            "--------------------------\n"
            "Data Usage for MoE:\n"
            "Expert 1-3: Use Nav, Grasp, Manip\n"
            "Discard: IDLE frames (Grey Regions)"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#BDC3C7')
        ax1.text(0.01, 0.95, logic_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props, family='monospace', zorder=10)

        # 图例
        legend_elements = [
            Line2D([0], [0], color=MPL_COLORS[PHASE_NAV], lw=8, label='NAVIGATION'),
            Line2D([0], [0], color=MPL_COLORS[PHASE_GRASP], lw=8, label='GRASP/REACH'),
            Line2D([0], [0], color=MPL_COLORS[PHASE_MANIP], lw=8, label='MANIPULATION'),
            Line2D([0], [0], color=MPL_COLORS[PHASE_IDLE], lw=8, label='IDLE (Discard)')
        ]
        ax3.legend(handles=legend_elements + [ax3.get_lines()[0], ax3.collections[0]], 
                   loc='lower right', frameon=True, fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Professional 3-Phase analysis plot saved to: {save_path}")

    @staticmethod
    def draw_aria_phases_panel(img, idx, phase):
        # 阶段配置与三阶段颜色适配
        p_name = PHASE_NAMES.get(phase, 'UNKNOWN')
        p_color = PHASE_COLORS_BGR.get(phase, (255, 255, 255))
        
        display_text = f"MODE: {p_name}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness = 1
        
        (t_w, t_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
        w_vis = img.shape[1]
        center_x = w_vis // 2
        
        rect_x1, rect_y1 = center_x - (t_w // 2) - 30, 10
        rect_x2, rect_y2 = center_x + (t_w // 2) + 30, 45
        
        img = draw_glass_rect(img, (rect_x1, rect_y1), (rect_x2, rect_y2), alpha=0.6)
        cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), p_color, 1, cv2.LINE_AA)

        # 指示灯逻辑：在 GRASP 和 MANIP 阶段增加不同频率的闪烁感
        dot_x = rect_x1 + 15
        dot_y = (rect_y1 + rect_y2) // 2
        
        if phase == PHASE_NAV:
            cv2.circle(img, (dot_x, dot_y), 4, p_color, -1, cv2.LINE_AA)
        else:
            # 操作或抓取阶段：闪烁增强提醒
            blink_rate = 5 if phase == PHASE_MANIP else 10
            blink = (idx // blink_rate) % 2 == 0
            dot_color = p_color if blink else (50, 50, 50)
            cv2.circle(img, (dot_x, dot_y), 4, dot_color, -1, cv2.LINE_AA)
            cv2.circle(img, (dot_x, dot_y), 6, p_color, 1, cv2.LINE_AA)

        cv2.putText(img, display_text, (dot_x + 15, 32), font, font_scale, p_color, thickness, cv2.LINE_AA)
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

    # get aria_nav_human
    aria_nav_human_generator = AriaNavHumanGenerator(mps_path, mps_provider, tss)
    aria_nav_human = aria_nav_human_generator.get_aria_nav_human(run_aug=False)

    # get aria_nav_robot
    aria_nav_robot_generator = AriaNavRobotGenerator(mps_path, aria_nav_human)
    aria_nav_robot = aria_nav_robot_generator.get_aria_nav_robot(run_aug=False)

    # get aria_phases
    aria_phases_generator = AriaPhasesGenerator(mps_path, aria_nav_robot, aria_hands)
    aria_phases = aria_phases_generator.get_aria_phases()

    # save data
    aria_hands.save_aria_hands_json()
    aria_nav_human.save_aria_nav_human_json()
    aria_nav_robot.save_aria_nav_robot_json()
    aria_phases.save_aria_phases_json()

    # visualize
    
    def visualize_all():
        frames_all = []
        for idx, ts in enumerate(tqdm(tss, desc="Visualization")):
            img = aria_cam.cam[idx].rgb.copy()

            ########## orig data #########
            # draw hands skeleton and wrist pose
            img = AriaHandsOps.draw_aria_hands_skeleton_and_wrist_pose(img, aria_hands.hands[idx], aria_cam.cam[idx].k, aria_cam.cam[idx].d)
            # draw nav huamn traj
            img = AriaNavHumanOps.draw_aria_nav_human_traj(img, idx, aria_nav_human, aria_cam.cam[idx].k, aria_cam.cam[idx].c2w)

            ########### Panels ###########
            # phases panel
            img = AriaPhasesOps.draw_aria_phases_panel(img, idx, aria_phases.phases[idx].phase)
            # hands panel
            img = AriaHandsOps.draw_aria_hands_panel(img, idx, aria_hands.hands[idx])
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
# python -m aria.AriaPhases --mps_path "./data/mps_open_cabinet_5_vrs/"