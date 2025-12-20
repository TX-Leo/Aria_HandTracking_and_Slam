# -*- coding: utf-8 -*-
# @FileName: AriaPhaseSegmenter_v2.py

import numpy as np
from scipy.signal import medfilt
from typing import List

# [Config] 阶段判定参数
# 线速度阈值: 高于此值认为是 Navigation，低于此值认为是 Manipulation
# 建议值: 0.1 m/s (10cm/s)，低于这个速度通常是在做精细操作或站立
VELOCITY_THRESHOLD = 0.1

# 状态滤波窗口: 用于消除瞬间停顿导致的噪声
# 31帧约等于 1秒 (30FPS)，意味着状态切换至少要持续0.5秒才会被确认
SMOOTH_KERNEL_SIZE = 31 

# 阶段定义
PHASE_NAV = 0  # Navigation
PHASE_MAN = 1  # Manipulation

class AriaPhaseSegmenter:
    """
    [Logic Core] 自动阶段分割器 (二分类版)
    根据机器人底盘速度和手部抓取状态，将视频分割为:
    0: Navigation (移动中)
    1: Manipulation (操作中/静止/抓取)
    """
    
    @staticmethod
    def segment_phases(structs: List) -> List[int]:
        """
        输入: AriaStruct 列表
        输出: 每帧对应的阶段 ID [0, 0, ..., 1, 1, ...]
        """
        if not structs: return []
        n = len(structs)
        
        # 1. 提取特征序列
        # 线速度 (绝对值)
        velocities = np.array([abs(s.nav.robot_v) for s in structs])
        
        # 抓取状态 (任意一只手抓住即视为抓住)
        # 假设 s.hands 是 AriaHand 对象列表
        grasp_states = np.array([max([h.grasp_state for h in s.hands] + [0]) for s in structs])
        
        # 2. 初始分类 (基于阈值)
        # 默认为 Navigation (0)
        raw_phases = np.zeros(n, dtype=int)
        
        # 如果速度低，或者是正在抓取，则认为是 Manipulation (1)
        # 逻辑: 低速 OR 抓取 = 操作
        is_manipulation = (velocities < VELOCITY_THRESHOLD) | (grasp_states == 1)
        raw_phases[is_manipulation] = PHASE_MAN
        
        # 3. 后处理 (平滑滤波)
        # 使用中值滤波消除毛刺 (Spikes)
        # 例如: 000010000 -> 000000000 (短暂的停顿不算操作)
        # 例如: 111101111 -> 111111111 (短暂的移动不算导航)
        filtered_phases = medfilt(raw_phases, kernel_size=SMOOTH_KERNEL_SIZE)
        
        # 4. (可选) 逻辑修正: 保证 Manipulation 是一段连续的区间
        # 很多任务是 Nav -> Man -> Nav 或者是 Nav -> Man
        # 如果你的任务很简单，上面的滤波就够了。
        # 如果想强制合并碎片，可以使用形态学膨胀腐蚀 (Dilation/Erosion)
        
        return filtered_phases.astype(int).tolist()