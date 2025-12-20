# TODO：
# 1. 设置非常详尽的状态 approach, grasp, unlock, open, traverse
# 2. 自动计算（加上一个手指pinch的功能）

# -*- coding: utf-8 -*-
# @FileName: AriaPhaseSegmenter.py

import numpy as np
from scipy.signal import medfilt
from typing import List

# [EMMA Phase] 阶段判定阈值配置
PHASE_VEL_THRESH = 0.15 
# 推门修正窗口 (单位: 帧)
PHASE_PUSH_EXTEND_FRAMES = 30 

# =========================================================================
# [Logic Core] 自动阶段分割 (Auto Phase Segmentation)
# =========================================================================

class AriaPhaseSegmenter:
    """
    负责根据手部状态和机器人速度，将时间序列划分为不同阶段。
    Stages: 0:APPROACH, 1:UNLOCK, 2:OPEN, 3:TRAVERSE
    """
    @staticmethod
    def segment_phases(structs: List) -> List[int]:
        """
        输入 AriaStruct 列表，返回每帧对应的阶段 ID。
        """
        if not structs: return []
        n = len(structs)
        phases = np.zeros(n, dtype=int)
        
        # 提取抓取状态序列 (max of left/right hand grasp state)
        # 注意: 这里假设 structs[i].hands 是 AriaHand 对象列表
        raw_grasp = np.array([max([h.grasp_state for h in s.hands] + [0]) for s in structs])
        smooth_grasp = medfilt(raw_grasp, kernel_size=11).astype(int)
        
        # 提取速度序列
        velocities = np.array([abs(s.nav.robot_v) for s in structs])
        
        grasp_indices = np.where(smooth_grasp == 1)[0]
        if len(grasp_indices) == 0:
            return phases.tolist()
            
        t_start = grasp_indices[0]
        t_end = grasp_indices[-1]
        
        # 基础划分: 抓取前为 0，抓取结束为 3
        phases[:t_start] = 0
        phases[t_end:] = 3
        
        # 中间过程划分: 根据速度区分 1(Unlock/Grasp) 和 2(Open/Push)
        for i in range(t_start, t_end + 1):
            if velocities[i] > PHASE_VEL_THRESH:
                phases[i] = 2 
            else:
                phases[i] = 1 
        
        # 延展推门阶段 (利用惯性或后处理)
        extend_limit = min(t_end + PHASE_PUSH_EXTEND_FRAMES, n)
        for i in range(t_end + 1, extend_limit):
            if velocities[i] > PHASE_VEL_THRESH:
                phases[i] = 2
            else:
                break 
        
        # 平滑处理，消除噪声跳变
        phases = medfilt(phases, kernel_size=15).astype(int)
        return phases.tolist()