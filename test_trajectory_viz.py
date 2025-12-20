# -*- coding: utf-8 -*-
# @FileName: test_trajectory_viz.py
# @Description: 快速测试 Aria 轨迹可视化功能 (Rainbow Trail)

import os
import argparse
import numpy as np
from AriaDataset import AriaDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to MPS folder")
    args = parser.parse_args()
    
    mps_root = args.mps_path
    
    # 1. 查找 VRS
    vrs_file = os.path.join(mps_root, "sample.vrs")
    if not os.path.exists(vrs_file):
        files = [f for f in os.listdir(mps_root) if f.endswith('.vrs')]
        vrs_file = os.path.join(mps_root, files[0]) if files else ""
    
    # 2. 初始化 Dataset (只加载必要数据)
    print(">>> 初始化 Dataset...")
    dataset = AriaDataset(mps_root, 
                          vrs_file, 
                          os.path.join(mps_root, "hand_tracking/hand_tracking_results.csv"), 
                          os.path.join(mps_root, "test_viz_output")) # 输出到单独文件夹
    
    # 3. 强制加载 Structs (这是 create_visualization_videos 所需的)
    print(">>> 读取所有帧结构 (Skipping heavy computations)...")
    # 我们不调用 process_and_save_all，而是手动执行轻量级步骤
    
    # (A) 触发 __getitem__ 读取所有数据到内存 (为了获得 Pose)
    # 注意: 为了速度，这里可以只读 Pose，但在 AriaDataset 架构里，
    # __getitem__ 做了很多事。为了复用代码，我们还是遍历一遍。
    # 如果想更快，可以重写一个简化的 loader，但为了保证和主流程一致，这里还是全读。
    temp_structs = []
    for i in range(len(dataset)):
        if i % 10 == 0: print(f"Loading {i}/{len(dataset)}...", end='\r')
        temp_structs.append(dataset[i])
    dataset.all_aria_structs = temp_structs
    print("\nData loaded.")

    # (B) 确保有轨迹数据
    # AriaDataset 初始化时只定义了空的 trajectory_points_3d
    # 我们需要手动触发一下轨迹提取，或者如果有缓存直接读取
    print(">>> 提取导航轨迹...")
    if dataset.has_online_calibration:
        dataset._process_full_navigation_trajectory()
    else:
        print("警告: 没有 Online Calibration，轨迹将基于 Device Pose (可能不平滑)")
        # 简单的 Fallback: 使用每一帧的 cam c2w translation
        traj = []
        for s in dataset.all_aria_structs:
            # s.cam.c2w 是 device/cam pose
            traj.append(s.cam.c2w[:3, 3])
        dataset.trajectory_points_3d = np.array(traj)

    # 4. 运行可视化
    print(">>> 开始生成测试视频 (video_all.mp4)...")
    # 假设没有音频，传 None
    dataset._create_visualization_videos(audio_path='')
    
    print(f"完成！请检查: {os.path.join(dataset.save_path, 'video_all.mp4')}")

    # python test_trajectory_viz.py --mps_path "../data/mps_open_door_13_vrs/"