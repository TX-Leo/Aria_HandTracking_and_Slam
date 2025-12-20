# TODO:
# 1. 添加一个AriaPointCloud的class来存点云和一些参数

# -*- coding: utf-8 -*-
# @FileName: AriaPointCloud.py

import os
import numpy as np
import open3d as o3d

class AriaPointCloudOps:
    """
    [工具类] 提供 Aria 点云数据的处理、合并和保存功能。
    """
    
    @staticmethod
    def save_merged_point_cloud(point_cloud: np.ndarray, trajectory_points: np.ndarray, save_path: str):
        """
        保存环境点云和轨迹点云的合并结果 (.ply)。
        """
        if point_cloud is None or len(trajectory_points) == 0:
            return
            
        pcd_scene = o3d.geometry.PointCloud()
        pcd_scene.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_scene.paint_uniform_color([0.6, 0.6, 0.6]) # 灰色环境
        
        pcd_traj = o3d.geometry.PointCloud()
        pcd_traj.points = o3d.utility.Vector3dVector(trajectory_points.astype(np.float64))
        pcd_traj.paint_uniform_color([0.0, 1.0, 0.0]) # 绿色轨迹
        
        pcd_merged = pcd_scene + pcd_traj
        o3d.io.write_point_cloud(save_path, pcd_merged)
        print(f"融合点云已保存至: {save_path}")

    @staticmethod
    def save_point_cloud_ply(point_cloud: np.ndarray, save_path: str):
        """
        仅保存环境点云 (.ply)。
        """
        if point_cloud is None: return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.io.write_point_cloud(save_path, pcd)

    @staticmethod
    def visualize_scene_with_trajectory(point_cloud: np.ndarray, trajectory_points: np.ndarray, save_path: str = None):
        """
        弹出 Open3D 窗口可视化场景和轨迹。
        注意: 此函数会阻塞进程，直到窗口关闭。
        """
        if point_cloud is None: return
        geometries = []
        
        pcd_scene = o3d.geometry.PointCloud()
        pcd_scene.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_scene.paint_uniform_color([0.6, 0.6, 0.6]) 
        geometries.append(pcd_scene)
        
        if len(trajectory_points) > 0:
            pcd_traj = o3d.geometry.PointCloud()
            pcd_traj.points = o3d.utility.Vector3dVector(trajectory_points.astype(np.float64))
            pcd_traj.paint_uniform_color([0, 1, 0]) 
            geometries.append(pcd_traj)
            
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        geometries.append(axes)
        
        try:
            vis = o3d.visualization.Visualizer()
            # 设置可见以便交互
            vis.create_window(window_name="Aria EMMA Trajectory", width=1280, height=720, visible=True)
            
            for geom in geometries: vis.add_geometry(geom)
            
            opt = vis.get_render_option()
            if opt is not None:
                opt.point_size = 5.0
                opt.background_color = np.asarray([0.1, 0.1, 0.1])
            
            # 阻塞运行
            vis.run()
            
            # 如果需要保存截图
            if save_path:
                screenshot_path = os.path.join(save_path, "trajectory_preview.png")
                vis.capture_screen_image(screenshot_path)
                print(f"可视化截图已保存: {screenshot_path}")
                
            vis.destroy_window()
        except Exception as e:
            print(f"可视化跳过: {e}")