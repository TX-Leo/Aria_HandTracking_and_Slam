# -*- coding: utf-8 -*-
# @FileName: AriaPointCloud.py

import os
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import Optional
import argparse

from projectaria_tools.core.mps.utils import filter_points_from_confidence
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeDomain

from aria.AriaCam import AriaCamGenerator
from aria.AriaNavHuman import AriaNavHuman, AriaNavHumanGenerator

@dataclass
class AriaPointCloud:
    pc_scene: Optional[o3d.geometry.PointCloud] = None
    pc_human_traj: Optional[o3d.geometry.PointCloud] = None
    pc_scene_with_human_traj: Optional[o3d.geometry.PointCloud] = None
    mps_path: str = None
    
    def _save_pc(self, pc: Optional[o3d.geometry.PointCloud], filename: str):
        if pc is not None:
            save_path = os.path.join(self.mps_path, "aria", filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            o3d.io.write_point_cloud(save_path, pc)
        
    def save_all_pc(self):
        self._save_pc(self.pc_scene, "aria_pc_scene.ply")
        self._save_pc(self.pc_human_traj, "aria_pc_human_traj.ply")
        self._save_pc(self.pc_scene_with_human_traj, "aria_pc_scene_with_human_traj.ply")

class AriaPointCloudGenerator():
    def __init__(self, mps_path: str, mps_provider: MpsDataProvider , aria_nav_human: AriaNavHuman, min_confidence: float = 0.001, max_uncertainty: float = 0.15):
        self.mps_path = mps_path
        self.mps_provider = mps_provider
        self.aria_nav_human = aria_nav_human
        self.min_confidence = min_confidence
        self.max_uncertainty = max_uncertainty
    
    def _create_pcd_obj(self, xyz: np.ndarray, color=[0.0, 1.0, 0.0]) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        if xyz.size == 0:
            return pcd
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.paint_uniform_color(color)
        return pcd
    
    def get_aria_point_cloud(self) -> Optional[AriaPointCloud]:
        print("Processing Aria Point Cloud")

        raw_points = self.mps_provider.get_semidense_point_cloud()
        filtered_points = filter_points_from_confidence(raw_points, self.min_confidence, self.max_uncertainty)
        if not filtered_points:
            return AriaPointCloud(np.empty((0, 3)), 0)
        pc_scene = np.stack([it.position_world for it in filtered_points])
        if self.aria_nav_human.traj is not None:
            human_traj_xyz = [aria_nav_human_data.t_world for aria_nav_human_data in self.aria_nav_human.traj]
            pc_human_traj = self._create_pcd_obj(human_traj_xyz)
        else:
            pc_human_traj = None
        pc_scene_with_huamn_traj = pc_scene + pc_human_traj
        aria_point_cloud = AriaPointCloud(pc_scene, pc_human_traj, pc_scene_with_huamn_traj)
        aria_point_cloud.mps_path = self.mps_path
        return aria_point_cloud
    
    def get_aria_point_cloud(self) -> Optional[AriaPointCloud]:
        raw_points = self.mps_provider.get_semidense_point_cloud()
        filtered_points = filter_points_from_confidence(raw_points, self.min_confidence, self.max_uncertainty)
        if not filtered_points:
            return None

        scene_xyz = np.stack([it.position_world for it in filtered_points])
        pcd_scene = self._create_pcd_obj(scene_xyz, color=[0.6, 0.6, 0.6]) # grey

        pcd_human_traj = None
        if self.aria_nav_human and self.aria_nav_human.traj:
            traj_xyz = np.array([data.t_world for data in self.aria_nav_human.traj])
            pcd_human_traj = self._create_pcd_obj(traj_xyz, color=[0.0, 1.0, 0.0]) # green

        pcd_combined = None
        if pcd_scene is not None and pcd_human_traj is not None:
            pcd_combined = pcd_scene + pcd_human_traj
        elif pcd_scene is not None:
            pcd_combined = pcd_scene

        return AriaPointCloud(
            pc_scene=pcd_scene, 
            pc_human_traj=pcd_human_traj, 
            pc_scene_with_human_traj=pcd_combined,
            mps_path=self.mps_path
        )

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
    aria_nav_human = aria_nav_human_generator.get_aria_nav_human()

    # get aria_point_cloud
    aria_point_cloud_generator = AriaPointCloudGenerator(mps_path, mps_provider, aria_nav_human)
    aria_point_cloud = aria_point_cloud_generator.get_aria_point_cloud()

    # save data
    aria_point_cloud.save_all_pc()

# conda activate aria
# cd src
# python -m aria.AriaPointCloud --mps_path "./data/mps_open_cabinet_5_vrs/"