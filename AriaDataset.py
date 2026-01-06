# -*- coding: utf-8 -*-
# @FileName: AriaDataset.py

import os
import argparse
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from torch.utils.data import Dataset

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeDomain

from utils.utils_media import create_video_from_frames
from aria.AriaCam import AriaCam, AriaCamData, AriaCamGenerator
from aria.AriaStereo import AriaStereo, AriaStereoData, AriaStereoGenerator
from aria.AriaHands import AriaJointAngle, AriaHandData, AriaHandsData,AriaHands, AriaHandsGenerator, AriaHandsOps                      
from aria.AriaNavHuman import AriaNavHumanData, AriaNavHuman, AriaNavHumanGenerator, AriaNavHumanOps
from aria.AriaNavRobot import AriaNavRobotData, AriaNavRobot, AriaNavRobotGenerator, AriaNavHuman2Robot, AriaNavHuman2RobotEvaluator, AriaNavRobotOps
from aria.AriaPointCloud import AriaPointCloud, AriaPointCloudGenerator
from aria.AriaPhases import AriaPhasesData, AriaPhases, AriaPhasesGenerator, AriaPhasesOps

@dataclass
class AriaDataset(Dataset):
    mps_path: str = None
    vrs_path: str = None
    hand_tracking_results_path: str = None

    tss: List[int] = field(default_factory=list)
    aria_cam: Optional[AriaCam] = None
    aria_stereo: Optional[AriaStereo] = None
    aria_hands: Optional[AriaHands] = None
    aria_nav_human: Optional[AriaNavHuman] = None
    aria_nav_robot: Optional[AriaNavRobot] = None
    aria_phases: Optional[AriaPhases] = None

    run_aug: bool = False

    def __init__(self, mps_path: str, run_aug: bool=False):
        self.mps_path = mps_path
        self.run_aug = run_aug
        self.vrs_path = os.path.join(self.mps_path, "sample.vrs")
        self.hand_tracking_results_path = os.path.join(self.mps_path, "hand_tracking", "hand_tracking_results.csv")

        self.vrs_provider, self.mps_provider = self._init_providers()
        
        self.rgb_timestamps_ns = self.vrs_provider.get_timestamps_ns(self.vrs_provider.get_stream_id_from_label("camera-rgb"), TimeDomain.DEVICE_TIME)
        self.aria_hands_mps = mps.hand_tracking.read_hand_tracking_results(self.hand_tracking_results_path)
        self.total_rgb_frames = len(self.rgb_timestamps_ns)
        self.mps_not_detected_number = self.total_rgb_frames - len(self.aria_hands_mps)
        self.start_idx = self.mps_not_detected_number
        self.num_frames = len(self.aria_hands_mps)
        self.end_idx = self.start_idx + self.num_frames - 1

        print(f'self.total_rgb_frames: {self.total_rgb_frames}')
        print(f'len(self.aria_hands_mps: {len(self.aria_hands_mps)}')
        print(f'self.mps_not_detected_number: {self.mps_not_detected_number}')
        print(f'self.start_idx: {self.start_idx}')
        print(f'self.end_idx: {self.end_idx}')

    def __len__(self) -> int:
        return self.num_frames
    
    def _init_providers(self) -> Tuple[Optional[MpsDataProvider], VrsDataProvider]:
        vrs_provider = data_provider.create_vrs_data_provider(self.vrs_path)
        try:
            mps_provider = MpsDataProvider(MpsDataPathsProvider(self.mps_path).get_data_paths())
        except Exception as e:
            print(f"Warning: No MPS Data Provider ({e})")
            mps_provider = None
        return vrs_provider, mps_provider

    def _process_aria_cam(self):
        aria_cam_generator = AriaCamGenerator(self.mps_path, self.vrs_provider, self.mps_provider)
        aria_cam = aria_cam_generator.get_aria_cam(self.start_idx, self.end_idx)
        tss = aria_cam.tss
        return tss, aria_cam
    
    def _process_aria_stereo(self):
        aria_stereo_generator = AriaStereoGenerator(self.mps_path, self.vrs_provider)
        aria_stereo = aria_stereo_generator.get_aria_stereo(self.tss)
        return aria_stereo
    
    def _process_aria_hand(self):
        aria_hands_generator = AriaHandsGenerator(self.mps_path, self.aria_hands_mps, self.aria_cam)
        aria_hands = aria_hands_generator.get_aria_hands()
        return aria_hands

    def _process_aria_nav_human(self):
        aria_nav_human_generator = AriaNavHumanGenerator(self.mps_path, self.mps_provider, self.tss)
        aria_nav_human = aria_nav_human_generator.get_aria_nav_human(self.run_aug)
        return aria_nav_human
    
    def _process_aria_nav_robot(self):
        aria_nav_robot_generator = AriaNavRobotGenerator(self.mps_path, self.aria_nav_human)
        aria_nav_robot = aria_nav_robot_generator.get_aria_nav_robot(self.run_aug)
        return aria_nav_robot

    def _process_aria_point_cloud(self):
        aria_point_cloud_generator = AriaPointCloudGenerator(self.mps_path, self.mps_provider, self.aria_nav_human)
        aria_point_cloud = aria_point_cloud_generator.get_aria_point_cloud()
        return aria_point_cloud
    
    def _process_aria_phases(self):
        aria_phases_generator = AriaPhasesGenerator(self.mps_path, self.aria_nav_robot, self.aria_hands)
        aria_phases = aria_phases_generator.get_aria_phases()
        return aria_phases
    
    def process_and_save_all(self):
        os.makedirs(os.path.join(self.mps_path, "aria"), exist_ok=True)
        self.tss, self.aria_cam = self._process_aria_cam()
        self.aria_cam.save_aria_cam_json()
        self.aria_cam.save_aria_cam_video_orig()

        self.aria_stereo = self._process_aria_stereo()
        self.aria_stereo.save_aria_stereo_json()
        self.aria_stereo.save_aria_stereo_video_orig()

        self.aria_hands = self._process_aria_hand()
        self.aria_hands.save_aria_hands_json()

        self.aria_nav_human = self._process_aria_nav_human()
        self.aria_nav_human.save_aria_nav_human_json()

        self.aria_nav_robot = self._process_aria_nav_robot()
        self.aria_nav_robot.save_aria_nav_robot_json()

        self.aria_point_cloud = self._process_aria_point_cloud()
        self.aria_point_cloud.save_all_pc()

        self.aria_phases = self._process_aria_phases()
        self.aria_phases.save_aria_phases_json()

    def visualize_all(self):
        frames_all = []
        for idx, ts in enumerate(tqdm(self.tss, desc="Visualization")):
            img = self.aria_cam.cam[idx].rgb.copy()

            ########## orig data #########
            # draw hands skeleton and wrist pose
            img = AriaHandsOps.draw_aria_hands_skeleton_and_wrist_pose(img, self.aria_hands.hands[idx], self.aria_cam.cam[idx].k, self.aria_cam.cam[idx].d)
            # draw nav huamn traj
            img = AriaNavHumanOps.draw_aria_nav_human_traj(img, idx, self.aria_nav_human, self.aria_cam.cam[idx].k, self.aria_cam.cam[idx].c2w)

            ########### Panels ###########
            # phases panel
            img = AriaPhasesOps.draw_aria_phases_panel(img, idx, self.aria_phases.phases[idx].phase)
            # hands panel
            img = AriaHandsOps.draw_aria_hands_panel(img, idx, self.aria_hands.hands[idx])
            # nav robot panel
            img = AriaNavRobotOps.draw_aria_nav_robot_panel(img, self.aria_nav_robot.traj[idx])
            
            frames_all.append(img)
        
        save_path = os.path.join(self.mps_path, "aria", "aria_video_vis.mp4")
        create_video_from_frames(
            frames=frames_all, 
            save_path=save_path,
            fps=self.aria_cam.fps)
        
    def run(self):
        self.process_and_save_all()
        self.visualize_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    parser.add_argument("--run_aug", action='store_true', default=False, help="If you want to do nav data augmentation")
    args = parser.parse_args()

    aria_dataset = AriaDataset(args.mps_path, args.run_aug)
    aria_dataset.run()

# conda activate aria
# cd src
# python -m aria.AriaDataset --mps_path "./data/open_cabinet_0/mps_open_cabinet_0_005_vrs/" 