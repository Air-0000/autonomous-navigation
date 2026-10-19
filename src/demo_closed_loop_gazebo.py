"""Closed-Loop 3D Gazebo Simulation with Biological CANN Navigation.

Connects:
- Gazebo Sim 3D Robot (/camera/image_raw, /imu/data, /odom, /cmd_vel)
- Bio-Inspired Cognitive Map:
    * Head Direction (HD) Ring Attractor (continuous angular velocity integration)
    * Continuous Attractor Neural Network (CANN) 2D Grid Cells
    * Multi-scale Place Cell Associative Memory & Landmark Drift Reset
    * Vector-based Motor Command Decoding
"""

import argparse
import math
import os
from pathlib import Path
import sys
import time
from typing import List, Tuple
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cognition.cognitive_map import CognitiveMap
from src.interface.gazebo_bridge import GazeboBridge
from src.perception.landmark_detector import LandmarkDetector


class GazeboClosedLoopNavigator:
    """Manages closed-loop biological navigation inside the Gazebo 3D simulation."""

    # Known visual landmarks in navigation_world.sdf
    KNOWN_LANDMARKS = {
        "landmark_red": (3.0, 3.0),
        "landmark_blue": (-3.0, 3.0),
        "landmark_green": (-3.0, -3.0),
        "landmark_yellow": (3.0, -3.0)
    }

    def __init__(self,
                 target_waypoints: List[Tuple[float, float]] = None,
                 control_rate_hz: float = 20.0):
        self.control_rate = control_rate_hz
        self.dt = 1.0 / control_rate_hz

        self.bridge = GazeboBridge()
        self.detector = LandmarkDetector()
        self.cognitive_map = CognitiveMap()

        # Seed Place Cell associative memory with known landmark locations
        for lm_id, (lx, ly) in self.KNOWN_LANDMARKS.items():
            self.cognitive_map.place_net.associate_landmark(lm_id, (lx, ly))

        self.waypoints = target_waypoints or [
            (2.2, 2.2),    # Near Red Landmark
            (-2.2, 2.2),   # Near Blue Landmark
            (-2.2, -2.2),  # Near Green Landmark
            (2.2, -2.2),   # Near Yellow Landmark
            (0.0, 0.0)     # Return to Center
        ]
        self.current_waypoint_idx = 0

        self.step_count = 0
        self.last_time = time.time()

    def step(self) -> dict:
        """Executes a single closed-loop control step."""
        now = time.time()
        actual_dt = min(0.1, max(0.005, now - self.last_time))
        self.last_time = now
        self.step_count += 1

        # 1. Read Sensors from Gazebo
        imu_data = self.bridge.get_imu_data()
        odom_data = self.bridge.get_odometry()
        frame = self.bridge.get_camera_frame()

        # 2. Extract self-motion cues
        gyro_z = imu_data.get("gyro_z", 0.0)
        linear_vel = odom_data.get("linear_vel", 0.0)

        # 3. Path Integration via HD Ring Attractor & 2D Grid Cells
        state = self.cognitive_map.update_motion(
            linear_vel=(linear_vel, 0.0),
            angular_vel=gyro_z,
            dt=actual_dt
        )
        est_heading = state.heading
        est_x, est_y = state.position

        # 4. Visual Landmark Perception & Associative Drift Correction
        landmark_corrections = 0
        if frame is not None:
            detections = self.detector.detect(frame)
            for det in detections:
                lm_id = det["id"]
                if lm_id in self.KNOWN_LANDMARKS:
                    bearing = det["bearing"]
                    dist = det["distance"]
                    self.cognitive_map.correct_with_visual_landmark(
                        landmark_name=lm_id,
                        relative_dist=dist,
                        relative_bearing=bearing
                    )
                    landmark_corrections += 1

        # 5. Cognitive Planning & Target Navigation
        est_x, est_y = self.cognitive_map.current_x, self.cognitive_map.current_y
        target_x, target_y = self.waypoints[self.current_waypoint_idx]

        dist_to_target = math.hypot(target_x - est_x, target_y - est_y)
        if dist_to_target < 0.5:
            # Advance waypoint
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            target_x, target_y = self.waypoints[self.current_waypoint_idx]

        # Obstacle avoidance vectors (world obstacles)
        obstacles = [(0.0, 2.0, 0.8), (0.0, -2.0, 0.8)]
        v_cmd, w_cmd = self.cognitive_map.compute_navigation_command(
            goal_pos=(target_x, target_y),
            obstacles=obstacles
        )

        # 6. Actuation command to Gazebo
        self.bridge.send_cmd_vel(v_cmd, w_cmd)

        gt_x = odom_data.get("x", 0.0)
        gt_y = odom_data.get("y", 0.0)
        gt_yaw = odom_data.get("yaw", 0.0)
        pos_error = math.hypot(gt_x - est_x, gt_y - est_y)
        heading_error = abs(math.atan2(math.sin(gt_yaw - est_heading), math.cos(gt_yaw - est_heading)))

        metrics = {
            "step": self.step_count,
            "est_pos": (round(est_x, 3), round(est_y, 3)),
            "gt_pos": (round(gt_x, 3), round(gt_y, 3)),
            "pos_error_m": round(pos_error, 3),
            "est_heading_deg": round(math.degrees(est_heading), 2),
            "gt_heading_deg": round(math.degrees(gt_yaw), 2),
            "heading_error_deg": round(math.degrees(heading_error), 2),
            "cmd_vel": (round(v_cmd, 2), round(w_cmd, 2)),
            "target_waypoint": (target_x, target_y),
            "waypoint_idx": self.current_waypoint_idx,
            "landmarks_seen": landmark_corrections,
            "accumulated_drift": round(self.cognitive_map.accumulated_drift, 3)
        }
        return metrics

    def run(self, max_steps: int = 100):
        """Runs the navigation loop for a specified number of steps."""
        print(f"Starting Gazebo Biological Closed-Loop Navigation ({max_steps} steps)...")
        print(f"Connected to Gazebo: {self.bridge.is_connected}")
        print("Target Waypoints:", self.waypoints)
        print("-" * 80)

        try:
            for i in range(max_steps):
                start_step = time.time()
                metrics = self.step()

                if (i + 1) % 10 == 0 or i == 0:
                    print(f"Step {metrics['step']:3d} | "
                          f"Est: ({metrics['est_pos'][0]:5.2f}, {metrics['est_pos'][1]:5.2f}) | "
                          f"GT: ({metrics['gt_pos'][0]:5.2f}, {metrics['gt_pos'][1]:5.2f}) | "
                          f"Pos Err: {metrics['pos_error_m']:4.2f}m | "
                          f"Head Err: {metrics['heading_error_deg']:4.1f}° | "
                          f"Cmd: v={metrics['cmd_vel'][0]:.2f}, w={metrics['cmd_vel'][1]:.2f} | "
                          f"WP: #{metrics['waypoint_idx']} {metrics['target_waypoint']} | "
                          f"Landmarks: {metrics['landmarks_seen']}")

                elapsed = time.time() - start_step
                sleep_time = max(0.0, self.dt - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nNavigation interrupted by user.")
        finally:
            self.stop()

    def stop(self):
        """Brake robot and release resources."""
        print("Stopping robot and shutting down bridge...")
        self.bridge.send_cmd_vel(0.0, 0.0)
        self.bridge.close()


def main():
    parser = argparse.ArgumentParser(description="Gazebo 3D Biological CANN Navigation")
    parser.add_argument("--steps", type=int, default=100, help="Number of control steps to run")
    parser.add_argument("--rate", type=float, default=20.0, help="Control loop rate (Hz)")
    args = parser.parse_args()

    navigator = GazeboClosedLoopNavigator(control_rate_hz=args.rate)
    navigator.run(max_steps=args.steps)


if __name__ == "__main__":
    main()
