#!/usr/bin/env python3
"""
3D Visualization Demo - UAV Navigation Trajectory & Perception
Using matplotlib for real-time 3D rendering
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import sys
import os

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interface.simulator import WebotsSimulator
from src.perception.detector import YOLODetector
from src.cognition.cognitive_map import CognitiveMap

class NavVisualizer3D:
    """3D Navigation Visualizer"""

    def __init__(self):
        # Initialize modules
        print("Initializing perception modules...")
        self.detector = YOLODetector()
        self.simulator = WebotsSimulator()
        self.cognitive_map = CognitiveMap()

        # Connect simulator
        self.simulator.connect()

        # Trajectory recording
        self.trajectory = []
        self.detections_history = []

        # Target position
        self.goal = (25.0, 25.0)

        # Setup matplotlib 3D
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 8))

        # Create subplot layout
        self.ax_main = self.fig.add_subplot(121, projection='3d')
        self.ax_top = self.fig.add_subplot(122)

        # Set title
        self.fig.suptitle('UAV Intelligent Navigation - 3D Trajectory Visualization', fontsize=14, color='cyan')

        # Initialize
        self.init_plots()

    def init_plots(self):
        """Initialize 3D plots"""
        # Main 3D plot
        self.ax_main.set_xlabel('X (m)', color='cyan')
        self.ax_main.set_ylabel('Z (m)', color='cyan')
        self.ax_main.set_zlabel('Y (m)', color='cyan')
        self.ax_main.set_title('3D Trajectory View', color='yellow', fontsize=12)

        # Top-down 2D map
        self.ax_top.set_xlabel('X (m)', color='cyan')
        self.ax_top.set_ylabel('Z (m)', color='cyan')
        self.ax_top.set_title('Top-down Trajectory Map', color='yellow', fontsize=12)
        self.ax_top.set_aspect('equal')

    def update(self, frame):
        """Update animation frame"""
        # Execute simulation step
        timestep = self.simulator.step()

        # Get state
        pose = self.simulator.get_pose()
        x, y, z = pose[0], pose[1], pose[2]

        # Record trajectory
        self.trajectory.append([x, y, z])

        # Object detection
        image = self.simulator.get_camera_image()
        detections = self.detector.detect(image)

        # Update cognitive map
        current_pos = (x, z)
        velocity = self.simulator.get_velocity()
        self.cognitive_map.update(current_pos, velocity)

        # Move toward goal
        dx = self.goal[0] - x
        dz = self.goal[1] - z
        dist = np.sqrt(dx**2 + dz**2)

        if dist > 0.5:
            speed = 0.8
            vx = speed * dx / dist
            vz = speed * dz / dist
            self.simulator.set_velocity(vx, 0, vz)

        # Clear and redraw
        self.ax_main.clear()
        self.ax_top.clear()

        # === 3D View ===
        # Draw ground grid
        grid_size = 30
        grid_step = 5
        for gx in np.arange(-grid_size, grid_size + 1, grid_step):
            for gz in np.arange(-grid_size, grid_size + 1, grid_step):
                verts = [
                    [gx, 0, gz],
                    [gx + grid_step, 0, gz],
                    [gx + grid_step, 0, gz + grid_step],
                    [gx, 0, gz + grid_step]
                ]
                verts_z = [[v[0], v[2], v[1]] for v in verts]  # Convert coords
                quad = Poly3DCollection([verts_z], alpha=0.1, facecolor='cyan')
                self.ax_main.add_collection3d(quad)

        # Draw trajectory and start/end points
        n_obstacles = len(detections)
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax_main.plot(traj[:, 0], traj[:, 2], traj[:, 1], 'b-', linewidth=2, alpha=0.8, label='Trajectory')
            self.ax_main.scatter(traj[0, 0], traj[0, 2], traj[0, 1], c='green', s=100, marker='^', label='Start')
            self.ax_main.scatter(traj[-1, 0], traj[-1, 2], traj[-1, 1], c='red', s=100, marker='v', label='Current')
        else:
            traj = np.array([[x, y, z]])

        # Draw goal point
        self.ax_main.scatter(self.goal[0], self.goal[1], 0.5, c='yellow', s=200, marker='*', label='Goal')

        # Draw UAV position
        self.ax_main.scatter(x, z, y, c='cyan', s=150, marker='o')

        # Draw detected obstacles (simplified)
        for i in range(min(n_obstacles, 5)):
            obs_x = x + np.random.uniform(-5, 5)
            obs_z = z + np.random.uniform(-5, 5)
            obs_y = np.random.uniform(0, 2)
            self.ax_main.scatter(obs_x, obs_z, obs_y, c='red', s=50, alpha=0.6)

        # Set 3D axis limits
        self.ax_main.set_xlim(-5, max(35, x + 10))
        self.ax_main.set_ylim(-5, max(35, z + 10))
        self.ax_main.set_zlim(0, max(5, y + 2))

        # === Top-down View ===
        self.ax_top.set_xlim(-5, max(35, x + 10))
        self.ax_top.set_ylim(-5, max(35, z + 10))

        # Draw grid
        self.ax_top.grid(True, alpha=0.3)

        # Draw trajectory
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax_top.plot(traj[:, 0], traj[:, 2], 'b-', linewidth=2, alpha=0.8, label='Trajectory')
            self.ax_top.scatter(traj[0, 0], traj[0, 2], c='green', s=100, marker='^', label='Start')
            self.ax_top.scatter(traj[-1, 0], traj[-1, 2], c='red', s=100, marker='o', label='Current')

        # Draw goal
        self.ax_top.scatter(self.goal[0], self.goal[1], c='yellow', s=200, marker='*', label='Goal')

        # Draw obstacles
        for i in range(min(n_obstacles, 8)):
            obs_x = x + np.random.uniform(-8, 8)
            obs_z = z + np.random.uniform(-8, 8)
            self.ax_top.scatter(obs_x, obs_z, c='red', s=80, alpha=0.5)

        # Draw field of view range
        circle = plt.Circle((x, z), 5, fill=False, color='cyan', linestyle='--', alpha=0.5)
        self.ax_top.add_patch(circle)

        # Reset top view labels
        self.ax_top.set_xlabel('X (m)')
        self.ax_top.set_ylabel('Z (m)')
        self.ax_top.legend(loc='upper left', fontsize=8)

        # Add info text
        info_text = f'Step: {frame}\nPos: ({x:.1f}, {y:.1f}, {z:.1f})\nDetect: {n_obstacles}'
        self.ax_main.text2D(0.02, 0.95, info_text, transform=self.ax_main.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        # Set titles
        self.ax_main.set_title(f'3D Trajectory View (Step: {frame})', color='yellow')
        self.ax_top.set_title('Top-down Map', color='yellow')

        plt.tight_layout()

        # Check arrival every 50 steps
        if frame > 50 and dist < 1.0:
            print(f"Goal reached! Step: {frame}, Pos: ({x:.1f}, {z:.1f})")
            return

        # Max 1000 steps
        if frame >= 1000:
            print("Max steps reached")
            self.fig.close()

        return

    def run(self, max_frames=500):
        """Run animation"""
        print("="*50)
        print("UAV 3D Navigation Trajectory Visualization")
        print("Press Ctrl+C to exit")
        print("="*50)

        try:
            anim = animation.FuncAnimation(
                self.fig, self.update, frames=max_frames,
                interval=50, blit=False
            )
            plt.show()
        except KeyboardInterrupt:
            print("\nUser interrupted")
        finally:
            self.close()

    def close(self):
        """Close"""
        self.simulator.close()
        print(f"Trajectory recorded: {len(self.trajectory)} points")
        plt.close()

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='3D Navigation Trajectory Visualization')
    parser.add_argument('--frames', type=int, default=300, help='Max frames')
    args = parser.parse_args()

    viz = NavVisualizer3D()
    viz.run(max_frames=args.frames)

if __name__ == "__main__":
    main()