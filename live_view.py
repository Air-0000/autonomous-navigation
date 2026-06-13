#!/usr/bin/env python3
"""
UAV 3D Simulation Viewer
APF-based Obstacle Avoidance
"""

import sys
import os
import math

os.environ['SDL_VIDEODRIVER'] = 'cocoa'

import pygame
from pygame.locals import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.interface.simulator import WebotsSimulator


class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Camera3D:
    def __init__(self):
        self.x = 0
        self.y = 2
        self.z = 0
        self.yaw = 0
        self.pitch = -0.3

    def project(self, point):
        dx = point.x - self.x
        dy = point.y - self.y
        dz = point.z - self.z

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        rx = dx * cos_yaw - dz * sin_yaw
        rz = dx * sin_yaw + dz * cos_yaw
        ry = dy

        if rz <= 0.1:
            return None

        scale = 500 / rz
        screen_x = 400 + rx * scale
        screen_y = 300 - ry * scale

        return (screen_x, screen_y, rz)


class Renderer3D:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("UAV 3D View")
        self.width = 800
        self.height = 600
        self.camera = Camera3D()
        self.clock = pygame.time.Clock()

    def clear(self):
        self.screen.fill((100, 150, 200))

    def draw_point_3d(self, p, color, size=3):
        sp = self.camera.project(p)
        if sp:
            pygame.draw.circle(self.screen, color, (int(sp[0]), int(sp[1])), size)

    def draw_line_3d(self, p1, p2, color):
        sp1 = self.camera.project(p1)
        sp2 = self.camera.project(p2)
        if sp1 and sp2:
            pygame.draw.line(self.screen, color,
                          (int(sp1[0]), int(sp1[1])),
                          (int(sp2[0]), int(sp2[1])), 2)

    def draw_box_3d(self, center, size, color):
        cx, cy, cz = center
        hw, hh, hd = size[0]/2, size[1]/2, size[2]/2
        p = [
            Point3D(cx-hw, cy-hh, cz-hd),
            Point3D(cx+hw, cy-hh, cz-hd),
            Point3D(cx+hw, cy-hh, cz+hd),
            Point3D(cx-hw, cy-hh, cz+hd),
            Point3D(cx-hw, cy+hh, cz-hd),
            Point3D(cx+hw, cy+hh, cz-hd),
            Point3D(cx+hw, cy+hh, cz+hd),
            Point3D(cx-hw, cy+hh, cz+hd),
        ]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            self.draw_line_3d(p[e[0]], p[e[1]], color)

    def flip(self):
        pygame.display.flip()

    def draw_text(self, text, pos, color=(255,255,255), size=20):
        font = pygame.font.SysFont('Arial', size)
        self.screen.blit(font.render(text, True, color), pos)

    def draw_minimap(self, drone_pos, goal, pillars, obstacles, drone_yaw):
        """Draw minimap in top right corner"""
        map_size = 120
        map_x = self.width - map_size - 10
        map_y = 10
        scale = 5  # pixels per meter

        # Background
        pygame.draw.rect(self.screen, (20, 20, 30), (map_x, map_y, map_size, map_size))
        pygame.draw.rect(self.screen, (100, 100, 100), (map_x, map_y, map_size, map_size), 1)

        # Convert drone_pos tuple to x, z
        drone_x = drone_pos[0]
        drone_z = drone_pos[2] if len(drone_pos) >= 3 else drone_pos[0]

        # Draw goal (green square)
        gx = (goal[0] - drone_x) * scale + map_size / 2
        gz = (goal[1] - drone_z) * scale + map_size / 2
        if 0 <= gx <= map_size and 0 <= gz <= map_size:
            pygame.draw.rect(self.screen, (50, 180, 50), (int(map_x + gx - 3), int(map_y + gz - 3), 6, 6))

        # Draw obstacles (purple rectangles)
        for ox, oz in obstacles:
            dx = (ox - drone_x) * scale + map_size / 2
            dz = (oz - drone_z) * scale + map_size / 2
            if 0 <= dx <= map_size and 0 <= dz <= map_size:
                pygame.draw.rect(self.screen, (200, 0, 200), (int(map_x + dx - 2), int(map_y + dz - 2), 4, 4))

        # Draw pillars (gray circles)
        for ox, oz in pillars:
            dx = (ox - drone_x) * scale + map_size / 2
            dz = (oz - drone_z) * scale + map_size / 2
            if 0 <= dx <= map_size and 0 <= dz <= map_size:
                pygame.draw.circle(self.screen, (150, 150, 150), (int(map_x + dx), int(map_y + dz)), 3)

        # Draw drone (green circle at center)
        cx = map_x + map_size / 2
        cy = map_y + map_size / 2
        pygame.draw.circle(self.screen, (0, 255, 0), (int(cx), int(cy)), 4)

        # Draw drone direction indicator (line pointing forward)
        dir_len = 15
        pygame.draw.line(self.screen, (0, 255, 0), (cx, cy),
                        (cx + math.sin(drone_yaw) * dir_len, cy - math.cos(drone_yaw) * dir_len), 2)

        # Draw start position (cyan dot)
        sx = (0 - drone_x) * scale + map_size / 2
        sz = (0 - drone_z) * scale + map_size / 2
        if 0 <= sx <= map_size and 0 <= sz <= map_size:
            pygame.draw.circle(self.screen, (0, 255, 255), (int(map_x + sx), int(map_y + sz)), 2)


class UAV3DViewer:
    def __init__(self):
        self.renderer = Renderer3D()
        self.simulator = WebotsSimulator()
        self.simulator.connect()
        self.goal = (18, 18)  # Inside simulation bounds (-10 to 20)

        self.pillars = [(5,5), (10,5), (15,5), (5,10), (10,15)]
        self.obstacles = [(3,3), (7,3), (12,3), (5,8), (10,8), (3,13), (13,18)]
        self.colors = [(0,200,0), (0,150,200), (200,0,200), (200,200,0), (255,100,100)]

        # Collision radii based on actual size
        self.pillar_radius = 0.2  # 0.4 width / 2
        self.obstacle_radius = 0.4  # 0.8 width / 2
        self.drone_radius = 0.2  # Small drone
        self.avoid_dist = 5.0  # Start avoiding at this distance

    def reset(self):
        """Reset simulator to initial position"""
        self.simulator._simulated_state = {
            'position': (0.0, 0.5, 0.0),
            'orientation': 0.0,
            'velocity': (0.0, 0.0, 0.0),
        }

    def check_collision(self, pose):
        """Check if drone collides with any obstacle"""
        x, z = pose[0], pose[2]

        for ox, oz in self.pillars:
            dist = math.sqrt((x-ox)**2 + (z-oz)**2)
            if dist < self.pillar_radius + self.drone_radius:
                return True, f"COLLISION at ({ox:.0f}, {oz:.0f})"

        for ox, oz in self.obstacles:
            dist = math.sqrt((x-ox)**2 + (z-oz)**2)
            if dist < self.obstacle_radius + self.drone_radius:
                return True, f"COLLISION at ({ox:.0f}, {oz:.0f})"

        return False, None

    def navigate(self, pose):
        """Navigation with obstacle avoidance - prefer paths around obstacles"""
        x, z = pose[0], pose[2]

        # Goal direction
        goal_dx = self.goal[0] - x
        goal_dz = self.goal[1] - z
        goal_dist = math.sqrt(goal_dx**2 + goal_dz**2)

        if goal_dist < 0.5:
            return 0, 0, False

        goal_dir_x = goal_dx / goal_dist
        goal_dir_z = goal_dz / goal_dist

        # Calculate avoidance direction
        avoid_x, avoid_z = 0, 0
        is_avoiding = False

        for ox, oz in self.pillars:
            dx = x - ox
            dz = z - oz
            dist = math.sqrt(dx**2 + dz**2)

            if dist < self.avoid_dist:
                strength = (self.avoid_dist - dist) / self.avoid_dist * 8.0
                if dist > 0.01:
                    avoid_x += (dx / dist) * strength
                    avoid_z += (dz / dist) * strength
                is_avoiding = True

        for ox, oz in self.obstacles:
            dx = x - ox
            dz = z - oz
            dist = math.sqrt(dx**2 + dz**2)

            if dist < self.avoid_dist:
                strength = (self.avoid_dist - dist) / self.avoid_dist * 8.0
                if dist > 0.01:
                    avoid_x += (dx / dist) * strength
                    avoid_z += (dz / dist) * strength
                is_avoiding = True

        # Combine goal and avoidance
        if is_avoiding:
            total_x = goal_dir_x * 0.2 + avoid_x * 0.8
            total_z = goal_dir_z * 0.2 + avoid_z * 0.8

            # Check if directions cancel out (total too small)
            mag = math.sqrt(total_x**2 + total_z**2)
            if mag < 0.1:
                # Use perpendicular to avoidance direction
                total_x = -avoid_z
                total_z = avoid_x
                if abs(total_x) < 0.01 and abs(total_z) < 0.01:
                    # Fallback: use perpendicular to goal
                    total_x = -goal_dir_z
                    total_z = goal_dir_x
        else:
            total_x = goal_dir_x
            total_z = goal_dir_z

        # Normalize
        mag = math.sqrt(total_x**2 + total_z**2)
        if mag > 0.01:
            total_x /= mag
            total_z /= mag

        return total_x, total_z, is_avoiding

    def run(self):
        print("=" * 50)
        print("UAV 3D Navigation")
        print("Q: Quit | S: Restart")
        print("=" * 50)

        running = True
        end_state = None
        collision_msg = ""

        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_q:
                        running = False
                    elif event.key == K_s:
                        self.reset()
                        end_state = None
                        collision_msg = ""

            self.simulator.step()
            pose = self.simulator.get_pose()

            dx = self.goal[0] - pose[0]
            dz = self.goal[1] - pose[2]
            dist = math.sqrt(dx**2 + dz**2)

            # Check collision
            collided, collision_msg = self.check_collision(pose)
            if collided and end_state is None:
                end_state = 'collision'
                self.simulator.set_velocity(0, 0, 0)

            # Check arrival
            if dist <= 0.5 and end_state is None:
                end_state = 'arrived'
                self.simulator.set_velocity(0, 0, 0)

            # Navigation
            if end_state is None and dist > 0.5:
                speed = 2.0  # Faster speed
                dir_x, dir_z, is_avoiding = self.navigate(pose)
                self.simulator.set_velocity(dir_x * speed, 0, dir_z * speed)
            elif end_state is None:
                self.simulator.set_velocity(0, 0, 0)

            self.renderer.camera.x = pose[0]
            self.renderer.camera.z = pose[2]
            self.renderer.camera.yaw = pose[3]

            self.renderer.clear()

            # Draw grid
            for i in range(-15, 16, 3):
                self.renderer.draw_line_3d(
                    Point3D(pose[0]-15, 0, pose[2]+i),
                    Point3D(pose[0]+15, 0, pose[2]+i), (150,150,150))
            for i in range(-15, 16, 3):
                self.renderer.draw_line_3d(
                    Point3D(pose[0]+i, 0, pose[2]-5),
                    Point3D(pose[0]+i, 0, pose[2]+15), (150,150,150))

            # Draw obstacles
            for i, (ox, oz) in enumerate(self.obstacles):
                color = self.colors[i % len(self.colors)]
                self.renderer.draw_box_3d((ox, 0.5, oz), (0.8, 1, 0.8), color)

            # Draw pillars
            for ox, oz in self.pillars:
                self.renderer.draw_box_3d((ox, 1.5, oz), (0.4, 3, 0.4), (80,80,80))

            # Draw goal
            self.renderer.draw_box_3d((18, 1.5, 18), (1.5, 3, 0.2), (50,180,50))
            self.renderer.draw_box_3d((18, 3, 18), (2, 0.2, 0.5), (50,180,50))

            # Draw start
            self.renderer.draw_box_3d((0, 0.1, 0), (0.6, 0.2, 0.6), (0,255,255))

            # Draw minimap
            self.renderer.draw_minimap(pose, self.goal, self.pillars, self.obstacles, pose[3])

            # HUD
            self.renderer.draw_text(f"Step: {self.simulator.timestep_elapsed // self.simulator.timestep}", (10, 10))
            self.renderer.draw_text(f"Position: ({pose[0]:.1f}, {pose[2]:.1f})", (10, 35))
            self.renderer.draw_text(f"Yaw: {math.degrees(pose[3]):.1f}", (10, 60))
            self.renderer.draw_text(f"Distance: {dist:.1f}m", (10, 85))
            self.renderer.draw_text(f"Step: {self.simulator.timestep_elapsed // self.simulator.timestep}", (10, 10))
            self.renderer.draw_text(f"Position: ({pose[0]:.1f}, {pose[2]:.1f})", (10, 35))
            self.renderer.draw_text(f"Yaw: {math.degrees(pose[3]):.1f}", (10, 60))
            self.renderer.draw_text(f"Distance: {dist:.1f}m", (10, 85))

            if end_state == 'collision':
                self.renderer.draw_text("END - COLLISION!", (320, 280), (255,50,50), 36)
                self.renderer.draw_text("Press S to restart", (320, 320), (200,200,200), 20)
            elif end_state == 'arrived':
                self.renderer.draw_text("END - ARRIVED!", (320, 280), (50,255,50), 36)
                self.renderer.draw_text("Press S to restart", (320, 320), (200,200,200), 20)
            else:
                self.renderer.draw_text("Q: Quit | S: Restart", (10, 110))

            self.renderer.flip()
            self.renderer.clock.tick(60)  # Faster frame rate

        pygame.quit()
        self.simulator.close()
        print("Done!")


if __name__ == "__main__":
    viewer = UAV3DViewer()
    viewer.run()