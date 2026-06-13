#!/usr/bin/env python3
"""
Webots Crazyflie Viewer
Connects to Crazyflie controller via TCP port 5599
"""

import sys
import os
import math
import json
import socket
import threading
import io
import struct

os.environ['SDL_VIDEODRIVER'] = 'cocoa'
import pygame
from pygame.locals import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class WebotsViewer:
    def __init__(self, host='127.0.0.1', port=5599):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.running = True

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Crazyflie Viewer")
        self.width = 800
        self.height = 600
        self.clock = pygame.time.Clock()

        # Drone state
        self.pose = [0, 0, 0, 0, 0, 0]  # x, y, z, yaw, roll, pitch
        self.target_thrust = 0.0  # Start with motors off
        self.target_pitch = 0.0
        self.target_roll = 0.0

        # Smoothing filter
        self.yaw_filtered = 0.0
        self.filter_alpha = 0.3  # Low-pass filter

        # Camera image
        self.camera_surface = None
        self.camera_width = 0
        self.camera_height = 0

        # Command rate limiting
        self.last_cmd_time = 0
        self.cmd_interval = 0.05

        # Connect to Webots
        self.connect()

    def connect(self):
        """Connect to Webots controller"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2.0)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to Crazyflie at {self.host}:{self.port}")

            # Start receiving thread
            self.recv_thread = threading.Thread(target=self.receive_loop, daemon=True)
            self.recv_thread.start()

        except Exception as e:
            print(f"Cannot connect: {e}")
            print("Make sure Webots is running with Crazyflie simulation")
            self.connected = False

    def receive_loop(self):
        """Receive data from Webots"""
        buffer = ""
        while self.running and self.connected:
            try:
                data = self.socket.recv(16384)  # Larger buffer
                if not data:
                    break
                buffer += data.decode('utf-8', errors='ignore')

                # Process all complete lines
                lines = buffer.split('\n')
                buffer = lines[-1]  # Keep incomplete line

                for line in lines[:-1]:
                    if line.strip():
                        try:
                            msg = json.loads(line)
                            if 'pose' in msg:
                                raw_pose = msg['pose']
                                # Apply smoothing to yaw for display only
                                if len(raw_pose) > 3:
                                    raw_yaw = raw_pose[3]
                                    self.yaw_filtered = 0.7 * self.yaw_filtered + 0.3 * raw_yaw
                                    self.pose = [raw_pose[0], raw_pose[1], raw_pose[2], self.yaw_filtered, raw_pose[4] if len(raw_pose) > 4 else 0, raw_pose[5] if len(raw_pose) > 5 else 0]
                                else:
                                    self.pose = raw_pose
                        except json.JSONDecodeError:
                            pass
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Receive error: {e}")
                break

        self.connected = False

    def send_command(self):
        """Send thrust command to Crazyflie"""
        if self.connected and self.socket:
            cmd = json.dumps({'t': self.target_thrust, 'p': self.target_pitch, 'r': self.target_roll})
            try:
                self.socket.sendall((cmd + '\n').encode('utf-8'))
            except:
                pass

    def draw_hud(self):
        """Draw HUD overlay"""
        pose = self.pose
        x, y, z = pose[0], pose[1], pose[2]
        yaw = pose[3] if len(pose) > 3 else 0

        # Info panel
        pygame.draw.rect(self.screen, (20, 20, 30), (10, 10, 280, 180))
        pygame.draw.rect(self.screen, (80, 80, 80), (10, 10, 280, 180), 2)

        font = pygame.font.SysFont('Arial', 18)
        texts = [
            f"Position: ({x:.3f}, {y:.3f}, {z:.3f})",
            f"X: {x:.2f}m  Z: {z:.2f}m",
            f"Altitude: {y:.3f}m",
            f"Yaw: {math.degrees(yaw):.1f}deg",
            f"Thrust: {self.target_thrust:.0f}  Cam: OFF",
            f"Status: {'Connected' if self.connected else 'Disconnected'}"
        ]

        for i, text in enumerate(texts):
            color = (0, 255, 0) if self.connected else (255, 100, 100)
            surf = font.render(text, True, color)
            self.screen.blit(surf, (20, 18 + i * 28))

        # Draw camera feed placeholder
        pygame.draw.rect(self.screen, (40, 40, 50), (300, 200, 320, 240))
        font = pygame.font.SysFont('Arial', 16)
        text = font.render("Camera: Disabled", True, (100, 100, 100))
        self.screen.blit(text, (350, 308))
        pygame.draw.rect(self.screen, (80, 80, 80), (300, 200, 320, 240), 1)

        # Draw minimap
        self.draw_minimap()

        # Controls
        font_small = pygame.font.SysFont('Arial', 14)
        controls = ["UP/DOWN: Thrust", "W/S: Forward/Back", "A/D: Spin (hold)", "Q: Quit"]
        for i, text in enumerate(controls):
            surf = font_small.render(text, True, (180, 180, 180))
            self.screen.blit(surf, (10, 555 - (len(controls) - i) * 18))

    def draw_minimap(self):
        """Draw top-down minimap"""
        map_size = 140
        map_x = self.width - map_size - 10
        map_y = 10
        scale = 15  # pixels per meter

        # Background
        pygame.draw.rect(self.screen, (20, 20, 30), (map_x, map_y, map_size, map_size))
        pygame.draw.rect(self.screen, (80, 80, 80), (map_x, map_y, map_size, map_size), 2)

        # Grid
        for i in range(-3, 4, 1):
            px = map_x + map_size/2 + i * scale
            pygame.draw.line(self.screen, (40, 40, 40), (px, map_y), (px, map_y + map_size))
            pz = map_y + map_size/2 + i * scale
            pygame.draw.line(self.screen, (40, 40, 40), (map_x, pz), (map_x + map_size, pz))

        # Start point
        sx = map_x + map_size/2
        sz = map_y + map_size/2
        pygame.draw.circle(self.screen, (0, 255, 255), (int(sx), int(sz)), 4)

        # Drone (center)
        cx = map_x + map_size/2
        cy = map_y + map_size/2
        pygame.draw.circle(self.screen, (0, 255, 0), (int(cx), int(cy)), 5)

        # Apply low-pass filter to yaw for display
        raw_yaw = self.pose[3] if len(self.pose) > 3 else 0
        self.yaw_filtered = self.filter_alpha * raw_yaw + (1 - self.filter_alpha) * self.yaw_filtered

        # Direction (use filtered yaw)
        pygame.draw.line(self.screen, (0, 255, 0),
                        (cx, cy),
                        (cx + math.sin(self.yaw_filtered) * 12, cy - math.cos(self.yaw_filtered) * 12), 2)

    def run(self):
        """Main loop"""
        print("=" * 50)
        print("Crazyflie Webots Viewer")
        print("Controls: W/S = Forward/Back, A/D = Turn")
        print("=" * 50)

        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_q:
                        self.running = False

            # Keyboard input
            keys = pygame.key.get_pressed()

            # Default: motors off
            self.target_thrust = 0.0
            self.target_pitch = 0.0
            self.target_roll = 0.0

            # UP = fly, DOWN = descend
            if keys[K_UP]:
                self.target_thrust = min(400, self.target_thrust + 3)
            elif keys[K_DOWN]:
                self.target_thrust = max(0, self.target_thrust - 3)

            # W/S = pitch (forward/back)
            if keys[K_w]:
                self.target_pitch = 0.8   # Forward
            elif keys[K_s]:
                self.target_pitch = -0.8  # Backward

            # A/D = yaw (spin) - only when pressed
            if keys[K_a]:
                self.target_roll = -0.5   # Spin left (reduced from 0.8)
            elif keys[K_d]:
                self.target_roll = 0.5    # Spin right (reduced from 0.8)

            # Send command (rate limited)
            import time
            now = time.time()
            if now - self.last_cmd_time > self.cmd_interval:
                self.send_command()
                self.last_cmd_time = now

            # Draw
            self.screen.fill((60, 60, 70))
            self.draw_hud()

            pygame.display.flip()
            self.clock.tick(30)

        if self.socket:
            self.socket.close()
        pygame.quit()
        print("Viewer closed")


if __name__ == "__main__":
    viewer = WebotsViewer()
    viewer.run()