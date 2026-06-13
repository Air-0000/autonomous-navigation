# Webots Controller for UAV
# Put this in your Webots world's robot controller

from controller import Robot, Camera, GPS, Gyro, Compass
import socket
import json
import threading

class UAVController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Get devices
        self.gps = self.robot.getGPS("gps")
        self.compass = self.robot.getCompass("compass")
        self.camera = self.robot.getCamera("camera")

        if self.gps:
            self.gps.enable(self.timestep)
        if self.compass:
            self.compass.enable(self.timestep)
        if self.camera:
            self.camera.enable(self.timestep)

        # Motor setup
        self.motors = []
        for i in range(4):
            motor = self.robot.getMotor(f"propeller{i}")
            if motor:
                motor.setPosition(float('inf'))
                motor.setVelocity(0)
                self.motors.append(motor)

        # TCP connection to viewer
        self.socket = None
        self.connected = False
        self.start_tcp_server()

    def start_tcp_server(self):
        """Start TCP server to accept viewer connection"""
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind(('127.0.0.1', 5599))
            self.server.listen(1)
            self.server.settimeout(0.1)
            print("Waiting for viewer connection on port 5599...")
            self.accept_loop()
        except Exception as e:
            print(f"TCP server error: {e}")

    def accept_loop(self):
        """Accept viewer connections"""
        while self.robot.step(self.timestep) != -1:
            try:
                self.socket, addr = self.server.accept()
                self.connected = True
                print(f"Viewer connected from {addr}")
                break
            except socket.timeout:
                continue

        if self.connected:
            self.send_loop()

    def send_loop(self):
        """Send data to viewer"""
        while self.robot.step(self.timestep) != -1:
            # Get sensor data
            pose = self.get_pose()

            # Send data
            data = {'pose': pose}
            try:
                self.socket.sendall((json.dumps(data) + '\n').encode('utf-8'))

                # Receive commands
                try:
                    self.socket.settimeout(0.001)
                    cmd = self.socket.recv(1024)
                    if cmd:
                        self.handle_command(json.loads(cmd.decode('utf-8')))
                except socket.timeout:
                    pass
            except:
                self.connected = False
                break

        self.accept_loop()

    def get_pose(self):
        """Get robot pose"""
        pos = [0, 0, 0]
        yaw = 0

        if self.gps:
            gp = self.gps.getValues()
            pos = [gp[0], gp[1], gp[2]]

        if self.compass:
            north = self.compass.getValues()
            yaw = math.atan2(north[0], north[2])

        return [pos[0], pos[1], pos[2], yaw]

    def handle_command(self, cmd):
        """Handle velocity command"""
        if 'velocity' in cmd:
            vx, vy, vz = cmd['velocity']
            # Set motor velocities
            for motor in self.motors:
                motor.setVelocity(vx * 10)

        if 'yaw_rate' in cmd:
            yaw_rate = cmd['yaw_rate']
            # Adjust motor speeds for turning
            pass

    def run(self):
        """Main loop - just wait for connections"""
        while self.robot.step(self.timestep) != -1:
            pass

if __name__ == "__main__":
    import math
    controller = UAVController()
    controller.run()