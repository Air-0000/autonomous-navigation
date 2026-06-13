#!/usr/bin/env python3
"""
Webots无人机控制器
运行在Webots仿真器内部，与主程序通过TCP通信
"""

import json
import socket
import threading
from controller import Robot, Camera, Gyro, GPS, Compass, Motor

class WebotsController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # 初始化设备
        self.camera = self.robot.getCamera("camera")
        self.gyro = self.robot.getGyro("gyro")
        self.gps = self.robot.getGPS("gps")
        self.compass = self.robot.getCompass("compass")

        # 启用设备
        if self.camera:
            self.camera.enable(self.timestep)
        if self.gyro:
            self.gyro.enable(self.timestep)
        if self.gps:
            self.gps.enable(self.timestep)
        if self.compass:
            self.compass.enable(self.timestep)

        # 获取螺旋桨电机
        self.motors = []
        motor_names = [
            "front left propeller motor",
            "front right propeller motor",
            "rear left propeller motor",
            "rear right propeller motor"
        ]
        for name in motor_names:
            motor = self.robot.getMotor(name)
            if motor:
                motor.setPosition(float('inf'))
                motor.setVelocity(0)
                self.motors.append(motor)

        # 状态
        self.velocity = [0.0, 0.0, 0.0]  # vx, vy, vz
        self.yaw_rate = 0.0

        # TCP服务器
        self.server_socket = None
        self.client_socket = None
        self.server_port = 5599

        # 启动TCP服务器
        self._start_server()

    def _start_server(self):
        """启动TCP服务器接收主程序指令"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('127.0.0.1', self.server_port))
        self.server_socket.listen(1)
        print(f"Webots控制器等待连接端口 {self.server_port}...")

        self.server_socket.settimeout(0.1)  # 非阻塞

    def _accept_connection(self):
        """接受主程序连接"""
        try:
            if self.client_socket is None:
                self.client_socket, addr = self.server_socket.accept()
                print(f"主程序已连接: {addr}")
        except socket.timeout:
            pass

    def _receive_command(self):
        """接收主程序指令"""
        if self.client_socket is None:
            return None

        try:
            self.client_socket.settimeout(0.001)
            data = self.client_socket.recv(1024)
            if data:
                return json.loads(data.decode('utf-8'))
        except socket.timeout:
            pass
        except Exception as e:
            print(f"接收错误: {e}")
            self.client_socket = None

        return None

    def _send_data(self, data):
        """发送数据到主程序"""
        if self.client_socket:
            try:
                self.client_socket.sendall(json.dumps(data).encode('utf-8'))
            except:
                self.client_socket = None

    def run(self):
        """主循环"""
        print("Webots控制器启动!")

        frame_count = 0

        while self.robot.step(self.timestep) != -1:
            # 接受连接
            self._accept_connection()

            # 接收指令
            cmd = self._receive_command()
            if cmd:
                self._process_command(cmd)

            # 发送数据
            data = {
                'camera': self.get_camera_image(),
                'pose': self.get_pose(),
                'velocity': self.get_velocity(),
            }
            self._send_data(data)

            frame_count += 1

        # 清理
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()

    def _process_command(self, cmd):
        """处理控制指令"""
        if 'velocity' in cmd:
            vx, vy, vz = cmd['velocity']
            self.velocity = [vx, vy, vz]

        if 'yaw_rate' in cmd:
            self.yaw_rate = cmd['yaw_rate']

        # 更新电机速度
        base_thrust = self._calculate_thrust(vx, vy, vz)
        for motor in self.motors:
            motor.setVelocity(base_thrust)

    def _calculate_thrust(self, vx, vy, vz):
        """计算螺旋桨推力"""
        # 简化模型：基于总速度计算基础推力
        speed = (vx**2 + vy**2 + vz**2)**0.5
        base_thrust = min(10.0, 5.0 + speed * 2)
        return base_thrust

    def get_camera_image(self):
        """获取摄像头图像"""
        if self.camera:
            image = self.camera.getImage()
            if image:
                return list(image)
        return None

    def get_pose(self):
        """获取位姿"""
        if self.gps:
            gps_values = self.gps.getValues()
            x, y, z = gps_values[0], gps_values[1], gps_values[2]
        else:
            x, y, z = 0, 0.5, 0

        if self.compass:
            compass_values = self.compass.getValues()
            yaw = self._get_yaw_from_compass(compass_values)
        else:
            yaw = 0

        return [x, y, z, yaw]

    def _get_yaw_from_compass(self, compass):
        """从罗盘计算偏航角"""
        return -1.5708 + compass[0] * 3.14159

    def get_velocity(self):
        """获取速度"""
        if self.gps:
            speed = self.gps.getSpeedVector()
            return list(speed)
        return [0.0, 0.0, 0.0]


if __name__ == "__main__":
    controller = WebotsController()
    controller.run()