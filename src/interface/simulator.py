"""
Webots仿真接口模块
支持两种模式：
1. 模拟模式 - 生成测试图像
2. 真实模式 - 通过TCP连接Webots仿真器
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import time
import os
import socket
import json
import threading

# 设置Webots环境变量
os.environ.setdefault('WEBOTS_HOME', '/Applications/Webots.app')

# 尝试添加Webots lib路径
WEBOTS_LIB_PATH = "/Applications/Webots.app/Contents/lib/controller/python"
if os.path.exists(WEBOTS_LIB_PATH):
    import sys
    if WEBOTS_LIB_PATH not in sys.path:
        sys.path.insert(0, WEBOTS_LIB_PATH)


class WebotsSimulator:
    """Webots仿真器接口"""

    def __init__(self, config: Optional[Dict] = None, use_real=False):
        self.config = config or {}
        self.use_real = use_real

        # 机器人配置
        self.robot_name = self.config.get('robot_name', 'uav_navigation')
        self.timestep = self.config.get('timestep', 32)  # ms

        # 传感器配置
        self.camera_width = self.config.get('camera_width', 640)
        self.camera_height = self.config.get('camera_height', 480)
        self.camera_fov = self.config.get('camera_fov', 1.5708)

        # 初始位置
        self.initial_position = self.config.get('initial_position',
                                               {'x': 0, 'y': 0.5, 'z': 0})

        # 状态
        self.robot = None
        self.timestep_elapsed = 0
        self.is_running = False
        self._has_controller = False

        # 模拟状态
        self._simulated_state = {
            'position': (0.0, 0.5, 0.0),
            'orientation': 0.0,
            'velocity': (0.0, 0.0, 0.0),
        }

        # TCP连接（用于真实Webots）
        self._tcp_socket = None
        self._tcp_host = '127.0.0.1'
        self._tcp_port = 5599
        self._tcp_connected = False
        self._tcp_thread = None
        self._data_buffer = {}
        self._data_lock = threading.Lock()

        # 默认尝试连接真实Webots
        self._connect_to_webots()

    def connect(self):
        """兼容旧代码的连接方法"""
        self.is_running = True
        return True

    def _connect_mock(self):
        """连接模拟模式"""
        self._has_controller = False
        self.is_running = True
        print("Webots控制器库未安装，使用模拟模式")

    def _connect_to_webots(self):
        """连接到真实Webots仿真器"""
        try:
            self._tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._tcp_socket.connect((self._tcp_host, self._tcp_port))
            self._tcp_connected = True
            self._has_controller = True
            print(f"已连接到Webots仿真器: {self._tcp_host}:{self._tcp_port}")
        except Exception as e:
            print(f"连接Webots失败: {e}")
            self._connect_mock()

    def _tcp_send(self, cmd):
        """通过TCP发送命令"""
        if self._tcp_connected and self._tcp_socket:
            try:
                self._tcp_socket.sendall(json.dumps(cmd).encode('utf-8'))
            except:
                self._tcp_connected = False

    def _tcp_receive(self):
        """通过TCP接收数据"""
        if self._tcp_connected and self._tcp_socket:
            try:
                self._tcp_socket.settimeout(0.01)
                data = self._tcp_socket.recv(4096)
                if data:
                    with self._data_lock:
                        self._data_buffer = json.loads(data.decode('utf-8'))
            except socket.timeout:
                pass
            except Exception as e:
                print(f"接收错误: {e}")
                self._tcp_connected = False

    def step(self) -> int:
        """执行一个仿真步"""
        self.timestep_elapsed += self.timestep

        if self._tcp_connected:
            # 接收数据
            self._tcp_receive()
            time.sleep(self.timestep / 1000.0)
            return self.timestep
        else:
            # 模拟模式：更新模拟状态
            self._update_simulated_state()
            time.sleep(self.timestep / 1000.0)
            return self.timestep

    def _update_simulated_state(self):
        """更新模拟状态"""
        vx, vy, vz = self._simulated_state['velocity']
        dt = self.timestep / 1000.0

        pos = list(self._simulated_state['position'])
        # 坐标系: pos[0]=X(左右), pos[1]=Y(高度), pos[2]=Z(前后)
        # 速度: vx=X方向, vy=高度, vz=Z方向
        pos[0] += vx * dt      # X方向移动
        pos[1] += vy * dt      # 高度变化
        pos[2] += vz * dt      # Z方向移动

        # 更新朝向（根据移动方向）
        if abs(vx) > 0.01 or abs(vz) > 0.01:
            self._simulated_state['orientation'] = np.arctan2(vz, vx)

        # 限制范围
        pos[0] = max(-10, min(20, pos[0]))
        pos[1] = max(0.5, min(5, pos[1]))  # 高度
        pos[2] = max(-10, min(20, pos[2]))  # 前后

        self._simulated_state['position'] = tuple(pos)

    def get_camera_image(self) -> np.ndarray:
        """获取摄像头图像"""
        if self._tcp_connected:
            with self._data_lock:
                if 'camera' in self._data_buffer and self._data_buffer['camera']:
                    # 从Webots接收的图像数据
                    img_data = self._data_buffer['camera']
                    if isinstance(img_data, list):
                        # Webots返回的是字节列表
                        image = np.array(img_data, dtype=np.uint8)
                        image = image.reshape(self.camera_height, self.camera_width, 4)
                        return image[:, :, :3]  # RGBA -> RGB

        # 模拟模式：返回测试图像
        return self._generate_test_image()

    def _generate_test_image(self) -> np.ndarray:
        """生成测试图像 - 随时间动态变化"""
        import cv2

        h, w = self.camera_height, self.camera_width
        image = np.zeros((h, w, 3), dtype=np.uint8)

        # 获取当前位置用于生成动态视角
        pos = self._simulated_state['position']
        yaw = self._simulated_state['orientation']

        # 时间因子（用于动画）
        time_factor = self.timestep_elapsed / 1000.0

        # 绘制透视网格地面（模拟走廊/仓库）
        horizon = h // 3

        # 天空渐变
        for y in range(horizon):
            factor = y / horizon
            image[y, :] = [
                int(135 + 40 * factor),  # B
                int(206 - 30 * factor),  # G
                int(235 - 50 * factor)   # R
            ]

        # 地面渐变（模拟深度）
        for y in range(horizon, h):
            factor = (y - horizon) / (h - horizon)
            gray = int(60 + 80 * factor)
            image[y, :] = [gray, gray, gray]

        # 绘制透视网格线（随位置变化）
        num_lines = 12
        center_x = w // 2

        # 水平线（模拟地面网格）
        for i in range(1, num_lines + 1):
            y = horizon + int((h - horizon) * i / num_lines)
            cv2.line(image, (0, y), (w, y), (80, 80, 80), 1)

        # 垂直消失线（从地平线中心向外发散）
        for i in range(-6, 7, 2):
            x_end = center_x + i * w // 5
            cv2.line(image, (center_x, horizon), (x_end, h), (70, 70, 70), 1)

        # 添加墙壁
        wall_left = int(50 + np.sin(time_factor) * 10)
        wall_right = int(w - 50 - np.sin(time_factor) * 10)
        cv2.line(image, (wall_left, horizon), (wall_left, h), (50, 50, 50), 3)
        cv2.line(image, (wall_right, horizon), (wall_right, h), (50, 50, 50), 3)

        # 添加动态立柱（模拟货架，根据位置变化）
        pillar_positions = [120, 250, 380, 500]
        for px in pillar_positions:
            offset = int(np.sin(time_factor + px * 0.01) * 5)
            px = max(50, min(w - 50, px + offset))
            # 柱子高度随位置变化
            pillar_height = int(80 + np.sin(px * 0.1) * 20)
            cv2.rectangle(image,
                         (px - 15, horizon + 20),
                         (px + 15, h - 30),
                         (60 + int(px % 40), 60, 60), -1)
            cv2.rectangle(image,
                         (px - 15, horizon + 20),
                         (px + 15, h - 30),
                         (100, 100, 100), 2)

        # 添加前景障碍物（动态颜色）
        obstacles = [
            (80 + int(time_factor * 20) % 30, h - 140, 70, 100, (0, 180, 0)),      # 左绿色（移动）
            (w - 200 - int(time_factor * 15) % 30, h - 160, 80, 120, (0, 150, 200)),  # 右蓝色（移动）
            (center_x - 100 + int(np.sin(time_factor) * 30), h - 110, 60, 70, (0, 200, 200)),  # 中间青色
            (center_x + 60 + int(time_factor * 25) % 40, h - 130, 50, 90, (200, 0, 200)),  # 中右紫色
        ]

        for ox, oy, ow, oh, color in obstacles:
            # 确保在范围内
            ox = max(0, min(w - ow, ox))
            oy = max(horizon, min(h - oh, oy))

            # 动态颜色变化
            dynamic_color = (
                int(color[0] + 30 * np.sin(time_factor)),
                int(color[1] + 30 * np.sin(time_factor + 1)),
                int(color[2] + 30 * np.sin(time_factor + 2))
            )
            dynamic_color = tuple(max(0, min(255, c)) for c in dynamic_color)

            cv2.rectangle(image, (ox, oy), (ox + ow, oy + oh), dynamic_color, -1)
            cv2.rectangle(image, (ox, oy), (ox + ow, oy + oh), (255, 255, 255), 2)

        # 目标门（右侧）
        door_x, door_y = w - 120, horizon + 40
        cv2.rectangle(image, (door_x, door_y), (door_x + 90, h - 40), (50, 180, 50), 2)
        cv2.putText(image, "GOAL", (door_x + 20, door_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

        # 起点标记
        cv2.circle(image, (60, h - 60), 25, (0, 255, 255), 2)
        cv2.putText(image, "START", (30, h - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 当前步数显示
        step = self.timestep_elapsed // self.timestep
        cv2.putText(image, f"Step: {step} Pos:({pos[0]:.1f},{pos[2]:.1f})",
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 边界装饰
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), (50, 50, 50), 2)

        # 添加轻微噪点
        noise = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image

    def get_pose(self) -> Tuple[float, float, float, float]:
        """获取机器人位姿"""
        if self._tcp_connected:
            with self._data_lock:
                if 'pose' in self._data_buffer:
                    pose = self._data_buffer['pose']
                    return tuple(pose)

        # 模拟模式
        return (*self._simulated_state['position'][:2],
                self._simulated_state['position'][2],
                self._simulated_state['orientation'])

    def get_velocity(self) -> Tuple[float, float, float]:
        """获取机器人速度"""
        if self._tcp_connected:
            with self._data_lock:
                if 'velocity' in self._data_buffer:
                    vel = self._data_buffer['velocity']
                    return tuple(vel)

        return self._simulated_state['velocity']

    def set_velocity(self, vx: float, vy: float, vz: float,
                    yaw_rate: float = 0.0):
        """设置机器人速度"""
        # 更新模拟状态
        self._simulated_state['velocity'] = (vx, vy, vz)

        # 发送到Webots
        if self._tcp_connected:
            self._tcp_send({
                'velocity': [vx, vy, vz],
                'yaw_rate': yaw_rate
            })

    def reset(self):
        """重置仿真"""
        self.timestep_elapsed = 0
        self._simulated_state = {
            'position': (0.0, 0.5, 0.0),
            'orientation': 0.0,
            'velocity': (0.0, 0.0, 0.0),
        }

    def close(self):
        """关闭连接"""
        self.is_running = False
        if self._tcp_socket:
            self._tcp_socket.close()
        print("仿真器连接已关闭")

    def get_simulation_time(self) -> float:
        """获取仿真时间（秒）"""
        return self.timestep_elapsed / 1000.0


class SimulationRecorder:
    """仿真数据记录器"""

    def __init__(self):
        self.trajectory = []
        self.detections_log = []
        self.path_log = []
        self.metrics = {
            'collision_count': 0,
            'path_length': 0.0,
            'success': False,
            'duration': 0.0,
        }

    def record_state(self, pose: Tuple, detections: List = None,
                    path: List = None):
        """记录状态"""
        self.trajectory.append(pose)

        if detections:
            self.detections_log.append({
                'time': len(self.trajectory),
                'count': len(detections),
                'types': [d['class_name'] for d in detections]
            })

        if path:
            self.path_log.append(path)

    def finalize(self, success: bool):
        """完成记录"""
        self.metrics['success'] = success
        self.metrics['duration'] = len(self.trajectory) * 0.032

        if len(self.trajectory) > 1:
            total_length = 0.0
            for i in range(1, len(self.trajectory)):
                dist = np.linalg.norm(
                    np.array(self.trajectory[i][:2]) -
                    np.array(self.trajectory[i-1][:2])
                )
                total_length += dist
            self.metrics['path_length'] = total_length

    def get_summary(self) -> Dict:
        """获取统计摘要"""
        return {
            'total_steps': len(self.trajectory),
            'total_detections': sum(d['count'] for d in self.detections_log),
            'path_length': self.metrics['path_length'],
            'collision_count': self.metrics['collision_count'],
            'success': self.metrics['success'],
            'duration': self.metrics['duration'],
            'success_rate': 1.0 if self.metrics['success'] else 0.0,
        }


# 单元测试
if __name__ == "__main__":
    config = {
        'robot_name': 'test_robot',
        'timestep': 32,
        'camera_width': 640,
        'camera_height': 480,
        'initial_position': {'x': 0, 'y': 0.5, 'z': 0},
    }

    # 测试模拟模式
    print("测试模拟模式...")
    sim = WebotsSimulator(config)
    image = sim.get_camera_image()
    print(f"图像形状: {image.shape}")
    sim.close()
    print("模拟模式测试完成!")

    # 测试真实模式
    print("\n测试真实模式连接...")
    sim_real = WebotsSimulator(config, use_real=False)  # 先用模拟模式
    sim_real.close()