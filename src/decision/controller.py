"""
控制器模块
PID控制和速度指令生成
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class VelocityCommand:
    """速度指令"""
    vx: float  # x方向速度 (m/s)
    vy: float  # y方向速度 (m/s)
    vz: float  # z方向速度 (m/s)
    yaw_rate: float  # 偏航角速度 (rad/s)
    timestamp: float


@dataclass
class Pose:
    """位姿"""
    x: float
    y: float
    z: float
    yaw: float


class PIDController:
    """PID控制器"""

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float = 0.032) -> float:
        """
        计算PID控制量

        Args:
            error: 误差（当前值 - 目标值）
            dt: 时间步长

        Returns:
            control: 控制量
        """
        # 积分
        self.integral += error * dt

        # 微分
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        # PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error

        return output

    def reset(self):
        """重置积分项"""
        self.integral = 0.0
        self.prev_error = 0.0


class PurePursuitController:
    """纯追踪控制器"""

    def __init__(self, lookahead_distance: float = 1.0,
                 lookahead_gain: float = 1.0):
        self.lookahead_distance = lookahead_distance
        self.lookahead_gain = lookahead_gain

    def compute(self, current_pose: Pose,
                path: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        计算速度和转向指令

        Args:
            current_pose: 当前位姿
            path: 路径点列表 [(x,y), ...]

        Returns:
            (velocity, yaw_rate): 速度和转向速率
        """
        if len(path) < 2:
            return 0.0, 0.0

        # 找到前瞻点
        lookahead_point = self._find_lookahead_point(current_pose, path)

        # 计算角度误差
        dx = lookahead_point[0] - current_pose.x
        dy = lookahead_point[1] - current_pose.y

        # 转换为车体坐标系
        cos_yaw = np.cos(current_pose.yaw)
        sin_yaw = np.sin(current_pose.yaw)

        # 旋转到车体坐标系
        x_car = dx * cos_yaw + dy * sin_yaw
        y_car = -dx * sin_yaw + dy * cos_yaw

        # 计算朝向角误差
        alpha = np.arctan2(y_car, x_car)

        # 计算转向速率
        L = 0.5  #  wheelbase
        K = 2 * np.sin(alpha) / self.lookahead_distance
        yaw_rate = K * 1.0  # 简化的速度

        # 限制yaw_rate
        yaw_rate = np.clip(yaw_rate, -1.5, 1.5)

        # 速度（与转向相反，简单策略）
        velocity = 1.0

        return velocity, yaw_rate

    def _find_lookahead_point(self, pose: Pose,
                              path: List[Tuple[float, float]]) -> Tuple[float, float]:
        """找到前瞻点"""
        # 动态计算前瞻距离
        lookahead = self.lookahead_distance + self.lookahead_gain * 0.5

        # 找到路径上距离当前位置 lookahead 距离的点
        best_point = path[-1]  # 默认返回终点
        best_dist = float('inf')

        for point in path:
            dist = np.sqrt(
                (point[0] - pose.x) ** 2 +
                (point[1] - pose.y) ** 2
            )

            if abs(dist - lookahead) < abs(best_dist - lookahead):
                best_dist = dist
                best_point = point

        return best_point


class NavigationController:
    """导航控制器 - 综合控制"""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {
                'max_velocity': 2.0,
                'max_acceleration': 1.0,
                'robot_radius': 0.3,
            }

        self.max_velocity = config['max_velocity']
        self.max_acceleration = config['max_acceleration']
        self.robot_radius = config['robot_radius']

        # 控制器
        self.pid_x = PIDController(kp=1.5, ki=0.1, kd=0.3)
        self.pid_y = PIDController(kp=1.5, ki=0.1, kd=0.3)
        self.pid_yaw = PIDController(kp=2.0, ki=0.0, kd=0.5)

        self.pure_pursuit = PurePursuitController()

        # 状态
        self.current_pose = Pose(x=0, y=0, z=0, yaw=0)
        self.current_velocity = (0.0, 0.0, 0.0)
        self.last_timestamp = 0.0

    def compute_command(self,
                      current_pose: Pose,
                      target_pose: Pose,
                      path: Optional[List[Tuple[float, float]]] = None
                      ) -> VelocityCommand:
        """
        计算速度指令

        Args:
            current_pose: 当前位姿
            target_pose: 目标位姿
            path: 路径（可选，用于纯追踪）

        Returns:
            command: 速度指令
        """
        self.current_pose = current_pose

        # 计算位置误差
        error_x = target_pose.x - current_pose.x
        error_y = target_pose.y - current_pose.y

        # 计算角度误差
        target_yaw = np.arctan2(error_y, error_x)
        error_yaw = target_yaw - current_pose.yaw

        # 归一化角度误差到 [-pi, pi]
        while error_yaw > np.pi:
            error_yaw -= 2 * np.pi
        while error_yaw < -np.pi:
            error_yaw += 2 * np.pi

        # 时间步长
        dt = 0.032

        # PID控制
        vx = self.pid_x.compute(error_x, dt)
        vy = self.pid_y.compute(error_y, dt)
        yaw_rate = self.pid_yaw.compute(error_yaw, dt)

        # 限制速度
        speed = np.sqrt(vx ** 2 + vy ** 2)
        if speed > self.max_velocity:
            scale = self.max_velocity / speed
            vx *= scale
            vy *= scale

        # 限制加速度
        dvx = vx - self.current_velocity[0]
        dvy = vy - self.current_velocity[1]
        accel = np.sqrt(dvx ** 2 + dvy ** 2) / dt

        if accel > self.max_acceleration:
            scale = self.max_acceleration * dt / max(accel, 1e-6)
            vx = self.current_velocity[0] + dvx * scale
            vy = self.current_velocity[1] + dvy * scale

        # 更新速度
        self.current_velocity = (vx, vy, 0.0)

        return VelocityCommand(
            vx=vx, vy=vy, vz=0.0,
            yaw_rate=yaw_rate,
            timestamp=self.last_timestamp
        )

    def compute_command_from_path(self,
                                  current_pose: Pose,
                                  path: List[Tuple[float, float]]
                                  ) -> VelocityCommand:
        """
        基于路径计算速度指令（纯追踪）

        Args:
            current_pose: 当前位姿
            path: 路径点

        Returns:
            command: 速度指令
        """
        self.current_pose = current_pose

        velocity, yaw_rate = self.pure_pursuit.compute(current_pose, path)

        # 限制速度
        velocity = min(velocity, self.max_velocity)

        return VelocityCommand(
            vx=velocity * np.cos(current_pose.yaw),
            vy=velocity * np.sin(current_pose.yaw),
            vz=0.0,
            yaw_rate=yaw_rate,
            timestamp=self.last_timestamp
        )

    def emergency_stop(self) -> VelocityCommand:
        """紧急停止"""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_yaw.reset()
        self.current_velocity = (0.0, 0.0, 0.0)

        return VelocityCommand(vx=0, vy=0, vz=0, yaw_rate=0, timestamp=0)

    def check_arrival(self, current_pose: Pose,
                     target_pose: Pose,
                     threshold: float = 0.2) -> bool:
        """检查是否到达目标"""
        dist = np.sqrt(
            (current_pose.x - target_pose.x) ** 2 +
            (current_pose.y - target_pose.y) ** 2
        )
        return dist < threshold


# 单元测试
if __name__ == "__main__":
    controller = NavigationController()

    # 模拟路径
    path = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)]

    # 模拟导航
    current = Pose(x=0.1, y=0.1, z=0, yaw=0)
    target = Pose(x=4, y=2, z=0, yaw=0)

    for step in range(100):
        command = controller.compute_command_from_path(current, path)

        # 模拟移动（简化）
        current.x += command.vx * 0.032
        current.y += command.vy * 0.032
        current.yaw += command.yaw_rate * 0.032

        if controller.check_arrival(current, target):
            print(f"到达目标！步数: {step}")
            break

        if step % 20 == 0:
            print(f"步数 {step}: 位置 ({current.x:.2f}, {current.y:.2f}), "
                  f"速度 ({command.vx:.2f}, {command.vy:.2f})")

    print("导航完成！")