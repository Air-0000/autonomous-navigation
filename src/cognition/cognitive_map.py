"""
类脑导航模块
模拟大脑导航系统：位置细胞(Place Cell)和网格细胞(Grid Cell)
基于2014年诺贝尔奖发现的神经科学原理
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter


@dataclass
class PlaceCell:
    """位置细胞"""
    position: Tuple[float, float]  # 世界坐标 (x, z)
    field_size: float = 2.0  # 感受野大小
    activation: float = 0.0  # 激活值 0-1


@dataclass
class PlaceCellActivity:
    """位置细胞活动状态"""
    cells: List[PlaceCell]
    activity_map: np.ndarray  # 活动地图 [grid_size, grid_size]
    peak_position: Tuple[float, float]  # 峰值位置（估计的当前位置）
    confidence: float  # 置信度


class GridCell:
    """网格细胞"""

    def __init__(self, position: Tuple[float, float], orientation: float,
                 spatial_scale: float):
        """
        Args:
            position: 位置 (x, z)
            orientation: 方向角度（弧度）
            spatial_scale: 空间尺度（网格间距）
        """
        self.position = position
        self.orientation = orientation
        self.spatial_scale = spatial_scale

    def get_activation(self, location: Tuple[float, float]) -> float:
        """
        计算在给定位置的激活值

        基于网格细胞的周期性地形图特性
        """
        dx = location[0] - self.position[0]
        dz = location[1] - self.position[1]

        # 沿主轴的距离
        cos_o = np.cos(self.orientation)
        sin_o = np.sin(self.orientation)
        d_along = dx * cos_o + dz * sin_o
        d_perp = -dx * sin_o + dz * cos_o

        # 六边形网格模式（简化为周期性模式）
        lambda_ = self.spatial_scale
        d_along_mod = np.mod(d_along + lambda_ / 2, lambda_) - lambda_ / 2
        d_perp_mod = np.mod(d_perp + lambda_ / 2, lambda_) - lambda_ / 2

        # 距离
        distance = np.sqrt(d_along_mod ** 2 + d_perp_mod ** 2)

        # 高斯激活
        sigma = lambda_ / 3
        activation = np.exp(-distance ** 2 / (2 * sigma ** 2))

        return activation


class CognitiveMap:
    """认知地图 - 整合位置细胞和网格细胞"""

    def __init__(self, config: Optional[Dict] = None):
        # 配置
        if config is None:
            config = {
                'place_cells': {
                    'field_size': 2.0,
                    'activation_threshold': 0.5,
                    'decay_rate': 0.95,
                },
                'grid_cells': {
                    'spatial_scale': 0.5,
                    'orientation_angles': [0, np.pi / 3, 2 * np.pi / 3],
                },
                'map_resolution': 0.2,
            }

        self.field_size = config['place_cells']['field_size']
        self.activation_threshold = config['place_cells']['activation_threshold']
        self.decay_rate = config['place_cells']['decay_rate']
        self.spatial_scale = config['grid_cells']['spatial_scale']
        self.orientation_angles = config['grid_cells']['orientation_angles']
        self.map_resolution = config['map_resolution']

        # 地图范围（米）
        self.map_size = 20.0  # 20x20米
        self.grid_resolution = int(self.map_size / self.map_resolution)

        # 初始化细胞
        self.place_cells: List[PlaceCell] = []
        self.grid_cells: List[GridCell] = []
        self._initialize_cells()

        # 当前状态
        self.current_position = (0.0, 0.0)
        self.visited_map = np.zeros((self.grid_resolution, self.grid_resolution))
        self.activity_map = np.zeros((self.grid_resolution, self.grid_resolution))

        # 路径整合相关
        self.velocity_integrated_position = (0.0, 0.0)

    def _initialize_cells(self):
        """初始化位置细胞和网格细胞"""
        # 创建位置细胞网格
        spacing = self.field_size / 2
        half_size = self.map_size / 2

        for x in np.arange(-half_size, half_size, spacing):
            for z in np.arange(-half_size, half_size, spacing):
                cell = PlaceCell(position=(x, z), field_size=self.field_size)
                self.place_cells.append(cell)

        # 创建网格细胞（三个方向）
        grid_spacing = self.spatial_scale
        n_grids = int(self.map_size / grid_spacing)

        for angle in self.orientation_angles:
            for i in range(n_grids):
                for j in range(n_grids):
                    x = -half_size + i * grid_spacing
                    z = -half_size + j * grid_spacing
                    cell = GridCell(
                        position=(x, z),
                        orientation=angle,
                        spatial_scale=self.spatial_scale
                    )
                    self.grid_cells.append(cell)

        print(f"初始化了 {len(self.place_cells)} 个位置细胞")
        print(f"初始化了 {len(self.grid_cells)} 个网格细胞")

    def update(self, location: Tuple[float, float],
               velocity: Optional[Tuple[float, float]] = None,
               obstacles: Optional[List] = None):
        """
        更新认知地图

        Args:
            location: 当前估计位置
            velocity: 当前速度 (vx, vz)
            obstacles: 感知到的障碍物列表
        """
        self.current_position = location

        # 更新位置细胞活动
        self._update_place_cells(location)

        # 更新网格细胞活动
        self._update_grid_cells(location)

        # 更新访问地图
        self._update_visited_map(location)

        # 路径整合
        if velocity is not None:
            self._path_integration(velocity)

        # 障碍物整合到地图
        if obstacles is not None:
            self._integrate_obstacles(obstacles)

    def _update_place_cells(self, location: Tuple[float, float]):
        """更新位置细胞激活"""
        self.activity_map.fill(0)

        for cell in self.place_cells:
            # 计算到该位置细胞中心的距离
            distance = np.sqrt(
                (location[0] - cell.position[0]) ** 2 +
                (location[1] - cell.position[1]) ** 2
            )

            # 高斯激活
            sigma = cell.field_size / 2
            activation = np.exp(-distance ** 2 / (2 * sigma ** 2))

            cell.activation = activation

            # 更新活动地图
            gx = int((cell.position[0] + self.map_size / 2) / self.map_resolution)
            gz = int((cell.position[1] + self.map_size / 2) / self.map_resolution)

            if 0 <= gx < self.grid_resolution and 0 <= gz < self.grid_resolution:
                self.activity_map[gz, gx] = max(
                    self.activity_map[gz, gx],
                    activation
                )

        # 平滑活动地图
        self.activity_map = gaussian_filter(self.activity_map, sigma=2)

    def _update_grid_cells(self, location: Tuple[float, float]):
        """更新网格细胞激活（用于方向感知）"""
        activations = []

        for cell in self.grid_cells:
            act = cell.get_activation(location)
            activations.append(act)

        return np.mean(activations) if activations else 0.0

    def _update_visited_map(self, location: Tuple[float, float]):
        """更新访问地图"""
        gx = int((location[0] + self.map_size / 2) / self.map_resolution)
        gz = int((location[1] + self.map_size / 2) / self.map_resolution)

        if 0 <= gx < self.grid_resolution and 0 <= gz < self.grid_resolution:
            # 增加访问次数
            self.visited_map[gz, gx] += 1

            # 衰减旧访问记录
            self.visited_map *= self.decay_rate

            # 更新当前位置的访问记录
            self.visited_map[gz, gx] = 1.0

    def _path_integration(self, velocity: Tuple, dt: float = 0.032):
        """
        路径整合 - 根据运动推算位置
        类似于大脑的内耳前庭系统和本体感觉
        """
        # 支持2维或3维速度
        if len(velocity) >= 2:
            vx, vz = velocity[0], velocity[1]
        else:
            vx, vz = velocity[0], 0.0

        self.velocity_integrated_position = (
            self.velocity_integrated_position[0] + vx * dt,
            self.velocity_integrated_position[1] + vz * dt
        )

    def _integrate_obstacles(self, obstacles: List):
        """将障碍物整合到认知地图"""
        for obs in obstacles:
            cx, cy, cz = obs.center_3d

            # 转换为地图坐标
            gx = int((cx + self.map_size / 2) / self.map_resolution)
            gz = int((cz + self.map_size / 2) / self.map_resolution)

            # 在一定范围内标记为占用
            radius = 5  # 5个格子
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    nx, nz = gx + dx, gz + dz
                    if 0 <= nx < self.grid_resolution and 0 <= nz < self.grid_resolution:
                        # 根据距离衰减
                        dist = np.sqrt(dx ** 2 + dz ** 2)
                        if dist <= radius:
                            self.visited_map[nz, nx] = min(
                                self.visited_map[nz, nx],
                                0.1
                            )

    def estimate_position(self) -> Tuple[float, float, float]:
        """
        基于位置细胞活动估计当前位置
        返回: (x, z, confidence)
        """
        # 找到活动最强的位置细胞
        max_activation = 0.0
        estimated_pos = (0.0, 0.0)

        for cell in self.place_cells:
            if cell.activation > max_activation:
                max_activation = cell.activation
                estimated_pos = cell.position

        return (*estimated_pos, max_activation)

    def get_place_cell_activity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取位置细胞活动数据
        返回: (x_coords, z_coords, activations)
        """
        x_coords = []
        z_coords = []
        activations = []

        for cell in self.place_cells:
            x_coords.append(cell.position[0])
            z_coords.append(cell.position[1])
            activations.append(cell.activation)

        return (
            np.array(x_coords),
            np.array(z_coords),
            np.array(activations)
        )

    def get_grid_cell_pattern(self, location: Tuple[float, float]) -> np.ndarray:
        """
        获取网格细胞模式（用于可视化）
        """
        pattern = np.zeros((100, 100))

        for cell in self.grid_cells[:1000]:  # 限制数量以加速
            gx = int((cell.position[0] + self.map_size / 2) / 0.2)
            gz = int((cell.position[1] + self.map_size / 2) / 0.2)

            if 0 <= gx < 100 and 0 <= gz < 100:
                pattern[gz, gx] = cell.get_activation(location)

        return pattern

    def visualize(self) -> np.ndarray:
        """
        可视化认知地图
        返回: 彩色RGB图像
        """
        import cv2

        # 创建RGB图像
        h, w = self.activity_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # 活动地图 -> 红色通道（位置细胞激活）
        activity_norm = (self.activity_map / (self.activity_map.max() + 1e-6) * 255).astype(np.uint8)
        rgb[:, :, 2] = activity_norm

        # 访问地图 -> 绿色通道（已访问区域）
        visited_norm = np.clip(self.visited_map * 200, 0, 255).astype(np.uint8)
        rgb[:, :, 1] = visited_norm

        # 当前估计位置 -> 黄色圆点
        cx = int((self.current_position[0] + self.map_size / 2) / self.map_resolution)
        cz = int((self.current_position[1] + self.map_size / 2) / self.map_resolution)

        # 绘制当前位置（黄色）
        if 0 <= cx < w and 0 <= cz < h:
            cv2.circle(rgb, (cx, cz), 8, (0, 255, 255), -1)  # 中心点
            cv2.circle(rgb, (cx, cz), 12, (0, 200, 200), 2)  # 外圈

        # 绘制路径整合位置（蓝色）
        vx = int((self.velocity_integrated_position[0] + self.map_size / 2) / self.map_resolution)
        vz = int((self.velocity_integrated_position[1] + self.map_size / 2) / self.map_resolution)
        if 0 <= vx < w and 0 <= vz < h:
            cv2.circle(rgb, (vx, vz), 5, (255, 0, 0), -1)

        # 添加网格线
        for i in range(0, h, 10):
            rgb[i, :] = [50, 50, 50]
        for j in range(0, w, 10):
            rgb[:, j] = [50, 50, 50]

        # 上采样显示
        rgb = cv2.resize(rgb, (500, 500))

        # 添加标签
        cv2.putText(rgb, f"Place Cell Activity: {int(self.activity_map.max() * 100)}%",
                   (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(rgb, f"Position: ({self.current_position[0]:.1f}, {self.current_position[1]:.1f})",
                   (10, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return rgb


# 单元测试
if __name__ == "__main__":
    config = {
        'place_cells': {
            'field_size': 2.0,
            'activation_threshold': 0.5,
            'decay_rate': 0.95,
        },
        'grid_cells': {
            'spatial_scale': 0.5,
            'orientation_angles': [0, np.pi / 3, 2 * np.pi / 3],
        },
        'map_resolution': 0.2,
    }

    # 创建认知地图
    cognitive_map = CognitiveMap(config)

    # 模拟移动
    locations = [(0, 0), (1, 1), (2, 1.5), (3, 1)]
    velocities = [(1, 1), (1, 0), (1, -0.5), (1, -0.3)]

    for loc, vel in zip(locations, velocities):
        cognitive_map.update(location=loc, velocity=vel)

        # 估计位置
        est_x, est_z, conf = cognitive_map.estimate_position()
        print(f"真实位置: {loc}, 估计位置: ({est_x:.2f}, {est_z:.2f}), 置信度: {conf:.2f}")

    print("\n认知地图更新完成！")
    print(f"活动地图范围: {cognitive_map.activity_map.min():.3f} - "
          f"{cognitive_map.activity_map.max():.3f}")