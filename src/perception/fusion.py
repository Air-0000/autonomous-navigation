"""
感知融合模块
融合目标检测、深度估计、语义分割结果，构建环境感知图
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Obstacle:
    """障碍物数据结构"""
    class_name: str
    bbox: List[float]  # [x1, y1, x2, y2]
    center_2d: List[float]  # [cx, cy]
    center_3d: List[float]  # [x, y, z] 世界坐标
    distance: float  # 距离（米）
    velocity: List[float] = None  # [vx, vy]
    trajectory: List[List[float]] = None  # 历史轨迹
    is_dynamic: bool = False


class PerceptionFusion:
    """感知融合器"""

    def __init__(self, config_path: Optional[str] = None):
        # 相机内参（默认）
        self.fx = 500.0
        self.fy = 500.0
        self.cx = 320.0
        self.cy = 240.0
        self.depth_scale = 0.001  # 深度图缩放因子

        # 障碍物跟踪
        self.tracked_objects: Dict[int, Obstacle] = {}
        self.next_id = 0

    def fuse(self,
             detections: List[Dict],
             depth: np.ndarray,
             segmentation: np.ndarray,
             ego_pose: Tuple[float, float, float] = (0, 0, 0)) -> List[Obstacle]:
        """
        融合感知结果

        Args:
            detections: 目标检测结果
            depth: 深度图
            segmentation: 语义分割掩码
            ego_pose: 自车位置 (x, y, z)

        Returns:
            obstacles: 融合后的障碍物列表
        """
        obstacles = []

        for det in detections:
            # 从深度图获取该目标的距离
            bbox = det['bbox']
            center_2d = det['center'][:2]  # [cx, cy]

            # 计算3D位置
            distance = self._get_depth_at_point(depth, int(center_2d[0]), int(center_2d[1]))
            center_3d = self._pixel_to_3d(center_2d, distance)

            # 转换到世界坐标系（简化版）
            world_x = ego_pose[0] + center_3d[0]
            world_y = ego_pose[2]  # Y轴是高度
            world_z = ego_pose[1] + center_3d[2]

            obstacle = Obstacle(
                class_name=det['class_name'],
                bbox=bbox,
                center_2d=center_2d,
                center_3d=[world_x, world_y, world_z],
                distance=distance
            )

            # 判断是否动态障碍物
            obstacle.is_dynamic = self._is_dynamic_class(det['class_name'])

            obstacles.append(obstacle)

        return obstacles

    def _get_depth_at_point(self, depth: np.ndarray, x: int, y: int) -> float:
        """获取特定像素点的深度"""
        h, w = depth.shape
        if 0 <= y < h and 0 <= x < w:
            return depth[y, x] * self.depth_scale
        return 10.0  # 默认10米

    def _pixel_to_3d(self, pixel: np.ndarray, depth: float) -> np.ndarray:
        """像素坐标转相机坐标系3D坐标"""
        cx, cy = pixel
        x = (cx - self.cx) * depth / self.fx
        y = (cy - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z])

    def _is_dynamic_class(self, class_name: str) -> bool:
        """判断是否为动态障碍物类别"""
        dynamic_classes = ['person', 'bicycle', 'motorcycle', 'car', 'bus', 'truck']
        return class_name.lower() in dynamic_classes

    def track_obstacles(self, obstacles: List[Obstacle],
                       max_distance: float = 2.0) -> List[Obstacle]:
        """
        跟踪障碍物（简化版IoU匹配）

        Args:
            obstacles: 当前帧检测到的障碍物
            max_distance: 匹配最大距离阈值

        Returns:
            tracked: 带有跟踪ID的障碍物列表
        """
        for obs in obstacles:
            # 查找最接近的历史障碍物
            best_match = None
            best_dist = float('inf')

            for obj_id, prev_obs in self.tracked_objects.items():
                # 计算2D中心距离
                dist = np.linalg.norm(
                    np.array(obs.center_2d) - np.array(prev_obs.center_2d)
                )

                if dist < best_dist and dist < max_distance:
                    best_dist = dist
                    best_match = obj_id

            if best_match is not None:
                # 更新已有障碍物
                prev = self.tracked_objects[best_match]
                obs.velocity = [
                    obs.center_2d[0] - prev.center_2d[0],
                    obs.center_2d[1] - prev.center_2d[1]
                ]

                # 更新轨迹
                if prev.trajectory is None:
                    prev.trajectory = []
                prev.trajectory.append(obs.center_2d)
                if len(prev.trajectory) > 10:
                    prev.trajectory = prev.trajectory[-10:]
                obs.trajectory = prev.trajectory

                # 更新跟踪记录
                obs.class_name = prev.class_name
                self.tracked_objects[best_match] = obs
            else:
                # 新增障碍物
                self.tracked_objects[self.next_id] = obs
                self.next_id += 1

        return list(self.tracked_objects.values())

    def build_occupancy_grid(self, obstacles: List[Obstacle],
                           grid_size: float = 0.5,
                           grid_resolution: int = 100) -> np.ndarray:
        """
        构建占用栅格地图

        Args:
            obstacles: 障碍物列表
            grid_size: 栅格总大小（米）
            grid_resolution: 栅格分辨率

        Returns:
            grid: 占用栅格 [resolution, resolution]
                  1=占用, 0=空闲, -1=未知
        """
        grid = np.full((grid_resolution, grid_resolution), -1)

        cell_size = grid_size / grid_resolution
        center = grid_resolution // 2

        for obs in obstacles:
            cx, cy, cz = obs.center_3d

            # 转换为栅格坐标
            gx = int(center + cx / cell_size)
            gy = int(center + cy / cell_size)

            # 标记为占用（考虑大小）
            radius = max(obs.bbox[2] - obs.bbox[0],
                        obs.bbox[3] - obs.bbox[1]) / 100
            radius_cells = int(radius / cell_size) + 1

            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < grid_resolution and 0 <= ny < grid_resolution:
                        grid[ny, nx] = 1

        # 标记未占用区域
        unknown_mask = grid == -1
        grid[unknown_mask] = 0

        return grid

    def get_nearest_obstacle(self, obstacles: List[Obstacle]) -> Optional[Obstacle]:
        """获取最近的障碍物"""
        if not obstacles:
            return None

        min_dist = float('inf')
        nearest = None

        for obs in obstacles:
            if obs.distance < min_dist:
                min_dist = obs.distance
                nearest = obs

        return nearest

    def get_obstacles_by_distance(self, obstacles: List[Obstacle],
                                  max_distance: float = 10.0) -> List[Obstacle]:
        """筛选指定距离范围内的障碍物"""
        return [obs for obs in obstacles if obs.distance <= max_distance]


# 单元测试
if __name__ == "__main__":
    fusion = PerceptionFusion()

    # 模拟检测结果
    detections = [
        {'class_name': 'person', 'bbox': [100, 100, 150, 200],
         'center': [125, 150]},
        {'class_name': 'car', 'bbox': [300, 150, 450, 280],
         'center': [375, 215]}
    ]

    # 模拟深度图
    depth = np.ones((480, 640)) * 5.0  # 5米远的平面

    # 模拟分割图
    segmentation = np.zeros((480, 640), dtype=np.uint8)

    # 融合
    obstacles = fusion.fuse(detections, depth, segmentation)

    print(f"融合到 {len(obstacles)} 个障碍物")
    for obs in obstacles:
        print(f"  - {obs.class_name}: 距离 {obs.distance:.1f}m, "
              f"位置 ({obs.center_3d[0]:.1f}, {obs.center_3d[2]:.1f})")