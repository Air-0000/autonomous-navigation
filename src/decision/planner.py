"""
路径规划模块 - A* + RRT*
融合全局路径规划和局部避障
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import random


@dataclass
class Node:
    """搜索节点"""
    x: float
    y: float
    g_cost: float = 0.0  # 从起点到该点的代价
    h_cost: float = 0.0  # 从该点到终点的启发代价
    parent: 'Node' = None

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def __lt__(self, other: 'Node') -> bool:
        """支持heapq比较"""
        return self.f_cost < other.f_cost


class AStarPlanner:
    """A*全局路径规划器"""

    def __init__(self, grid_resolution: float = 0.1,
                 diagonal_cost: float = 1.414):
        self.grid_resolution = grid_resolution
        self.diagonal_cost = diagonal_cost

    def plan(self, start: Tuple[float, float],
            goal: Tuple[float, float],
            occupancy_grid: np.ndarray) -> List[Tuple[float, float]]:
        """
        A*路径规划

        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            occupancy_grid: 占用栅格 0=空闲, 1=占用

        Returns:
            path: 路径点列表 [(x1,y1), (x2,y2), ...]
        """
        # 获取栅格尺寸
        grid_h, grid_w = occupancy_grid.shape

        # 转换为栅格坐标
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        # 边界检查
        if not self._is_valid_grid(start_grid, grid_h, grid_w):
            start_grid = self._clamp_grid(start_grid, grid_h, grid_w)
        if not self._is_valid_grid(goal_grid, grid_h, grid_w):
            goal_grid = self._clamp_grid(goal_grid, grid_h, grid_w)

        # 如果起点或终点被占用，找最近的空闲点
        if occupancy_grid[start_grid] == 1:
            start_grid = self._find_nearest_free(start_grid, occupancy_grid)
        if occupancy_grid[goal_grid] == 1:
            goal_grid = self._find_nearest_free(goal_grid, occupancy_grid)

        # A*搜索
        open_set = []
        closed_set = set()
        node_map = {}

        start_node = Node(x=start_grid[0], y=start_grid[1])
        start_node.h_cost = self._heuristic(start_grid, goal_grid)
        heapq.heappush(open_set, (start_node.f_cost, start_node))

        node_map[(start_grid[0], start_grid[1])] = start_node

        # 8个方向的移动
        directions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, self.diagonal_cost), (1, -1, self.diagonal_cost),
            (-1, 1, self.diagonal_cost), (-1, -1, self.diagonal_cost)
        ]

        while open_set:
            _, current = heapq.heappop(open_set)
            current_key = (int(current.x), int(current.y))

            if current_key in closed_set:
                continue

            closed_set.add(current_key)

            # 到达目标
            if current_key == (int(goal_grid[0]), int(goal_grid[1])):
                return self._reconstruct_path(current, start, goal)

            # 扩展邻居
            for dx, dy, cost in directions:
                nx, ny = int(current.x + dx), int(current.y + dy)

                if not self._is_valid_grid((nx, ny), grid_h, grid_w):
                    continue

                if occupancy_grid[ny, nx] == 1:
                    continue

                neighbor_key = (nx, ny)
                if neighbor_key in closed_set:
                    continue

                # 计算新代价
                g_cost = current.g_cost + cost
                h_cost = self._heuristic((nx, ny), goal_grid)
                f_cost = g_cost + h_cost

                # 检查是否已在open_set中且有更优路径
                if neighbor_key in node_map:
                    existing = node_map[neighbor_key]
                    if g_cost < existing.g_cost:
                        existing.g_cost = g_cost
                        existing.parent = current
                        heapq.heappush(open_set, (f_cost, existing))
                else:
                    neighbor = Node(x=nx, y=ny)
                    neighbor.g_cost = g_cost
                    neighbor.h_cost = h_cost
                    neighbor.parent = current
                    node_map[neighbor_key] = neighbor
                    heapq.heappush(open_set, (f_cost, neighbor))

        # 未找到路径
        return []

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """启发函数（欧几里得距离）"""
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _world_to_grid(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        return (int(pos[0] / self.grid_resolution),
                int(pos[1] / self.grid_resolution))

    def _is_valid_grid(self, grid_pos: Tuple[int, int],
                       grid_h: int, grid_w: int) -> bool:
        """检查栅格坐标是否有效"""
        return 0 <= grid_pos[0] < grid_w and 0 <= grid_pos[1] < grid_h

    def _clamp_grid(self, grid_pos: Tuple[int, int],
                   grid_h: int, grid_w: int) -> Tuple[int, int]:
        """限制栅格坐标在有效范围内"""
        return (
            max(0, min(grid_w - 1, grid_pos[0])),
            max(0, min(grid_h - 1, grid_pos[1]))
        )

    def _find_nearest_free(self, grid_pos: Tuple[int, int],
                          occupancy_grid: np.ndarray) -> Tuple[int, int]:
        """找最近的空闲栅格"""
        grid_h, grid_w = occupancy_grid.shape

        for r in range(1, max(grid_h, grid_w)):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = grid_pos[0] + dx, grid_pos[1] + dy
                    if self._is_valid_grid((nx, ny), grid_h, grid_w):
                        if occupancy_grid[ny, nx] == 0:
                            return (nx, ny)

        return grid_pos

    def _reconstruct_path(self, node: Node,
                          start: Tuple[float, float],
                          goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """重构路径"""
        path = []
        current = node

        while current is not None:
            # 转换回世界坐标（取栅格中心）
            world_x = (current.x + 0.5) * self.grid_resolution
            world_y = (current.y + 0.5) * self.grid_resolution
            path.append((world_x, world_y))
            current = current.parent

        path.reverse()

        # 替换起点和终点
        if path:
            path[0] = start
            path[-1] = goal

        return path


class RRTStarPlanner:
    """RRT*局部规划器（增量式重规划）"""

    def __init__(self, max_iterations: int = 500,
                 step_size: float = 0.1,
                 goal_bias: float = 0.1,
                 search_radius: float = 0.3):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.search_radius = search_radius

        # 树节点
        self.nodes: List[Tuple[float, float]] = []

    @dataclass
    class TreeNode:
        x: float
        y: float
        parent: Optional['TreeNode']
        cost: float = 0.0

    def plan(self, start: Tuple[float, float],
             goal: Tuple[float, float],
             obstacles: List,  # 障碍物列表
             bounds: Tuple[float, float, float, float] = (-10, -10, 10, 10)
             ) -> List[Tuple[float, float]]:
        """
        RRT*路径规划

        Args:
            start: 起点
            goal: 目标点
            obstacles: 障碍物列表（每个障碍物有center_3d和distance）
            bounds: 搜索边界 (xmin, ymin, xmax, ymax)

        Returns:
            path: 路径点列表
        """
        xmin, ymin, xmax, ymax = bounds

        # 初始化
        self.nodes = []
        root = self.TreeNode(x=start[0], y=start[1], parent=None, cost=0.0)
        self.nodes.append(root)

        for _ in range(self.max_iterations):
            # 采样
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = (random.uniform(xmin, xmax),
                         random.uniform(ymin, ymax))

            # 找到最近的节点
            nearest = self._find_nearest(sample)

            # 扩展
            new_point = self._steer(nearest, sample)

            # 碰撞检测
            if not self._check_collision(new_point, obstacles):
                # 重连（找邻居）
                neighbors = self._find_neighbors(new_point)

                # 选择代价最小的父节点
                min_cost = nearest.cost + self._distance(nearest, new_point)
                new_parent = nearest

                for neighbor in neighbors:
                    cost = neighbor.cost + self._distance(neighbor, new_point)
                    if cost < min_cost:
                        min_cost = cost
                        new_parent = neighbor

                new_node = self.TreeNode(
                    x=new_point[0],
                    y=new_point[1],
                    parent=new_parent,
                    cost=min_cost
                )
                self.nodes.append(new_node)

                # 重连邻居
                for neighbor in neighbors:
                    new_cost = new_node.cost + self._distance(new_node, neighbor)
                    if new_cost < neighbor.cost:
                        neighbor.parent = new_node
                        neighbor.cost = new_cost

                # 检查是否到达目标
                if self._distance(new_point, goal) < self.step_size:
                    goal_node = self.TreeNode(
                        x=goal[0], y=goal[1],
                        parent=new_node,
                        cost=new_node.cost + self._distance(new_point, goal)
                    )
                    self.nodes.append(goal_node)
                    return self._get_path(goal_node)

        # 返回最近的路径
        closest = min(self.nodes, key=lambda n: self._distance((n.x, n.y), goal))
        return self._get_path(closest)

    def _find_nearest(self, point: Tuple[float, float]) -> 'TreeNode':
        """找最近的树节点"""
        return min(self.nodes,
                   key=lambda n: self._distance((n.x, n.y), point))

    def _find_neighbors(self, point: Tuple[float, float],
                       radius: float = None) -> List['TreeNode']:
        """找邻居节点"""
        if radius is None:
            radius = self.search_radius

        neighbors = []
        for node in self.nodes:
            if self._distance((node.x, node.y), point) < radius:
                neighbors.append(node)
        return neighbors

    def _steer(self, from_node: 'TreeNode',
               to_point: Tuple[float, float]) -> Tuple[float, float]:
        """从节点向目标点扩展"""
        from_point = (from_node.x, from_node.y)
        dist = self._distance(from_point, to_point)

        if dist < self.step_size:
            return to_point

        # 归一化方向
        dx = (to_point[0] - from_point[0]) / dist
        dy = (to_point[1] - from_point[1]) / dist

        return (
            from_point[0] + dx * self.step_size,
            from_point[1] + dy * self.step_size
        )

    def _check_collision(self, point: Tuple[float, float],
                         obstacles: List) -> bool:
        """碰撞检测"""
        if not obstacles:
            return False

        for obs in obstacles:
            if len(obs.center_3d) >= 2:
                obs_x = obs.center_3d[0]
                obs_y = obs.center_3d[2]  # Y是高度，Z是前后方向
                obs_dist = np.sqrt((point[0] - obs_x) ** 2 +
                                  (point[1] - obs_y) ** 2)

                # 障碍物安全距离
                safety_dist = 0.5 + obs.distance * 0.1

                if obs_dist < safety_dist:
                    return True

        return False

    def _distance(self, p1, p2) -> float:
        """欧几里得距离 - 支持元组和TreeNode"""
        # 处理TreeNode对象
        if hasattr(p1, 'x'):
            x1, y1 = p1.x, p1.y
        else:
            x1, y1 = p1[0], p1[1]

        if hasattr(p2, 'x'):
            x2, y2 = p2.x, p2.y
        else:
            x2, y2 = p2[0], p2[1]

        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _get_path(self, goal_node: 'TreeNode') -> List[Tuple[float, float]]:
        """从目标节点回溯路径"""
        path = []
        current = goal_node

        while current is not None:
            path.append((current.x, current.y))
            current = current.parent

        path.reverse()
        return path


class HybridPlanner:
    """混合规划器 - 全局A* + 局部RRT*"""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {
                'global': {'grid_resolution': 0.1, 'diagonal_cost': 1.414},
                'local': {'max_iterations': 200, 'step_size': 0.1}
            }

        self.global_planner = AStarPlanner(**config['global'])
        self.local_planner = RRTStarPlanner(**config['local'])

    def plan(self, start: Tuple[float, float],
             goal: Tuple[float, float],
             occupancy_grid: np.ndarray,
             obstacles: List = None) -> Dict:
        """
        综合路径规划

        Returns:
            {
                'path': [(x,y), ...],           # 全局路径
                'local_path': [(x,y), ...],      # 局部优化路径
                'success': bool,
                'path_length': float
            }
        """
        if obstacles is None:
            obstacles = []

        # 全局规划（A*）
        global_path = self.global_planner.plan(start, goal, occupancy_grid)

        if not global_path:
            # A*失败，用RRT*作为备选
            local_path = self.local_planner.plan(start, goal, obstacles)
            return {
                'path': local_path,
                'local_path': local_path,
                'success': len(local_path) > 0,
                'path_length': self._calc_path_length(local_path)
            }

        # 局部优化（RRT*）
        if len(global_path) > 2:
            # 在全局路径的中间点进行局部RRT*优化
            mid_idx = len(global_path) // 2
            mid_point = global_path[mid_idx]

            local_path = self.local_planner.plan(mid_point, goal, obstacles)

            if local_path:
                # 合并路径
                optimized_path = global_path[:mid_idx] + local_path
            else:
                optimized_path = global_path
        else:
            optimized_path = global_path

        return {
            'path': optimized_path,
            'local_path': global_path,
            'success': True,
            'path_length': self._calc_path_length(optimized_path)
        }

    def _calc_path_length(self, path: List[Tuple[float, float]]) -> float:
        """计算路径长度"""
        if len(path) < 2:
            return 0.0

        length = 0.0
        for i in range(1, len(path)):
            length += np.sqrt(
                (path[i][0] - path[i-1][0]) ** 2 +
                (path[i][1] - path[i-1][1]) ** 2
            )
        return length


# 单元测试
if __name__ == "__main__":
    # 创建混合规划器
    planner = HybridPlanner()

    # 创建简单占用栅格（中心有障碍）
    grid = np.zeros((100, 100), dtype=np.uint8)
    grid[40:60, 40:60] = 1  # 中心障碍

    # 规划路径
    start = (1.0, 1.0)
    goal = (9.0, 9.0)

    result = planner.plan(start, goal, grid)

    print(f"规划成功: {result['success']}")
    print(f"路径长度: {result['path_length']:.2f}米")
    print(f"路径点数: {len(result['path'])}")

    if result['path']:
        print(f"起点: {result['path'][0]}")
        print(f"终点: {result['path'][-1]}")