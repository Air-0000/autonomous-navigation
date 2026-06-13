"""
轨迹预测模块
基于社会力模型和恒速模型的轨迹预测
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    position: Tuple[float, float]  # (x, y)
    velocity: Tuple[float, float]  # (vx, vy)
    timestamp: float


class TrajectoryPredictor:
    """轨迹预测器"""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {
                'history_length': 10,
                'prediction_horizon': 5,
            }

        self.history_length = config['history_length']
        self.prediction_horizon = config['prediction_horizon']

        # 存储历史轨迹
        self.trajectories: Dict[int, List[TrajectoryPoint]] = {}
        self.next_id = 0

    def update(self, obstacles: List) -> Dict[int, List[Tuple[float, float]]]:
        """
        更新轨迹并预测未来轨迹

        Args:
            obstacles: 障碍物列表（包含center_2d, velocity, trajectory）

        Returns:
            predictions: {obstacle_id: [(x,y), ...]} 预测轨迹
        """
        predictions = {}

        for i, obs in enumerate(obstacles):
            # 分配或查找ID
            obs_id = self._get_or_create_id(obs, i)

            # 更新轨迹
            self._update_trajectory(obs_id, obs)

            # 预测未来轨迹
            predictions[obs_id] = self.predict(obs_id)

        return predictions

    def _get_or_create_id(self, obs, fallback_id: int) -> int:
        """为障碍物分配或查找ID"""
        # 简化版：基于位置匹配
        threshold = 50  # 像素距离阈值

        best_match = None
        best_dist = float('inf')

        for obs_id, traj in self.trajectories.items():
            if traj:
                last_pos = traj[-1].position
                curr_pos = obs.center_2d

                dist = np.linalg.norm(
                    np.array(last_pos) - np.array(curr_pos)
                )

                if dist < threshold and dist < best_dist:
                    best_match = obs_id
                    best_dist = dist

        return best_match if best_match is not None else fallback_id

    def _update_trajectory(self, obs_id: int, obs):
        """更新轨迹"""
        if obs_id not in self.trajectories:
            self.trajectories[obs_id] = []

        # 获取速度
        velocity = (0.0, 0.0)
        if hasattr(obs, 'velocity') and obs.velocity:
            velocity = tuple(obs.velocity[:2])
        else:
            traj = self.trajectories[obs_id]
            if len(traj) >= 2:
                # 从历史轨迹计算速度
                p1 = traj[-2].position
                p2 = traj[-1].position
                velocity = (p2[0] - p1[0], p2[1] - p1[1])

        # 添加新点
        point = TrajectoryPoint(
            position=tuple(obs.center_2d[:2]),
            velocity=velocity,
            timestamp=0.0  # 简化，不使用真实时间戳
        )

        self.trajectories[obs_id].append(point)

        # 限制历史长度
        if len(self.trajectories[obs_id]) > self.history_length:
            self.trajectories[obs_id] = \
                self.trajectories[obs_id][-self.history_length:]

    def predict(self, obs_id: int,
                method: str = 'constant_velocity') -> List[Tuple[float, float]]:
        """
        预测未来轨迹

        Args:
            obs_id: 障碍物ID
            method: 预测方法
                - 'constant_velocity': 恒速模型
                - 'linear_regression': 线性回归
                - 'social_force': 社会力模型（简化版）

        Returns:
            predictions: 预测的轨迹点 [(x,y), ...]
        """
        if obs_id not in self.trajectories:
            return []

        traj = self.trajectories[obs_id]

        if len(traj) < 2:
            if len(traj) == 1:
                return [traj[0].position]
            return []

        if method == 'constant_velocity':
            return self._predict_constant_velocity(traj)
        elif method == 'linear_regression':
            return self._predict_linear_regression(traj)
        else:
            return self._predict_constant_velocity(traj)

    def _predict_constant_velocity(self, traj: List[TrajectoryPoint]) -> List[Tuple[float, float]]:
        """恒速模型预测"""
        # 计算平均速度
        avg_vx = 0.0
        avg_vy = 0.0

        for i in range(1, len(traj)):
            avg_vx += traj[i].velocity[0]
            avg_vy += traj[i].velocity[1]

        if len(traj) > 1:
            avg_vx /= (len(traj) - 1)
            avg_vy /= (len(traj) - 1)

        # 预测
        predictions = []
        last_pos = traj[-1].position

        for t in range(1, self.prediction_horizon + 1):
            new_x = last_pos[0] + avg_vx * t
            new_y = last_pos[1] + avg_vy * t
            predictions.append((new_x, new_y))

        return predictions

    def _predict_linear_regression(self, traj: List[TrajectoryPoint]) -> List[Tuple[float, float]]:
        """线性回归预测"""
        n = len(traj)
        if n < 3:
            return self._predict_constant_velocity(traj)

        # 获取x, y序列
        x_vals = [p.position[0] for p in traj]
        y_vals = [p.position[1] for p in traj]
        t_vals = list(range(n))

        # 线性回归
        x_mean = np.mean(x_vals)
        y_mean = np.mean(t_vals)

        # 简化：使用最后几个点的趋势
        recent_n = min(5, n)
        recent_x = x_vals[-recent_n:]
        recent_t = list(range(n - recent_n, n))

        # 计算趋势
        if recent_n > 1:
            slope_x = (recent_x[-1] - recent_x[0]) / recent_n
            slope_y = (recent_x[-1] - recent_x[0]) / recent_n  # 简化：复用

            # 预测
            predictions = []
            last_x, last_y = x_vals[-1], y_vals[-1]

            for t in range(1, self.prediction_horizon + 1):
                new_x = last_x + slope_x * t
                new_y = last_y + slope_y * t
                predictions.append((new_x, new_y))

            return predictions

        return self._predict_constant_velocity(traj)

    def predict_all_obstacles(self, obstacles: List) -> Dict:
        """
        预测所有障碍物的未来轨迹

        Returns:
            {
                'predictions': {id: [(x,y), ...]},
                'collision_risks': [(obs_id, risk_score), ...]
            }
        """
        predictions = self.update(obstacles)

        # 计算碰撞风险（简化版）
        collision_risks = []

        for obs_id, pred_path in predictions.items():
            if len(pred_path) >= 2:
                # 简化为路径上的位移量
                total_displacement = sum(
                    np.linalg.norm(
                        np.array(pred_path[i]) - np.array(pred_path[i-1])
                    )
                    for i in range(1, len(pred_path))
                )

                # 位移越大，风险越高
                collision_risks.append((obs_id, total_displacement))

        # 按风险排序
        collision_risks.sort(key=lambda x: x[1], reverse=True)

        return {
            'predictions': predictions,
            'collision_risks': collision_risks
        }

    def visualize_trajectories(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """生成轨迹可视化图"""
        try:
            import cv2
        except ImportError:
            return np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

        vis = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        for obs_id, traj in self.trajectories.items():
            color = colors[obs_id % len(colors)]

            # 绘制历史轨迹
            for i in range(1, len(traj)):
                pt1 = (int(traj[i-1].position[0]), int(traj[i-1].position[1]))
                pt2 = (int(traj[i].position[0]), int(traj[i].position[1]))
                cv2.line(vis, pt1, pt2, color, 2)

            # 绘制预测轨迹
            pred = self.predict(obs_id)
            if pred:
                for i in range(1, len(pred)):
                    pt1 = (int(pred[i-1][0]), int(pred[i-1][1]))
                    pt2 = (int(pred[i][0]), int(pred[i][1]))
                    cv2.line(vis, pt1, pt2, (128, 128, 128), 2)

                # 预测终点
                last_pred = pred[-1]
                cv2.circle(vis, (int(last_pred[0]), int(last_pred[1])),
                          5, (255, 255, 255), -1)

        return vis


# 单元测试
if __name__ == "__main__":
    predictor = TrajectoryPredictor()

    # 模拟障碍物
    class MockObstacle:
        def __init__(self, pos, vel=None):
            self.center_2d = list(pos)
            self.velocity = vel if vel else [0, 0]

    # 模拟匀速移动的障碍物
    obstacles = [
        MockObstacle((100, 100), [2, 1]),
        MockObstacle((300, 200), [1, 2]),
    ]

    # 更新和预测
    for t in range(10):
        for obs in obstacles:
            obs.center_2d[0] += obs.velocity[0]
            obs.center_2d[1] += obs.velocity[1]

        predictions = predictor.update(obstacles)

        print(f"时刻 {t}:")
        for obs_id, pred in predictions.items():
            if pred:
                print(f"  障碍物 {obs_id}: 预测终点 {pred[-1]}")

    # 生成可视化
    vis = predictor.visualize_trajectories((480, 640))
    print(f"\n可视化图形状: {vis.shape}")