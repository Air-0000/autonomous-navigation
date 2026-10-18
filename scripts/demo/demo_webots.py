"""
主演示脚本
整合所有模块进行端到端导航演示
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple
import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.perception.detector import YOLODetector
from src.perception.depth_estimator import DepthEstimator
from src.perception.segmentor import DeepLabSegmentor
from src.perception.fusion import PerceptionFusion, Obstacle
from src.cognition.cognitive_map import CognitiveMap
from src.decision.planner import HybridPlanner
from src.decision.predictor import TrajectoryPredictor
from src.decision.controller import NavigationController, Pose
from src.interface.simulator import WebotsSimulator, SimulationRecorder


class NavigationSystem:
    """完整导航系统"""

    def __init__(self, config_path: Optional[str] = None, headless: bool = False):
        # 配置目录
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_dir = os.path.join(repo_root, 'config')

        # 初始化感知模块
        print("初始化感知模块...")
        self.detector = YOLODetector(
            os.path.join(config_dir, 'detection.yaml')
            if config_path else None
        )
        self.depth_estimator = DepthEstimator(
            os.path.join(config_dir, 'depth.yaml')
            if config_path else None
        )
        self.segmentor = DeepLabSegmentor(
            os.path.join(config_dir, 'segmentation.yaml')
            if config_path else None
        )
        self.fusion = PerceptionFusion()

        # 初始化认知模块（类脑导航）
        print("初始化类脑导航模块...")
        self.cognitive_map = CognitiveMap()

        # 初始化决策模块
        print("初始化决策模块...")
        self.planner = HybridPlanner()
        self.predictor = TrajectoryPredictor()
        self.controller = NavigationController()

        # 初始化仿真接口
        print("初始化仿真接口...")
        self.simulator = WebotsSimulator()
        self.simulator.connect()

        # 记录器
        self.recorder = SimulationRecorder()

        # 目标位置
        self.goal = (5.0, 5.0)

        # 可视化窗口 (支持 headless 运行)
        self.headless = headless or not os.environ.get("DISPLAY")
        self.window_name = "Navigation Demo"
        if not self.headless:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            except Exception:
                self.headless = True

    def run(self, max_steps: int = 500):
        """
        运行导航系统

        Args:
            max_steps: 最大步数
        """
        print(f"\n{'='*50}")
        print("开始导航演示")
        print(f"{'='*50}\n")

        step = 0
        success = False

        while step < max_steps:
            # 仿真步进
            timestep = self.simulator.step()
            if timestep == -1:
                print("仿真结束")
                break

            # 1. 获取传感器数据
            image = self.simulator.get_camera_image()
            pose = self.simulator.get_pose()

            # 2. 感知
            detections = self.detector.detect(image)
            depth, depth_norm = self.depth_estimator.estimate(image)
            segmentation, _ = self.segmentor.segment(image)

            # 3. 感知融合
            obstacles = self.fusion.fuse(
                detections, depth, segmentation,
                ego_pose=pose[:3]
            )

            # 4. 类脑导航更新
            current_pos = (pose[0], pose[2])  # (x, z)
            velocity = self.simulator.get_velocity()
            self.cognitive_map.update(
                location=current_pos,
                velocity=(velocity[0], velocity[1]),
                obstacles=obstacles
            )

            # 5. 轨迹预测
            trajectory_predictions = self.predictor.update(obstacles)

            # 6. 路径规划
            grid = self.fusion.build_occupancy_grid(obstacles)
            start_pos = (pose[0], pose[2])
            plan_result = self.planner.plan(
                start_pos, self.goal, grid, obstacles
            )

            # 7. 控制器计算
            current_pose = Pose(
                x=pose[0], y=pose[2], z=pose[1], yaw=pose[3]
            )

            if plan_result['success'] and plan_result['path']:
                command = self.controller.compute_command_from_path(
                    current_pose, plan_result['path']
                )
            else:
                command = self.controller.compute_command(
                    current_pose,
                    Pose(x=self.goal[0], y=self.goal[1],
                        z=0, yaw=0)
                )

            # 8. 执行控制
            self.simulator.set_velocity(
                command.vx, command.vy, command.vz, command.yaw_rate
            )

            # 9. 记录数据
            self.recorder.record_state(pose, detections, plan_result['path'])

            # 10. 检查是否到达目标
            dist_to_goal = np.sqrt(
                (pose[0] - self.goal[0])**2 +
                (pose[2] - self.goal[1])**2
            )

            if dist_to_goal < 0.5:
                success = True
                print(f"\n成功到达目标！步数: {step}, 距离: {dist_to_goal:.2f}m")
                break

            # 11. 可视化
            self.visualize(
                image, depth_norm, segmentation,
                detections, obstacles, plan_result['path'],
                current_pos, self.goal, step
            )

            step += 1

            # 每50步打印状态
            if step % 50 == 0:
                print(f"步骤 {step}: 位置 ({pose[0]:.2f}, {pose[2]:.2f}), "
                      f"距离目标 {dist_to_goal:.2f}m, "
                      f"检测到 {len(detections)} 个目标")

        # 结束记录
        self.recorder.finalize(success)
        self._print_summary()

        # 等待按键后关闭 (仅非 headless 模式)
        if not self.headless:
            print("\n按任意键关闭...")
            try:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception:
                pass
        self.simulator.close()

    def visualize(self, image: np.ndarray,
                  depth: np.ndarray,
                  segmentation: np.ndarray,
                  detections: List,
                  obstacles: List,
                  path: List,
                  current_pos: Tuple[float, float],
                  goal: Tuple[float, float],
                  step: int):
        """可视化所有感知和决策结果"""

        # 创建综合可视化图
        vis_images = []

        # 原始图像 + 检测框
        det_vis = self.detector.visualize(image.copy(), detections)
        det_vis = cv2.resize(det_vis, (320, 240))
        vis_images.append(('Detection', det_vis))

        # 深度图
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        depth_colored = cv2.resize(depth_colored, (320, 240))
        vis_images.append(('Depth', depth_colored))

        # 分割图
        seg_vis = self.segmentor.overlay_segmentation(image, segmentation)
        seg_vis = cv2.resize(seg_vis, (320, 240))
        vis_images.append(('Segmentation', seg_vis))

        # 认知地图
        cognitive_vis = self.cognitive_map.visualize()
        cognitive_vis = cv2.resize(cognitive_vis, (320, 240))
        vis_images.append(('Cognitive Map', cognitive_vis))

        # 轨迹预测可视化
        traj_vis = self.predictor.visualize_trajectories(image.shape[:2])
        traj_vis = cv2.resize(traj_vis, (320, 240))
        vis_images.append(('Trajectory', traj_vis))

        # 创建主视图
        main_vis = np.zeros((480, 640, 3), dtype=np.uint8)

        # 左上：检测结果
        main_vis[0:240, 0:320] = det_vis

        # 右上：深度图
        main_vis[0:240, 320:640] = depth_colored

        # 左下：分割图
        main_vis[240:480, 0:320] = seg_vis

        # 右下：认知地图
        main_vis[240:480, 320:640] = cognitive_vis

        # 路径可视化（在主视图上叠加）
        if path:
            for i in range(len(path) - 1):
                p1 = (int(path[i][0] * 50 + 320), int(path[i][1] * 50 + 240))
                p2 = (int(path[i+1][0] * 50 + 320), int(path[i+1][1] * 50 + 240))
                cv2.line(main_vis, p1, p2, (0, 255, 255), 2)

            # 起点
            cv2.circle(main_vis, (320, 240), 5, (0, 255, 0), -1)
            cv2.putText(main_vis, "Start", (325, 245),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 终点
            goal_point = (int(goal[0] * 50 + 320), int(goal[1] * 50 + 240))
            cv2.circle(main_vis, goal_point, 5, (0, 0, 255), -1)
            cv2.putText(main_vis, "Goal", (goal_point[0] + 5, goal_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 添加信息栏
        info = f"Step: {step} | Obstacles: {len(obstacles)} | "
        info += f"Path: {len(path) if path else 0} pts | "
        info += f"Place Cells Active: {int(self.cognitive_map.activity_map.max() * 100)}%"

        cv2.rectangle(main_vis, (0, 470), (640, 480), (0, 0, 0), -1)
        cv2.putText(main_vis, info, (10, 475),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 显示
        if not self.headless:
            try:
                cv2.imshow(self.window_name, main_vis)
                cv2.waitKey(1)
            except Exception:
                self.headless = True

    def _print_summary(self):
        """打印统计摘要"""
        summary = self.recorder.get_summary()

        print(f"\n{'='*50}")
        print("导航演示统计")
        print(f"{'='*50}")
        print(f"总步数: {summary['total_steps']}")
        print(f"总检测数: {summary['total_detections']}")
        print(f"路径长度: {summary['path_length']:.2f}m")
        print(f"碰撞次数: {summary['collision_count']}")
        print(f"成功: {'是' if summary['success'] else '否'}")
        print(f"时长: {summary['duration']:.2f}s")
        print(f"{'='*50}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='无人系统智能导航演示')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='最大仿真步数')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--headless', action='store_true',
                       help='以无界面 headless 模式运行')

    args = parser.parse_args()

    try:
        # 创建并运行系统
        system = NavigationSystem(args.config, headless=args.headless)
        system.run(max_steps=args.max_steps)

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()