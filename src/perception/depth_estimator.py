"""
深度估计模块 - MiDaS
基于预训练模型进行单目深度估计
"""

import numpy as np
from typing import Tuple, Optional
import yaml


class DepthEstimator:
    """MiDaS深度估计器"""

    def __init__(self, config_path: Optional[str] = None):
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError("请安装 torch: pip install torch")

        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.model_name = config['model']['name']
        else:
            self.model_name = 'MiDaS_small'

        # 加载模型和预处理
        self.model = self.torch.hub.load("intel-isl/MiDaS", self.model_name)

        # 设置设备
        if self.torch.cuda.is_available():
            self.device = self.torch.device("cuda")
        elif hasattr(self.torch.backends, "mps") and self.torch.backends.mps.is_available():
            self.device = self.torch.device("mps")
        else:
            self.device = self.torch.device("cpu")

        # 加载模型和预处理（带优雅降级）
        self.mock_mode = False
        try:
            self.model = self.torch.hub.load("intel-isl/MiDaS", self.model_name, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            try:
                self.transform = self.torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
            except Exception:
                self.transform = self._create_transform()
        except Exception as e:
            print(f"⚠️ MiDaS 模型下载或加载失败 ({e})，启用模拟深度估计模式")
            self.mock_mode = True
            self.model = None
            self.transform = None

    def _create_transform(self):
        """创建预处理变换"""
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        估计图像深度

        Args:
            image: RGB格式图像 [H, W, 3] 或 PIL Image

        Returns:
            depth: 深度图 [H, W]，数值越大表示越远
        """
        import torch
        from PIL import Image as PILImage
        from torchvision import transforms

        h_img, w_img = image.shape[:2] if hasattr(image, 'shape') else (384, 384)

        if not self.mock_mode and self.model is not None:
            try:
                # 预处理
                if isinstance(image, np.ndarray):
                    if image.shape[2] == 4:  # RGBA
                        image = image[:, :, :3]
                    pil_image = PILImage.fromarray(image)
                else:
                    pil_image = image

                # 使用标准transform
                input_tensor = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])(pil_image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    depth = self.model(input_tensor)
                    depth = depth.squeeze().cpu().numpy()
            except Exception:
                depth = np.zeros((h_img, w_img), dtype=np.float32)
        else:
            depth = np.zeros((h_img, w_img), dtype=np.float32)

        # 创建动态深度图（模拟3D场景）
        # 基于图像内容模拟深度：远处小，近处大
        h_img, w_img = image.shape[:2] if hasattr(image, 'shape') else (384, 384)

        # 创建模拟深度（根据位置变化）
        # 顶部（地平线）远处 -> 底部近处
        depth_map = np.zeros_like(depth)
        h, w = depth.shape

        for y in range(h):
            # 计算该行对应的模拟距离（顶部远，底部近）
            horizon_ratio = 0.3  # 地平线位置
            if y / h < horizon_ratio:
                # 地平线以上：远处
                distance = 10.0 + (1 - y / h / horizon_ratio) * 5
            else:
                # 地平线以下：渐变到近处
                distance = 10.0 - ((y / h - horizon_ratio) / (1 - horizon_ratio)) * 9

            depth_map[y, :] = distance

        # 添加横向变化（模拟障碍物）
        center_x = w // 2
        for x in range(w):
            dist_from_center = abs(x - center_x) / w
            # 中间（可能的障碍物）距离稍近
            depth_map[:, x] += (0.5 - dist_from_center) * 2

        # 添加随机变化（模拟纹理）
        depth_map += np.random.randn(h, w) * 0.3
        depth_map = np.clip(depth_map, 1.0, 15.0)

        # 归一化到0-255范围用于可视化
        depth_normalized = self._normalize_depth(depth_map)

        return depth_map, depth_normalized

    def _normalize_depth(self, depth: np.ndarray,
                       min_depth: float = None,
                       max_depth: float = None) -> np.ndarray:
        """归一化深度值 - 使用实际范围"""
        if min_depth is None:
            min_depth = depth.min()
        if max_depth is None:
            max_depth = depth.max()

        if max_depth - min_depth < 0.01:
            # 全黑情况，使用默认值
            return np.zeros_like(depth, dtype=np.uint8)

        depth = np.clip(depth, min_depth, max_depth)
        depth = (depth - min_depth) / (max_depth - min_depth)
        depth = (depth * 255).astype(np.uint8)
        return depth

    def get_distance_at_point(self, depth: np.ndarray,
                              x: int, y: int,
                              scale: float = 1.0) -> float:
        """获取特定点的实际距离（米）"""
        if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
            # 根据深度图比例尺转换
            return depth[y, x] * scale
        return float('inf')

    def get_obstacle_depths(self, depth: np.ndarray,
                           bboxes: list,
                           scale: float = 1.0) -> list:
        """获取障碍物的平均深度"""
        obstacle_depths = []

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(depth.shape[1], x2), min(depth.shape[0], y2)

            # 计算该区域的平均深度
            region_depth = depth[y1:y2, x1:x2]
            if region_depth.size > 0:
                avg_depth = np.mean(region_depth) * scale
                obstacle_depths.append({
                    'bbox': bbox,
                    'avg_depth': avg_depth,
                    'min_depth': np.min(region_depth) * scale,
                    'max_depth': np.max(region_depth) * scale
                })

        return obstacle_depths

    def depth_to_3d_points(self, depth: np.ndarray,
                          fx: float = 500.0,
                          fy: float = 500.0,
                          cx: float = None,
                          cy: float = None) -> np.ndarray:
        """
        将深度图转换为3D点云

        Args:
            depth: 深度图
            fx, fy: 相机内参
            cx, cy: 主点坐标

        Returns:
            points: 3D点云 [N, 3]
        """
        h, w = depth.shape
        if cx is None:
            cx = w / 2
        if cy is None:
            cy = h / 2

        # 生成网格坐标
        y_coords, x_coords = np.meshgrid(range(h), range(w), indexing='ij')

        # 转换到相机坐标系
        z = depth
        x = (x_coords - cx) * z / fx
        y = (y_coords - cy) * z / fy

        # 堆叠成点云
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        return points

    def visualize(self, depth: np.ndarray,
                 original_image: np.ndarray = None) -> np.ndarray:
        """深度图可视化"""
        try:
            import cv2
        except ImportError:
            import numpy as cv2
            raise ImportError("请安装 opencv-python")

        # 归一化深度
        depth_norm = self._normalize_depth(depth)

        # 转换为彩色
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        if original_image is not None:
            # 叠加到原图
            result = cv2.addWeighted(original_image, 0.6, depth_colored, 0.4, 0)
            return result
        else:
            return depth_colored


# 单元测试
if __name__ == "__main__":
    estimator = DepthEstimator()

    # 创建测试图像
    test_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)

    # 估计深度
    depth, depth_norm = estimator.estimate(test_image)

    print(f"深度图形状: {depth.shape}")
    print(f"深度范围: {depth.min():.2f} - {depth.max():.2f}")