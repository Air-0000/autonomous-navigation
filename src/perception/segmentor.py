"""
语义分割模块 - DeepLabV3+
基于预训练模型进行图像语义分割
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml


class DeepLabSegmentor:
    """DeepLabV3+语义分割器"""

    # Cityscapes风格类别颜色
    COLORS = [
        (128, 64, 128),   # road
        (244, 35, 232),   # sidewalk
        (70, 70, 70),     # building
        (102, 102, 102),  # wall
        (156, 102, 102),  # fence
        (153, 153, 153),  # pole
        (250, 170, 30),   # traffic light
        (220, 220, 0),    # traffic sign
        (107, 142, 35),   # vegetation
        (152, 251, 152),  # terrain
        (255, 0, 0),      # person
        (255, 255, 0),    # rider
        (0, 0, 142),      # car
        (0, 0, 70),       # truck
        (0, 60, 100),     # bus
        (0, 80, 100),     # train
        (0, 0, 230),      # motorcycle
        (119, 11, 32),    # bicycle
        (0, 0, 0),        # unknown
    ]

    def __init__(self, config_path: Optional[str] = None):
        try:
            import torch
            import torchvision
        except ImportError:
            raise ImportError("请安装 torch torchvision")

        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.model_name = config['model']['name']
        else:
            self.model_name = 'deeplabv3_resnet101'

        # 加载预训练模型
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DeepLabV3_ResNet101_Weights.DEFAULT"
        )

        # 设置设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

        # 类别列表
        self.labels = [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
            "person", "rider", "car", "truck", "bus", "train",
            "motorcycle", "bicycle", "unknown"
        ]

        # 可通行类别（用于路径规划）
        self.drivable_labels = ["road", "sidewalk"]

        # 障碍物类别
        self.obstacle_labels = ["person", "rider", "car", "truck", "bus",
                               "bicycle", "motorcycle", "wall", "fence", "pole"]

    @torch.no_grad()
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        分割图像

        Args:
            image: RGB格式图像 [H, W, 3]

        Returns:
            segmentation: 分割掩码 [H, W]，每个像素值是类别ID
            colored_mask: 彩色分割结果 [H, W, 3]
        """
        # 预处理
        input_tensor = self._preprocess(image)

        # 移动到设备
        input_tensor = input_tensor.to(self.device)

        # 推理
        output = self.model(input_tensor)['out'][0]
        predictions = output.argmax(0)

        # 转换为numpy
        segmentation = predictions.cpu().numpy()

        # 生成彩色掩码
        colored_mask = self._create_colored_mask(segmentation)

        return segmentation, colored_mask

    def _preprocess(self, image: np.ndarray):
        """预处理图像"""
        from PIL import Image as PILImage
        from torchvision import transforms

        # 确保图像是RGB
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # 转换为PIL Image
        pil_image = PILImage.fromarray(image)

        # 使用标准transform + 调整大小
        transform = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # 转换为张量 [1, 3, H, W]
        tensor = transform(pil_image).unsqueeze(0)

        return tensor

    def _create_colored_mask(self, segmentation: np.ndarray) -> np.ndarray:
        """创建彩色分割掩码"""
        import cv2

        h, w = segmentation.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in enumerate(self.COLORS):
            mask = segmentation == class_id
            colored[mask] = color

        # 高斯模糊使边界更平滑
        colored = cv2.GaussianBlur(colored, (5, 5), 0)

        return colored

    def get_drivable_mask(self, segmentation: np.ndarray) -> np.ndarray:
        """获取可通行区域掩码"""
        mask = np.zeros_like(segmentation, dtype=bool)
        for label in self.drivable_labels:
            label_id = self.labels.index(label) if label in self.labels else -1
            if label_id >= 0:
                mask |= (segmentation == label_id)
        return mask

    def get_obstacle_mask(self, segmentation: np.ndarray) -> np.ndarray:
        """获取障碍物区域掩码"""
        mask = np.zeros_like(segmentation, dtype=bool)
        for label in self.obstacle_labels:
            label_id = self.labels.index(label) if label in self.labels else -1
            if label_id >= 0:
                mask |= (segmentation == label_id)
        return mask

    def get_free_space_ratio(self, segmentation: np.ndarray) -> float:
        """计算可通行空间占比"""
        drivable = self.get_drivable_mask(segmentation)
        total_pixels = segmentation.size
        free_pixels = np.sum(drivable)
        return free_pixels / total_pixels if total_pixels > 0 else 0.0

    def overlay_segmentation(self, image: np.ndarray,
                            segmentation: np.ndarray,
                            alpha: float = 0.5) -> np.ndarray:
        """在原图上叠加分割结果"""
        import cv2

        colored = self._create_colored_mask(segmentation)

        # 调整大小以匹配原图
        colored = cv2.resize(colored, (image.shape[1], image.shape[0]))

        # 叠加
        overlay = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)

        return overlay

    def get_class_statistics(self, segmentation: np.ndarray) -> Dict[str, int]:
        """统计各类别像素数量"""
        stats = {}
        for class_id, label in enumerate(self.labels):
            count = np.sum(segmentation == class_id)
            if count > 0:
                stats[label] = int(count)
        return stats

    def visualize(self, image: np.ndarray,
                 segmentation: np.ndarray) -> np.ndarray:
        """可视化分割结果"""
        import cv2

        # 叠加半透明分割
        overlay = self.overlay_segmentation(image, segmentation, alpha=0.5)

        # 添加图例
        legend_h = 100
        legend = np.ones((legend_h, overlay.shape[1], 3), dtype=np.uint8) * 255

        # 绘制主要类别图例
        key_labels = ["person", "car", "road", "vegetation"]
        n_labels = len(key_labels)
        step = legend.shape[1] // n_labels

        for i, label in enumerate(key_labels):
            x = i * step + 10
            if label in self.labels:
                color = self.COLORS[self.labels.index(label)]
                cv2.rectangle(legend, (x, 20), (x + 30, 50),
                            color[::-1], -1)
                cv2.putText(legend, label, (x + 40, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 拼接
        result = np.vstack([overlay, legend])

        return result


# 单元测试
if __name__ == "__main__":
    segmentor = DeepLabSegmentor()

    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 执行分割
    segmentation, colored = segmentor.segment(test_image)

    print(f"分割掩码形状: {segmentation.shape}")
    print(f"类别统计: {segmentor.get_class_statistics(segmentation)}")
    print(f"可通行空间比例: {segmentor.get_free_space_ratio(segmentation):.2%}")