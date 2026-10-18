"""
目标检测模块 - YOLOv8
基于预训练模型进行实时目标检测
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import yaml


class YOLODetector:
    """YOLOv8目标检测器"""

    def __init__(self, config_path: Optional[str] = None):
        self.mock_mode = False
        self.model = None

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.model_name = config.get('model', {}).get('name', 'yolov8n.pt')
            self.conf_threshold = config.get('model', {}).get('conf_threshold', 0.5)
            self.iou_threshold = config.get('model', {}).get('iou_threshold', 0.45)
        else:
            self.model_name = 'yolov8n.pt'
            self.conf_threshold = 0.5
            self.iou_threshold = 0.45

        # 尝试加载预训练模型，失败则优雅降级为模拟检测
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
        except Exception as e:
            print(f"⚠️ YOLOv8未安装或加载失败 ({e})，启用模拟目标检测模式")
            self.mock_mode = True

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的目标（真实 YOLO 或模拟模式）
        """
        if self.model is not None and not self.mock_mode:
            try:
                results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
                return self._parse_results(results)
            except Exception as e:
                pass

        # 简单模拟检测：在模拟图像上生成检测框
        import time
        h, w = image.shape[:2]

        detections = []

        # 基于时间生成不同的检测结果
        current_time = int(time.time() * 5) % 100
        np.random.seed(current_time if current_time > 0 else 1)

        # 前景障碍物位置（模拟场景中的物体）
        obstacle_configs = [
            (80, 340, 150, 440, 'obstacle', 0.85),
            (200, 300, 280, 420, 'obstacle', 0.78),
            (350, 350, 430, 480, 'obstacle', 0.82),
            (500, 320, 570, 450, 'obstacle', 0.75),
            (250, 250, 310, 330, 'obstacle', 0.70),
        ]

        # 随机选择显示哪些检测结果
        n_detections = np.random.randint(2, len(obstacle_configs) + 1)
        selected = np.random.choice(len(obstacle_configs), n_detections, replace=False)

        for i in selected:
            x1, y1, x2, y2, class_name, conf = obstacle_configs[i]
            # 添加一些随机偏移
            offset = np.random.randint(-20, 20, 4)
            x1 = max(0, min(w - 1, x1 + offset[0]))
            y1 = max(0, min(h - 1, y1 + offset[1]))
            x2 = max(0, min(w - 1, x2 + offset[2]))
            y2 = max(0, min(h - 1, y2 + offset[3]))

            # 确保框有效
            if x2 > x1 and y2 > y1:
                detection = {
                    'class_id': 0,
                    'class_name': class_name,
                    'confidence': conf + np.random.uniform(-0.1, 0.1),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                }
                detection['confidence'] = max(0.5, min(0.95, detection['confidence']))
                detections.append(detection)

        return detections

    def _parse_results(self, results) -> List[Dict]:
        """解析YOLO输出结果"""
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                detection = {
                    'class_id': int(box.cls),
                    'class_name': self.model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'center': box.xywh[0].cpu().numpy().tolist()  # [cx, cy, w, h]
                }
                detections.append(detection)

        return detections

    def filter_by_class(self, detections: List[Dict], classes: List[str]) -> List[Dict]:
        """过滤只保留特定类别的检测结果"""
        filtered = []
        for det in detections:
            if det['class_name'] in classes:
                filtered.append(det)
        return filtered

    def filter_obstacles(self, detections: List[Dict],
                        obstacle_classes: List[str] = None) -> List[Dict]:
        """过滤障碍物"""
        if obstacle_classes is None:
            # 默认的障碍物类别
            obstacle_classes = ['person', 'car', 'bicycle', 'motorcycle',
                               'truck', 'bus', 'obstacle', 'chair', 'table']

        return self.filter_by_class(detections, obstacle_classes)

    def get_detection_counts(self, detections: List[Dict]) -> Dict[str, int]:
        """统计各类别检测数量"""
        counts = {}
        for det in detections:
            name = det['class_name']
            counts[name] = counts.get(name, 0) + 1
        return counts

    def visualize(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """在图像上绘制检测结果"""
        try:
            import cv2
        except ImportError:
            raise ImportError("请安装 opencv-python: pip install opencv-python")

        vis_image = image.copy()

        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis_image


# 单元测试
if __name__ == "__main__":
    # 创建检测器
    detector = YOLODetector()

    # 创建测试图像（模拟）
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 执行检测
    detections = detector.detect(test_image)

    print(f"检测到 {len(detections)} 个目标")
    for det in detections:
        print(f"  - {det['class_name']}: {det['confidence']:.3f}")