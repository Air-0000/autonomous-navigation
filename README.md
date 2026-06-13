# 无人系统智能感知与导航系统

基于深度学习的无人机/无人车环境感知与自主导航决策系统。

## 项目概述

本项目实现了一个完整的无人系统智能导航系统，包含：

- **感知层**：目标检测（YOLOv8）、深度估计（MiDaS）、语义分割（DeepLabV3+）
- **认知层**：类脑导航（位置细胞Place Cell + 网格细胞Grid Cell）
- **决策层**：轨迹预测、路径规划（A* + RRT*）
- **控制层**：PID控制、纯追踪控制
- **仿真层**：Webots仿真环境接口
- **展示层**：Web实时可视化界面

## 技术亮点

1. **类脑导航原理**：模拟大脑导航系统（2014诺贝尔奖发现）
2. **多感知融合**：CV + 深度学习多模态感知
3. **预训练模型**：全部使用预训练模型，开箱即用
4. **跨平台**：支持Mac M4 Air本地运行

## 项目结构

```
autonomous-navigation/
├── config/                    # 配置文件
│   ├── detection.yaml        # 目标检测配置
│   ├── depth.yaml            # 深度估计配置
│   ├── segmentation.yaml     # 语义分割配置
│   ├── planning.yaml         # 路径规划配置
│   └── webots.yaml           # Webots仿真配置
│
├── src/                       # 源代码
│   ├── perception/            # 感知模块
│   │   ├── detector.py        # YOLOv8目标检测
│   │   ├── depth_estimator.py # MiDaS深度估计
│   │   ├── segmentor.py       # DeepLabV3+语义分割
│   │   └── fusion.py          # 感知融合
│   │
│   ├── cognition/             # 类脑认知模块
│   │   └── cognitive_map.py  # 认知地图（Place Cell + Grid Cell）
│   │
│   ├── decision/             # 决策模块
│   │   ├── predictor.py       # 轨迹预测
│   │   ├── planner.py        # A* + RRT*路径规划
│   │   └── controller.py     # PID/纯追踪控制
│   │
│   ├── interface/            # 仿真接口
│   │   └── simulator.py      # Webots接口
│   │
│   └── utils/                # 工具函数
│
├── scripts/                   # 脚本
│   ├── demo/
│   │   └── demo_webots.py     # 主演示脚本
│   └── eval/
│       └── eval_system.py     # 评测脚本
│
├── web/                       # Web界面
│   └── app.py                # FastAPI可视化
│
└── requirements.txt           # 依赖清单
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行演示

```bash
# 命令行演示
python scripts/demo/demo_webots.py --max-steps 500

# Web界面演示
python web/app.py
# 访问 http://localhost:8000
```

### 3. 单独测试各模块

```bash
# 测试目标检测
python src/perception/detector.py

# 测试深度估计
python src/perception/depth_estimator.py

# 测试语义分割
python src/perception/segmentor.py

# 测试认知地图
python src/cognition/cognitive_map.py

# 测试路径规划
python src/decision/planner.py
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 目标检测 | YOLOv8 (ultralytics) |
| 深度估计 | MiDaS (intel-isl) |
| 语义分割 | DeepLabV3+ (torchvision) |
| 类脑导航 | Place Cell + Grid Cell (自研) |
| 轨迹预测 | Constant Velocity + Linear Regression |
| 路径规划 | A* + RRT* |
| 控制 | PID + Pure Pursuit |
| 仿真 | Webots |
| Web | FastAPI + WebSocket |

## 类脑导航原理

### 位置细胞 (Place Cells)
- 发现者：John O'Keefe（1971年）
- 功能：在特定位置激活，形成认知地图
- 模拟方式：高斯激活函数

### 网格细胞 (Grid Cells)
- 发现者：May-Britt Moser & Edvard Moser（2005年）
- 奖项：2014年诺贝尔生理学或医学奖
- 功能：形成空间坐标系统，提供度量信息
- 模拟方式：周期性激活模式

## 性能指标

| 模块 | 指标 | 预期值 |
|------|------|--------|
| 目标检测 | mAP@0.5 | ≥0.75 |
| 深度估计 | AbsRel | ≤0.15 |
| 语义分割 | mIoU | ≥0.65 |
| 路径规划 | 成功率 | ≥90% |
| 帧率 | FPS | ≥15 |

## 硬件要求

- **推荐**：Mac M4 Air（Apple Silicon）
- **最低**：任意有4GB以上内存的设备
- **可选**：GPU加速（NVIDIA CUDA）

## 适用场景

1. 室内无人机导航
2. 仓库物流机器人
3. 自动驾驶感知系统
4. 智能巡检机器人

## 致谢

本项目的类脑导航模块参考了以下研究：
- O'Keefe, J., & Nadel, L. (1978). The Hippocampus as a Cognitive Map
- Hafting, T., et al. (2005). Microstructure of a spatial map in the entorhinal cortex

## License

MIT License