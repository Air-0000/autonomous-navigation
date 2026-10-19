# 无人系统智能感知与导航系统

基于深度学习的无人机/无人车环境感知与自主导航决策系统。

## 项目概述

本项目实现了一个基于生物物理与神经计算机制的智能自主导航系统，面向真实 3D 物理仿真与物理具身机器人平台：

- **感知层**：单目视觉目标检测（YOLOv8）、特征地标几何测距与相对朝向估计（LandmarkDetector）、多模态状态融合
- **类脑认知层**：
  - **头向细胞 (Head Direction, HD) 环形吸引子网络**：通过非对称侧向抑制连续积分陀螺仪角速度 $\omega_z$，追踪平滑朝向 $\hat{\theta}$（Zhang 1996 模型）
  - **多尺度二维连续吸引子神经网络 (CANN Grid Cells)**：多波长神经拓扑薄片 ($\lambda \in \{0.8, 1.4, 2.6\}\text{m}$)，经速度调制突触移位实现无坐标泄露的连续路径积分（Burak & Fiete 2009）
  - **位置细胞 (Place Cell) 联想记忆与漂移校正**：基于赫布学习（Hebbian Plasticity）将视觉地标拓扑绑定至位置细胞表征，在重访已知地标时实现全局累积漂移清零与相位校准（Solstad et al. 2006）
  - **向量认知规划 (Vector-Based Cognitive Planning)**：解码目标地标向量差，直接驱动差速移动底盘线速度 $v$ 与角速度 $\omega$
- **3D 物理仿真层**：Gazebo Sim (Harmonic / ROS2 Jazzy) 真实刚体动力学环境，包含完整差速驱动插件、摩擦力模型、IMU 与物理相机传感器
- **桥接接口层**：原生 Gazebo 传输协议 (`gz.transport13` / `gz.msgs10`) 与 ROS2 兼容双向接口

## 项目结构

```
autonomous-navigation/
├── simulation/                # 3D Gazebo 物理仿真环境
│   ├── launch_gazebo.sh       # 一键启动 Gazebo 3D 世界脚本
│   ├── worlds/
│   │   └── navigation_world.sdf  # 10m×10m 封闭竞技场、物理障碍物与多色地标世界
│   └── models/
│       └── nav_robot/         # 差速驱动机器人 (带 IMU、单目相机、DiffDrive 插件)
│           ├── model.sdf
│           └── model.config
│
├── src/
│   ├── cognition/             # 生物神经认知导航核心
│   │   └── cognitive_map.py   # HD 环形吸引子 + 多尺度 CANN 网格细胞 + 赫布位置细胞
│   ├── perception/            # 环境感知模块
│   │   ├── detector.py        # YOLOv8 目标检测
│   │   └── landmark_detector.py # 视觉色度空间与针孔几何地标检测
│   ├── interface/             # 仿真与硬件桥接
│   │   ├── gazebo_bridge.py   # 原生 Gazebo 3D 话题收发桥接
│   │   └── simulator.py       # 轻量回退接口
│   ├── decision/              # 决策控制
│   │   ├── planner.py         # 向量认知规划与障碍物斥力场
│   │   └── controller.py      # PID 与差速运动控制器
│   └── demo_closed_loop_gazebo.py # 闭环 3D Gazebo 仿真导航主控入口
│
├── tests/
│   └── test_cognitive_navigation.py # CANN 环形吸引子与路径积分单元/集成测试套件
└── README.md
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
### 1. 运行 Gazebo 3D 物理仿真与类脑闭环导航

```bash
# 终端 1: 启动 3D Gazebo 物理仿真环境 (支持 GUI 渲染或 --headless 无头模式)
./simulation/launch_gazebo.sh

# 终端 2: 启动生物神经网络 (CANN + HD Ring Attractor) 闭环自主导航主控
python src/demo_closed_loop_gazebo.py --steps 200 --rate 20.0
```

### 2. 运行生物认知导航单元测试套件

```bash
# 验证 HD 吸引子环、多尺度网格细胞移位、地标漂移重置与向量导航规划
pytest tests/test_cognitive_navigation.py -v
```

### 3. 单独测试各感知与认知子模块

```bash
# 测试视觉特征地标检测与相对测距
python src/perception/landmark_detector.py

# 测试认知地图 (HD Ring Attractor + CANN 2D Grid Cells)
python src/cognition/cognitive_map.py

# 测试原生 Gazebo 话题通信桥接
python src/interface/gazebo_bridge.py
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