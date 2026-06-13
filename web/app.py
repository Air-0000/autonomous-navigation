"""
Web可视化界面
实时展示感知和决策结果
"""

import os
import sys
import asyncio
import json
import base64
import numpy as np
from io import BytesIO
from typing import Optional
from datetime import datetime

# Web框架
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# 图像处理
import cv2

# 获取项目根目录（web/ -> 项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.perception.detector import YOLODetector
from src.perception.depth_estimator import DepthEstimator
from src.perception.segmentor import DeepLabSegmentor
from src.perception.fusion import PerceptionFusion
from src.cognition.cognitive_map import CognitiveMap
from src.decision.planner import HybridPlanner
from src.decision.predictor import TrajectoryPredictor
from src.interface.simulator import WebotsSimulator


app = FastAPI(title="无人系统导航监控平台")

# 全局系统实例
navigation_system = None
websocket_clients: list = []


class SimpleNavigationSystem:
    """简化版导航系统（用于Web界面）- 优化版"""

    def __init__(self):
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'config'
        )

        # 初始化模块
        print("初始化感知模块...")
        self.detector = YOLODetector()
        print("加载深度估计...")
        self.depth_estimator = DepthEstimator()
        print("加载语义分割...")
        self.segmentor = DeepLabSegmentor()
        self.fusion = PerceptionFusion()
        self.cognitive_map = CognitiveMap()
        self.planner = HybridPlanner()
        self.predictor = TrajectoryPredictor()
        self.simulator = WebotsSimulator()

        # 连接仿真器
        self.simulator.connect()

        # 状态
        self.current_step = 0
        self.goal = (25.0, 25.0)  # 更远的目标，需要更长时间

        # 帧率控制 - 优化性能
        self.frame_skip = 10  # 每10帧做一次完整检测（5倍加速）
        self._cached_detection = []
        self._cached_depth = None
        self._cached_depth_norm = None
        self._cached_segmentation = None
        self._cached_image = None  # 目标更远，需要走更长时间

    def step(self):
        """执行一步 - 优化版"""
        self.current_step += 1

        # 更新仿真器状态（这会更新位置）
        self.simulator.step()

        # 获取图像
        image = self.simulator.get_camera_image()
        pose = self.simulator.get_pose()

        # 每隔n帧做完整检测，减少计算量
        do_full_detection = (self.current_step % self.frame_skip == 0)

        if do_full_detection:
            # 感知（完整版）
            detections = self.detector.detect(image)
            depth, depth_norm = self.depth_estimator.estimate(image)
            segmentation, _ = self.segmentor.segment(image)

            # 缓存结果
            self._cached_detection = detections
            self._cached_depth = depth
            self._cached_depth_norm = depth_norm
            self._cached_segmentation = segmentation

            # 融合
            obstacles = self.fusion.fuse(detections, depth, segmentation, pose[:3])

            # 类脑导航
            current_pos = (pose[0], pose[2])
            velocity = self.simulator.get_velocity()
            self.cognitive_map.update(current_pos, velocity, obstacles)

            # 轨迹预测
            predictions = self.predictor.update(obstacles)

            # 路径规划
            grid = self.fusion.build_occupancy_grid(obstacles)
            plan_result = self.planner.plan((pose[0], pose[2]), self.goal, grid, obstacles)
        else:
            # 快速模式：只做简单模拟
            detections = self._cached_detection
            depth = self._cached_depth if self._cached_depth is not None else np.zeros((480, 640))
            depth_norm = self._cached_depth_norm if self._cached_depth_norm is not None else np.zeros((480, 640))
            segmentation = self._cached_segmentation if self._cached_segmentation is not None else np.zeros((480, 640), dtype=np.uint8)

            # 简单模拟障碍物
            obstacles = []
            predictions = {}
            plan_result = {'path': [], 'success': False}

        # 模拟移动（持续向目标移动，不停止）
        dx = self.goal[0] - pose[0]
        dz = self.goal[1] - pose[2]
        dist = np.sqrt(dx**2 + dz**2)

        if dist > 0.5:  # 只有距离目标超过0.5米时才移动
            speed = 0.8  # 稍微快点
            vx = speed * dx / dist
            vz = speed * dz / dist
            self.simulator.set_velocity(vx, 0, vz)
            reached_goal = False
        else:
            # 到达目标，保持低速巡逻
            vx = 0.1 * np.sin(self.current_step * 0.1)
            vz = 0.1 * np.cos(self.current_step * 0.1)
            self.simulator.set_velocity(vx, 0, vz)
            reached_goal = True

        return {
            'step': self.current_step,
            'image': image,
            'depth': depth_norm,
            'segmentation': segmentation,
            'detections': detections,
            'obstacles': obstacles,
            'predictions': predictions,
            'path': plan_result.get('path', []),
            'cognition': self.cognitive_map.activity_map,
            'pose': pose,
            'success': reached_goal
        }

    def close(self):
        self.simulator.close()


def numpy_to_base64(img: np.ndarray) -> str:
    """将numpy数组转为base64字符串"""
    if img is None:
        return ""

    # 编码为JPEG
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


@app.get("/", response_class=HTMLResponse)
async def index():
    """主页"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>无人系统导航监控平台</title>
        <meta charset="utf-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #fff;
                min-height: 100vh;
            }
            .header {
                background: rgba(0,0,0,0.3);
                padding: 20px;
                text-align: center;
                border-bottom: 2px solid #0f3460;
            }
            .header h1 {
                color: #00d4ff;
                font-size: 28px;
                text-shadow: 0 0 10px rgba(0,212,255,0.5);
            }
            .header p {
                color: #aaa;
                margin-top: 10px;
            }
            .main-container {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                padding: 20px;
                max-width: 1600px;
                margin: 0 auto;
            }
            .panel {
                background: rgba(255,255,255,0.05);
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }
            .panel-header {
                background: rgba(0,212,255,0.2);
                padding: 10px 15px;
                font-weight: bold;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .panel-header h3 {
                font-size: 14px;
            }
            .panel-header .badge {
                background: #00d4ff;
                color: #000;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 12px;
            }
            .panel-content {
                padding: 10px;
                min-height: 200px;
            }
            .panel-content img {
                width: 100%;
                height: 180px;
                object-fit: cover;
                border-radius: 5px;
            }
            .status-bar {
                background: rgba(0,0,0,0.5);
                padding: 15px 20px;
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                gap: 20px;
                margin-top: 20px;
            }
            .status-item {
                text-align: center;
            }
            .status-item .label {
                color: #888;
                font-size: 12px;
            }
            .status-item .value {
                font-size: 20px;
                font-weight: bold;
                color: #00d4ff;
            }
            .controls {
                text-align: center;
                padding: 20px;
            }
            .btn {
                padding: 12px 30px;
                margin: 0 10px;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .btn-primary {
                background: linear-gradient(135deg, #00d4ff, #0099cc);
                color: #fff;
            }
            .btn-primary:hover {
                transform: scale(1.05);
                box-shadow: 0 0 20px rgba(0,212,255,0.5);
            }
            .btn-danger {
                background: linear-gradient(135deg, #ff6b6b, #cc5555);
                color: #fff;
            }
            .connection-status {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 5px;
            }
            .connected { background: #4ade80; }
            .disconnected { background: #ef4444; }
            .loading {
                display: none;
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(0,0,0,0.8);
                padding: 30px;
                border-radius: 10px;
                text-align: center;
            }
            .spinner {
                width: 40px;
                height: 40px;
                border: 4px solid #333;
                border-top-color: #00d4ff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛸 无人系统智能导航监控平台</h1>
            <p>
                <span class="connection-status disconnected" id="wsStatus"></span>
                <span id="connectionText">等待连接...</span>
            </p>
        </div>

        <div class="main-container">
            <div class="panel">
                <div class="panel-header">
                    <h3>📷 实时画面</h3>
                    <span class="badge">LIVE</span>
                </div>
                <div class="panel-content">
                    <img id="cameraImage" src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 320 180'><rect fill='%23333' width='320' height='180'/><text x='160' y='90' text-anchor='middle' fill='%23666' font-size='14'>等待视频流...</text></svg>" alt="Camera">
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <h3>🎯 目标检测</h3>
                    <span class="badge" id="detectionCount">0</span>
                </div>
                <div class="panel-content">
                    <img id="detectionImage" src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 320 180'><rect fill='%23333' width='320' height='180'/></svg>" alt="Detection">
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <h3>📐 深度估计</h3>
                    <span class="badge" id="depthRange">-</span>
                </div>
                <div class="panel-content">
                    <img id="depthImage" src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 320 180'><rect fill='%23333' width='320' height='180'/></svg>" alt="Depth">
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <h3>🧠 认知地图</h3>
                    <span class="badge" id="placeCellActivity">0%</span>
                </div>
                <div class="panel-content">
                    <img id="cognitionImage" src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 320 180'><rect fill='%23333' width='320' height='180'/></svg>" alt="Cognition">
                </div>
            </div>
        </div>

        <div class="status-bar">
            <div class="status-item">
                <div class="label">仿真步数</div>
                <div class="value" id="stepCount">0</div>
            </div>
            <div class="status-item">
                <div class="label">检测数量</div>
                <div class="value" id="totalDetections">0</div>
            </div>
            <div class="status-item">
                <div class="label">路径点数</div>
                <div class="value" id="pathPoints">0</div>
            </div>
            <div class="status-item">
                <div class="label">位置</div>
                <div class="value" id="position">(0, 0)</div>
            </div>
            <div class="status-item">
                <div class="label">状态</div>
                <div class="value" id="systemStatus">待启动</div>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-primary" id="startBtn" onclick="startSystem()">🚀 启动系统</button>
            <button class="btn btn-danger" id="stopBtn" onclick="stopSystem()">⏹ 停止</button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>正在初始化系统...</div>
        </div>

        <script>
            let ws = null;
            let isRunning = false;
            let updateInterval = null;

            function connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

                ws.onopen = () => {
                    document.getElementById('wsStatus').className = 'connection-status connected';
                    document.getElementById('connectionText').textContent = '已连接';
                };

                ws.onclose = () => {
                    document.getElementById('wsStatus').className = 'connection-status disconnected';
                    document.getElementById('connectionText').textContent = '连接断开';
                    isRunning = false;
                    if (updateInterval) clearInterval(updateInterval);
                };

                ws.onerror = () => {
                    alert('WebSocket连接错误');
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    updateUI(data);
                };
            }

            function updateUI(data) {
                // 更新图像
                if (data.camera) {
                    document.getElementById('cameraImage').src = 'data:image/jpeg;base64,' + data.camera;
                }
                if (data.detection) {
                    document.getElementById('detectionImage').src = 'data:image/jpeg;base64,' + data.detection;
                }
                if (data.depth) {
                    document.getElementById('depthImage').src = 'data:image/jpeg;base64,' + data.depth;
                }
                if (data.cognition) {
                    document.getElementById('cognitionImage').src = 'data:image/jpeg;base64,' + data.cognition;
                }

                // 更新状态
                document.getElementById('stepCount').textContent = data.step || 0;
                document.getElementById('detectionCount').textContent = data.detectionCount || 0;
                document.getElementById('totalDetections').textContent = data.totalDetections || 0;
                document.getElementById('pathPoints').textContent = data.pathPoints || 0;
                document.getElementById('depthRange').textContent = data.depthRange || '-';
                document.getElementById('placeCellActivity').textContent = data.placeCellActivity || '0%';
                document.getElementById('position').textContent = `(${(data.x || 0).toFixed(1)}, ${(data.z || 0).toFixed(1)})`;

                if (isRunning) {
                    document.getElementById('systemStatus').textContent = data.success ? '✓ 已到达' : '运行中';
                    document.getElementById('systemStatus').style.color = data.success ? '#4ade80' : '#00d4ff';
                }
            }

            async function startSystem() {
                if (isRunning) return;
                document.getElementById('loading').style.display = 'block';
                isRunning = true;

                try {
                    const response = await fetch('/start', { method: 'POST' });
                    const result = await response.json();
                    document.getElementById('loading').style.display = 'none';

                    if (result.status === 'started') {
                        updateInterval = setInterval(async () => {
                            if (!isRunning) {
                                clearInterval(updateInterval);
                                return;
                            }
                            try {
                                const res = await fetch('/state');
                                const data = await res.json();
                                updateUI(data);

                                // 持续运行，直到用户手动停止
                                // 最多运行1000步自动停止
                                if (data.step >= 1000) {
                                    clearInterval(updateInterval);
                                    isRunning = false;
                                    document.getElementById('systemStatus').textContent = '演示完成';
                                }
                            } catch (e) {
                                console.error('获取状态失败', e);
                            }
                        }, 100);
                    }
                } catch (e) {
                    document.getElementById('loading').style.display = 'none';
                    alert('启动失败: ' + e.message);
                }
            }

            async function stopSystem() {
                isRunning = false;
                if (updateInterval) {
                    clearInterval(updateInterval);
                }
                await fetch('/stop', { method: 'POST' });
                document.getElementById('systemStatus').textContent = '已停止';
            }

            // 连接WebSocket
            connect();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接"""
    await websocket.accept()
    websocket_clients.append(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_clients.remove(websocket)


@app.post("/start")
async def start_system():
    """启动系统"""
    global navigation_system

    if navigation_system is None:
        navigation_system = SimpleNavigationSystem()

    return {"status": "started"}


@app.post("/stop")
async def stop_system():
    """停止系统"""
    global navigation_system

    if navigation_system:
        navigation_system.close()
        navigation_system = None

    return {"status": "stopped"}


@app.get("/state")
async def get_state():
    """获取当前状态"""
    global navigation_system

    if navigation_system is None:
        return {"error": "System not started"}

    result = navigation_system.step()

    # 生成可视化图像
    det_vis = navigation_system.detector.visualize(
        result['image'].copy(),
        result['detections']
    )

    depth_colored = cv2.applyColorMap(result['depth'], cv2.COLORMAP_JET)

    cognitive_vis = navigation_system.cognitive_map.visualize()
    cognitive_vis = cv2.resize(cognitive_vis, (320, 240))

    # 编码图像
    camera_b64 = numpy_to_base64(result['image'])
    detection_b64 = numpy_to_base64(det_vis)
    depth_b64 = numpy_to_base64(depth_colored)
    cognition_b64 = numpy_to_base64(cognitive_vis)

    # 统计
    total_det = sum(1 for _ in result['detections'])

    # 转换numpy类型为Python原生类型
    depth_min = float(result['depth'].min())
    depth_max = float(result['depth'].max())
    place_cell_max = float(navigation_system.cognitive_map.activity_map.max()) if navigation_system.cognitive_map.activity_map.size > 0 else 0.0

    return {
        "step": int(result['step']),
        "camera": camera_b64,
        "detection": detection_b64,
        "depth": depth_b64,
        "cognition": cognition_b64,
        "detectionCount": int(len(result['detections'])),
        "totalDetections": int(total_det),
        "pathPoints": int(len(result['path'])) if result['path'] else 0,
        "depthRange": f"{depth_min:.1f}-{depth_max:.1f}m",
        "placeCellActivity": f"{int(place_cell_max * 100)}%",
        "x": float(result['pose'][0]),
        "z": float(result['pose'][2]),
        "success": bool(result['success'])
    }


if __name__ == "__main__":
    print("="*50)
    print("无人系统导航监控平台")
    print("访问 http://localhost:8000 查看")
    print("="*50)

    uvicorn.run(app, host="0.0.0.0", port=8000)