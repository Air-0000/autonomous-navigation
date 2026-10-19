"""Gazebo Sim 3D Bridge for Autonomous Navigation.

Provides direct bi-directional communication with Gazebo Sim (Harmonic / ROS2 Jazzy):
- Subscribes: /camera/image_raw (RGB Monocular Vision)
- Subscribes: /imu/data (Angular Velocity, Linear Acceleration)
- Subscribes: /odom (Ground Truth Odometry & Drift Assessment)
- Publishes: /cmd_vel (Differential Drive Velocity Control)
"""

import logging
import threading
import time
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger("GazeboBridge")

# Check availability of native Gazebo Transport (gz.transport13 & gz.msgs10)
GZ_NATIVE_AVAILABLE = False
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.twist_pb2 import Twist as GzTwist
    from gz.msgs10.image_pb2 import Image as GzImage
    from gz.msgs10.imu_pb2 import IMU as GzIMU
    from gz.msgs10.odometry_pb2 import Odometry as GzOdometry
    GZ_NATIVE_AVAILABLE = True
except ImportError:
    pass


class GazeboBridge:
    """Thread-safe interface bridging Gazebo Sim topics with Python navigation pipeline."""

    def __init__(self,
                 camera_topic: str = "/camera/image_raw",
                 imu_topic: str = "/imu/data",
                 odom_topic: str = "/odom",
                 cmd_vel_topic: str = "/cmd_vel"):
        self.camera_topic = camera_topic
        self.imu_topic = imu_topic
        self.odom_topic = odom_topic
        self.cmd_vel_topic = cmd_vel_topic

        self._lock = threading.Lock()
        self._latest_image: Optional[np.ndarray] = None
        self._latest_imu: Dict[str, float] = {
            "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
            "accel_x": 0.0, "accel_y": 0.0, "accel_z": 0.0,
            "timestamp": 0.0
        }
        self._latest_odom: Dict[str, float] = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "yaw": 0.0, "linear_vel": 0.0, "angular_vel": 0.0,
            "timestamp": 0.0
        }

        self.node = None
        self.cmd_publisher = None
        self.is_connected = False

        if GZ_NATIVE_AVAILABLE:
            self._init_gz_transport()
        else:
            logger.warning("Native Gazebo transport not available in current python environment. Fallback to mock.")

    def _init_gz_transport(self):
        try:
            self.node = GzNode()
            # Subscribe to camera
            self.node.subscribe(GzImage, self.camera_topic, self._on_image)
            # Subscribe to IMU
            self.node.subscribe(GzIMU, self.imu_topic, self._on_imu)
            # Subscribe to Odometry
            self.node.subscribe(GzOdometry, self.odom_topic, self._on_odom)
            # Publisher for cmd_vel
            self.cmd_publisher = self.node.advertise(self.cmd_vel_topic, GzTwist)
            self.is_connected = True
            logger.info("Connected to Gazebo transport on topics: %s, %s, %s, %s",
                        self.camera_topic, self.imu_topic, self.odom_topic, self.cmd_vel_topic)
        except Exception as e:
            logger.error("Failed to initialize Gazebo transport: %s", e)
            self.is_connected = False

    def _on_image(self, msg: 'GzImage'):
        try:
            width = msg.width
            height = msg.height
            raw_data = msg.data
            # Format 3 is RGB_INT8, 4 is RGBA_INT8, 1 is L_INT8
            if msg.pixel_format_type == 3:  # RGB
                img = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
            elif msg.pixel_format_type == 4:  # RGBA
                img = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
            elif msg.pixel_format_type == 1:  # Grayscale
                gray = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width))
                img = np.stack([gray, gray, gray], axis=-1)
            else:
                img = np.frombuffer(raw_data, dtype=np.uint8)
                if len(img) == height * width * 3:
                    img = img.reshape((height, width, 3))
                else:
                    return

            with self._lock:
                self._latest_image = img
        except Exception as e:
            logger.debug("Error decoding Gazebo image: %s", e)

    def _on_imu(self, msg: 'GzIMU'):
        try:
            sec = msg.header.stamp.sec if hasattr(msg.header, 'stamp') else time.time()
            nsec = msg.header.stamp.nsec if hasattr(msg.header, 'stamp') else 0
            ts = sec + nsec * 1e-9

            with self._lock:
                self._latest_imu = {
                    "gyro_x": float(msg.angular_velocity.x),
                    "gyro_y": float(msg.angular_velocity.y),
                    "gyro_z": float(msg.angular_velocity.z),
                    "accel_x": float(msg.linear_acceleration.x),
                    "accel_y": float(msg.linear_acceleration.y),
                    "accel_z": float(msg.linear_acceleration.z),
                    "timestamp": ts
                }
        except Exception as e:
            logger.debug("Error decoding IMU msg: %s", e)

    def _on_odom(self, msg: 'GzOdometry'):
        try:
            sec = msg.header.stamp.sec if hasattr(msg.header, 'stamp') else time.time()
            nsec = msg.header.stamp.nsec if hasattr(msg.header, 'stamp') else 0
            ts = sec + nsec * 1e-9

            pos = msg.pose.position
            ori = msg.pose.orientation
            # Quaternion to yaw
            siny_cosp = 2.0 * (ori.w * ori.z + ori.x * ori.y)
            cosy_cosp = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
            yaw = float(np.arctan2(siny_cosp, cosy_cosp))

            with self._lock:
                self._latest_odom = {
                    "x": float(pos.x),
                    "y": float(pos.y),
                    "z": float(pos.z),
                    "yaw": yaw,
                    "linear_vel": float(msg.twist.linear.x),
                    "angular_vel": float(msg.twist.angular.z),
                    "timestamp": ts
                }
        except Exception as e:
            logger.debug("Error decoding Odometry msg: %s", e)

    def get_camera_frame(self) -> Optional[np.ndarray]:
        """Fetch latest camera frame (RGB uint8, HxWx3). Returns None if not yet received."""
        with self._lock:
            if self._latest_image is not None:
                return self._latest_image.copy()
        return None

    def get_imu_data(self) -> Dict[str, float]:
        """Fetch latest IMU measurement."""
        with self._lock:
            return dict(self._latest_imu)

    def get_odometry(self) -> Dict[str, float]:
        """Fetch latest Ground Truth Odometry."""
        with self._lock:
            return dict(self._latest_odom)

    def send_cmd_vel(self, linear_x: float, angular_z: float) -> bool:
        """Publish velocity command (m/s, rad/s) to Gazebo diff drive."""
        if not GZ_NATIVE_AVAILABLE or self.cmd_publisher is None:
            return False
        try:
            msg = GzTwist()
            msg.linear.x = float(linear_x)
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = float(angular_z)
            return self.cmd_publisher.publish(msg)
        except Exception as e:
            logger.error("Failed to publish cmd_vel: %s", e)
            return False

    def close(self):
        """Cleanly unsubscribe and release nodes."""
        self.is_connected = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing GazeboBridge...")
    bridge = GazeboBridge()
    print(f"Connected: {bridge.is_connected}")
    if bridge.is_connected:
        print("Sending test cmd_vel(0.2, 0.5)...")
        bridge.send_cmd_vel(0.2, 0.5)
        time.sleep(1.0)
        odom = bridge.get_odometry()
        imu = bridge.get_imu_data()
        print(f"Odom: {odom}")
        print(f"IMU: {imu}")
        bridge.send_cmd_vel(0.0, 0.0)
        bridge.close()
