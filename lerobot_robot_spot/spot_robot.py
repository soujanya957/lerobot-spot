from __future__ import annotations
"""
LeRobot Robot wrapper for Boston Dynamics Spot.

This class integrates Spot into the LeRobot ecosystem by implementing
the standard Robot interface:
   - observation_features
   - action_features
   - connect / disconnect
   - get_observation / send_action
   - is_connected / is_calibrated / calibrate / configure

Observations include:
   - Base pose and velocity.
   - Arm hand pose target state (x, y, z, qw, qx, qy, qz) in body frame.
   - Onboard camera images (front pair + arm/hand camera).

Actions include:
   - Base SE2 body velocity (vx, vy, yaw rate).
   - Arm hand pose target (x, y, z, qw, qx, qy, qz) in body frame.

SAFETY:
   - Base velocities are clamped using limits in config.extra.
   - Arm pose uses absolute workspace clamps and per-step translation clamps.
   - Start with conservative values and small actions in a safe area.
"""

import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, HAND_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient
from lerobot.robots import Robot

import logging

from .config_spot_robot import DualSpotRobotConfig, SpotRobotConfig

logger = logging.getLogger(__name__)


class SpotRobot(Robot):
    """
    LeRobot Robot implementation for Boston Dynamics Spot.

    This class uses the standard BD SDK services (RobotState, RobotCommand,
    Image, Lease) to control the robot and read state, while presenting a
    hardware-agnostic interface to LeRobot.
    """

    config_class = SpotRobotConfig
    name = "spot_robot"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, config: SpotRobotConfig):
        """
        Initialize the SpotRobot wrapper.

        Only stores configuration and prepares internal structures.
        Actual connection happens in connect().
        """
        super().__init__(config)
        self.config: SpotRobotConfig = config

        # BD SDK clients; initialized in connect().
        self._sdk = None
        self._robot = None
        self._state_client: RobotStateClient | None = None
        self._lease_client: LeaseClient | None = None
        self._command_client: RobotCommandClient | None = None
        self._image_client: ImageClient | None = None
        self._lease = None

        # Dummy calibration container for compatibility with LeRobot.
        self.calibration: Dict[str, Any] = {}

        # Determine which Spot image sources to use (front pair + arm).
        self._image_sources: List[str] = self._build_image_source_list()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """True when all SDK clients are initialized."""
        sdk_ok = (
            self._robot is not None
            and self._state_client is not None
            and self._lease_client is not None
            and self._command_client is not None
        )
        img_ok = self._image_client is not None
        return sdk_ok and img_ok

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to Spot and prepare for control.

        Steps:
            1. Create SDK and Robot objects.
            2. Authenticate and time-sync.
            3. Create state, lease, command, and image clients.
            4. Acquire a lease.
            5. Power on and stand.
        """
        # 1) SDK and robot
        self._sdk = create_standard_sdk("lerobot_spot_client")
        self._robot = self._sdk.create_robot(self.config.hostname)
        self._robot.authenticate(self.config.username, self.config.password)
        self._robot.time_sync.wait_for_sync()

        # 2) Service clients
        self._state_client = self._robot.ensure_client(
            RobotStateClient.default_service_name
        )
        self._lease_client = self._robot.ensure_client(
            LeaseClient.default_service_name
        )
        self._command_client = self._robot.ensure_client(
            RobotCommandClient.default_service_name
        )
        self._image_client = self._robot.ensure_client(
            ImageClient.default_service_name
        )

        # 3) Lease and power
        self._lease = self._lease_client.acquire()
        self._robot.power_on(timeout_sec=30)
        blocking_stand(self._command_client, timeout_sec=15)

        # 4) Calibration / configuration hooks for compatibility
        if calibrate:
            self.calibrate()
        self.configure()

    def disconnect(self) -> None:
        """
        Release lease and power down Spot.
        Should be called in a finally block to ensure safe shutdown.
        """
        if self._lease_client is not None and self._lease is not None:
            try:
                self._lease_client.return_lease(self._lease)
            except Exception:
                pass

        if self._robot is not None:
            try:
                self._robot.power_off(cut_immediately=False, timeout_sec=30)
            except Exception:
                pass

        self._state_client = None
        self._lease_client = None
        self._command_client = None
        self._image_client = None
        self._lease = None
        self._robot = None
        self._sdk = None

    def disconnect_keep_powered(self) -> None:
        """
        Release the SDK lease and close clients WITHOUT powering off Spot.

        Use this when you want to end a LeRobot session but leave Spot
        standing so a human operator or another process can take over.
        Spot will remain standing until its own sit-down timeout triggers
        or another client acquires the lease.
        """
        if self._lease_client is not None and self._lease is not None:
            try:
                self._lease_client.return_lease(self._lease)
            except Exception:
                pass

        self._state_client = None
        self._lease_client = None
        self._command_client = None
        self._image_client = None
        self._lease = None
        self._robot = None
        self._sdk = None

    # ------------------------------------------------------------------
    # Calibration & configuration
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """Spot is factory-calibrated; always return True."""
        return True

    def calibrate(self) -> None:
        """Placeholder calibration hook."""
        return

    def configure(self) -> None:
        """
        Configuration hook.
        You may use this to unstow the arm, etc. Keep this idempotent.
        """
        return

    # ------------------------------------------------------------------
    # Feature specification
    # ------------------------------------------------------------------

    @property
    def _base_ft(self) -> Dict[str, type]:
        return {
            "base.pos_x":   float,
            "base.pos_y":   float,
            "base.yaw":     float,
            "base.vel_x":   float,
            "base.vel_y":   float,
            "base.vel_yaw": float,
        }

    @property
    def _arm_pose_ft(self) -> Dict[str, type]:
        return {
            "arm.pose.x": float,
            "arm.pose.y": float,
            "arm.pose.z": float,
            "arm.pose.qw": float,
            "arm.pose.qx": float,
            "arm.pose.qy": float,
            "arm.pose.qz": float,
        }

    @property
    def _camera_ft(self) -> Dict[str, Tuple[int, int, int]]:
        features: Dict[str, Any] = {}
        for src in self._image_sources:
            if self._is_depth_source(src):
                # Depth images: (H, W) float32 (meters); dimensions unknown until first frame.
                features[src] = (None, None)
            else:
                features[src] = (None, None, 3)
        return features

    @property
    def observation_features(self) -> Dict[str, Any]:
        return {**self._base_ft, **self._arm_pose_ft, **self._camera_ft}

    # FIX 1: Added missing @property decorator — without it action_features()
    # must be called as a method, which breaks the LeRobot interface that
    # accesses it as an attribute (robot.action_features).
    @property
    def action_features(self) -> Dict[str, Any]:
        """
        Full action feature specification.
        Base:
            base.vx   : linear velocity x (m/s)
            base.vy   : linear velocity y (m/s)
            base.vyaw : yaw rate (rad/s)
        Arm:
            arm.pose.{x,y,z,qw,qx,qy,qz} : target hand pose in odom frame
        """
        base_action = {
            "base.vx":   float,
            "base.vy":   float,
            "base.vyaw": float,
        }
        return {**base_action, **self._arm_pose_ft}

    # ------------------------------------------------------------------
    # Image sources
    # ------------------------------------------------------------------

    def _build_image_source_list(self) -> List[str]:
        """
        Build list of image source names to fetch from Spot.

        If SpotRobotConfig.image_sources is non-empty, use it as-is.
        Otherwise derive from high-level flags:
            use_front_cameras:    front stereo pair
            use_arm_camera:       arm/hand camera
            use_depth_cameras:    EAP RealSense depth cameras

        IMPORTANT: Verify source names with ImageClient.list_image_sources().
        """
        if self.config.image_sources:
            return list(self.config.image_sources)

        sources: List[str] = []
        if self.config.use_front_cameras:
            sources.extend(["frontleft_fisheye", "frontright_fisheye"])
        if self.config.use_arm_camera:
            sources.append("hand_color")
        if self.config.use_depth_cameras:
            sources.extend([
                "hand_depth_in_hand_color_frame",  # RealSense depth aligned to hand color
                "frontleft_depth",                 # Front-left depth
                "frontright_depth",                # Front-right depth
            ])
        return sources

    @staticmethod
    def _is_depth_source(source_name: str) -> bool:
        """Return True if the image source delivers depth data."""
        return "_depth" in source_name

    # ------------------------------------------------------------------
    # Helpers: base state, arm state, images
    # ------------------------------------------------------------------

    def _get_base_state(self) -> Dict[str, float]:
        """
        Read base pose and velocity from RobotState.

        Uses kinematic_state in the vision frame and approximates yaw from
        the body transform quaternion.
        """
        if self._state_client is None:
            raise ConnectionError("RobotStateClient not initialized.")

        state = self._state_client.get_robot_state()
        kin = state.kinematic_state
        tf = kin.transforms_snapshot

        body_edge = tf.child_to_parent_edge_map.get("odom", None)  # odom = world-frame pose; body/flat_body are always (0,0,0)
        if body_edge is None:
            pos_x = pos_y = yaw = 0.0
        else:
            t = body_edge.parent_tform_child
            pos_x = t.position.x
            pos_y = t.position.y
            qw = t.rotation.w
            qx = t.rotation.x
            qy = t.rotation.y
            qz = t.rotation.z
            yaw = np.arctan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )

        vel = kin.velocity_of_body_in_vision
        if vel is None:
            vel_x = vel_y = vel_yaw = 0.0
        else:
            vel_x   = vel.linear.x
            vel_y   = vel.linear.y
            vel_yaw = vel.angular.z

        return {
            "base.pos_x":   float(pos_x),
            "base.pos_y":   float(pos_y),
            "base.yaw":     float(yaw),
            "base.vel_x":   float(vel_x),
            "base.vel_y":   float(vel_y),
            "base.vel_yaw": float(vel_yaw),
        }

    def _get_hand_pose(self) -> Dict[str, float]:
        """Read current hand pose in body frame from RobotState transforms."""
        if self._state_client is None:
            raise ConnectionError("RobotStateClient not initialized.")

        state = self._state_client.get_robot_state()
        body_tform_hand = get_a_tform_b(
            state.kinematic_state.transforms_snapshot,
            BODY_FRAME_NAME,
            HAND_FRAME_NAME,
        )
        if body_tform_hand is None:
            # Conservative default if transform tree is temporarily unavailable.
            return {
                "arm.pose.x": 0.45,
                "arm.pose.y": 0.0,
                "arm.pose.z": 0.3,
                "arm.pose.qw": 1.0,
                "arm.pose.qx": 0.0,
                "arm.pose.qy": 0.0,
                "arm.pose.qz": 0.0,
            }

        return {
            "arm.pose.x": float(body_tform_hand.x),
            "arm.pose.y": float(body_tform_hand.y),
            "arm.pose.z": float(body_tform_hand.z),
            "arm.pose.qw": float(body_tform_hand.rot.w),
            "arm.pose.qx": float(body_tform_hand.rot.x),
            "arm.pose.qy": float(body_tform_hand.rot.y),
            "arm.pose.qz": float(body_tform_hand.rot.z),
        }

    def _get_images(self) -> Dict[str, np.ndarray]:
        """
        Fetch images from Spot's onboard cameras.

        Handles color and depth sources:
          Color sources:
            - FORMAT_JPEG: decoded with cv2.imdecode → (H, W, 3) uint8 RGB
            - FORMAT_RAW / PIXEL_FORMAT_GREYSCALE_U8: converted to 3-channel
            - FORMAT_RAW / PIXEL_FORMAT_RGB_U8: reshaped directly
          Depth sources (source name contains '_depth'):
            - PIXEL_FORMAT_DEPTH_MAP: 16-bit uint (mm) → (H, W) float32 (meters)

        Returns:
            dict mapping image_source_name -> np.ndarray.
        """
        if self._image_client is None:
            raise ConnectionError("ImageClient not initialized.")

        import cv2

        color_sources = [s for s in self._image_sources if not self._is_depth_source(s)]
        depth_sources = [s for s in self._image_sources if self._is_depth_source(s)]

        requests: List[image_pb2.ImageRequest] = []
        for src in color_sources:
            req = image_pb2.ImageRequest()
            req.image_source_name = src
            req.pixel_format = image_pb2.Image.PIXEL_FORMAT_RGB_U8
            requests.append(req)
        for src in depth_sources:
            req = image_pb2.ImageRequest()
            req.image_source_name = src
            req.pixel_format = image_pb2.Image.PIXEL_FORMAT_DEPTH_MAP
            requests.append(req)

        if not requests:
            return {}

        responses = self._image_client.get_image(requests)
        images: Dict[str, np.ndarray] = {}

        for resp in responses:
            src_name = resp.source.name
            img_proto = resp.shot.image
            h = img_proto.rows
            w = img_proto.cols

            # ---- Depth -------------------------------------------------------
            if self._is_depth_source(src_name):
                raw = np.frombuffer(img_proto.data, dtype=np.uint16)
                if h > 0 and w > 0 and raw.size == h * w:
                    # Convert mm → meters as float32
                    images[src_name] = (raw.reshape((h, w)).astype(np.float32) / 1000.0)
                continue

            # ---- Color -------------------------------------------------------
            data = np.frombuffer(img_proto.data, dtype=np.uint8)

            # JPEG / compressed
            if img_proto.format == image_pb2.Image.FORMAT_JPEG:
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                images[src_name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                continue

            # Raw greyscale
            if img_proto.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                if h <= 0 or w <= 0 or data.size != h * w:
                    continue
                images[src_name] = cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_GRAY2RGB)
                continue

            # Raw RGB
            if img_proto.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                if h <= 0 or w <= 0 or data.size != h * w * 3:
                    continue
                images[src_name] = data.reshape((h, w, 3))
                continue

            # Unknown format — skip silently.

        return images

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def get_observation(self) -> Dict[str, Any]:
        """
        Collect one observation from Spot.

        Includes base pose/velocity, arm pose, and camera images.
        """
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        obs: Dict[str, Any] = {}

        obs.update(self._get_base_state())

        obs.update(self._get_hand_pose())

        images = self._get_images()
        for src in self._image_sources:
            obs[src] = images.get(src, None)

        return obs

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a combined base + arm command to Spot.

        Action dict keys:
            base.vx   : linear velocity x (m/s)
            base.vy   : linear velocity y (m/s)
            base.vyaw : yaw rate (rad/s)
            arm.pose.x/y/z/qw/qx/qy/qz : target hand pose in body frame

        Safety:
            - Base velocities are clamped using values in config.extra.
            - Arm x/y/z are clamped to workspace bounds.
            - Arm x/y/z per-step deltas are clamped.
            - Quaternion is normalised.
        """
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        if self._command_client is None:
            raise ConnectionError("RobotCommandClient not initialized.")

        # ----------------------
        # Base velocity command
        # ----------------------
        vx   = float(action.get("base.vx",   0.0))
        vy   = float(action.get("base.vy",   0.0))
        vyaw = float(action.get("base.vyaw", 0.0))

        max_vx   = float(self.config.extra.get("max_vx",   0.5))
        max_vy   = float(self.config.extra.get("max_vy",   0.3))
        max_vyaw = float(self.config.extra.get("max_vyaw", 0.8))

        vx   = max(-max_vx,   min(max_vx,   vx))
        vy   = max(-max_vy,   min(max_vy,   vy))
        vyaw = max(-max_vyaw, min(max_vyaw, vyaw))

        # FIX 3: synchro_se2_velocity_command takes v_x, v_y, v_rot as
        # positional/keyword args — NOT a body_velocity proto message.
        # The original call would raise TypeError at runtime.
        mobility_cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=vx,
            v_y=vy,
            v_rot=vyaw,
        )
        # synchro_velocity_command requires an end_time or the command arrives
        # already "expired" (robot clock vs local clock skew).  Pass a deadline
        # 0.5 s from now, converted to robot time via time_sync.
        end_time_secs = time.time() + 0.5  # tune to your control-loop period
        self._command_client.robot_command(
            mobility_cmd, end_time_secs=end_time_secs
        )

        # ----------------------
        # Arm pose command (body frame)
        # ----------------------
        current_pose = self._get_hand_pose()

        x = float(action.get("arm.pose.x", current_pose["arm.pose.x"]))
        y = float(action.get("arm.pose.y", current_pose["arm.pose.y"]))
        z = float(action.get("arm.pose.z", current_pose["arm.pose.z"]))
        qw = float(action.get("arm.pose.qw", current_pose["arm.pose.qw"]))
        qx = float(action.get("arm.pose.qx", current_pose["arm.pose.qx"]))
        qy = float(action.get("arm.pose.qy", current_pose["arm.pose.qy"]))
        qz = float(action.get("arm.pose.qz", current_pose["arm.pose.qz"]))

        # Absolute workspace clamps (meters). Tune in config.extra if needed.
        min_x = float(self.config.extra.get("arm_min_x", 0.35))
        max_x = float(self.config.extra.get("arm_max_x", 1.05))
        min_y = float(self.config.extra.get("arm_min_y", -0.55))
        max_y = float(self.config.extra.get("arm_max_y", 0.55))
        min_z = float(self.config.extra.get("arm_min_z", -0.20))
        max_z = float(self.config.extra.get("arm_max_z", 0.60))

        # Per-step translation clamp (meters / control step).
        max_step_xyz = float(self.config.extra.get("arm_max_step_xyz", 0.03))

        dx = max(-max_step_xyz, min(max_step_xyz, x - current_pose["arm.pose.x"]))
        dy = max(-max_step_xyz, min(max_step_xyz, y - current_pose["arm.pose.y"]))
        dz = max(-max_step_xyz, min(max_step_xyz, z - current_pose["arm.pose.z"]))
        x = current_pose["arm.pose.x"] + dx
        y = current_pose["arm.pose.y"] + dy
        z = current_pose["arm.pose.z"] + dz

        x = max(min_x, min(max_x, x))
        y = max(min_y, min(max_y, y))
        z = max(min_z, min(max_z, z))

        # Normalize quaternion, fallback to current orientation if degenerate.
        q_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if q_norm < 1e-8:
            qw = current_pose["arm.pose.qw"]
            qx = current_pose["arm.pose.qx"]
            qy = current_pose["arm.pose.qy"]
            qz = current_pose["arm.pose.qz"]
        else:
            qw, qx, qy, qz = qw / q_norm, qx / q_norm, qy / q_norm, qz / q_norm

        arm_motion_time = float(self.config.extra.get("arm_pose_command_seconds", 0.1))
        arm_cmd = RobotCommandBuilder.arm_pose_command(
            x=x,
            y=y,
            z=z,
            qw=qw,
            qx=qx,
            qy=qy,
            qz=qz,
            frame_name=BODY_FRAME_NAME,
            seconds=arm_motion_time,
        )
        self._command_client.robot_command(arm_cmd)

        # Return the action actually sent after clamping
        sent_action: Dict[str, Any] = {
            "base.vx":   vx,
            "base.vy":   vy,
            "base.vyaw": vyaw,
            "arm.pose.x": x,
            "arm.pose.y": y,
            "arm.pose.z": z,
            "arm.pose.qw": qw,
            "arm.pose.qx": qx,
            "arm.pose.qy": qy,
            "arm.pose.qz": qz,
        }

        return sent_action


class DualSpotRobot(Robot):
    """
    Controls two Spot robots in parallel with a single teleop controller.

    The same action is broadcast to both robots on every `send_action` call.
    Observations from each robot are returned under prefixed keys
    (``robot1.<key>`` and ``robot2.<key>``).
    """

    config_class = DualSpotRobotConfig
    name = "dual_spot_robot"

    def __init__(self, config: DualSpotRobotConfig):
        super().__init__(config)
        self._robot1 = SpotRobot(config.robot1)
        self._robot2 = SpotRobot(config.robot2)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._robot1.is_connected and self._robot2.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self._robot1.connect(calibrate=calibrate)
        try:
            self._robot2.connect(calibrate=calibrate)
        except Exception:
            self._robot1.disconnect()
            raise

    def disconnect(self) -> None:
        errs = []
        for robot in [self._robot1, self._robot2]:
            try:
                robot.disconnect()
            except Exception as e:
                errs.append(e)
        if errs:
            raise errs[0]

    def disconnect_keep_powered(self) -> None:
        for robot in [self._robot1, self._robot2]:
            robot.disconnect_keep_powered()

    # ------------------------------------------------------------------
    # Calibration / configuration (delegated)
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._robot1.is_calibrated and self._robot2.is_calibrated

    def calibrate(self) -> None:
        self._robot1.calibrate()
        self._robot2.calibrate()

    def configure(self) -> None:
        self._robot1.configure()
        self._robot2.configure()

    # ------------------------------------------------------------------
    # Feature specification
    # ------------------------------------------------------------------

    @property
    def observation_features(self) -> Dict[str, Any]:
        """Prefix robot1/robot2 onto each robot's observation features."""
        features: Dict[str, Any] = {}
        for prefix, robot in [("robot1", self._robot1), ("robot2", self._robot2)]:
            for k, v in robot.observation_features.items():
                features[f"{prefix}.{k}"] = v
        return features

    @property
    def action_features(self) -> Dict[str, Any]:
        """Actions are shared; use robot1's action features."""
        return self._robot1.action_features

    # ------------------------------------------------------------------
    # Observations & actions
    # ------------------------------------------------------------------

    def get_observation(self) -> Dict[str, Any]:
        obs1 = self._robot1.get_observation()
        obs2 = self._robot2.get_observation()
        return {
            **{f"robot1.{k}": v for k, v in obs1.items()},
            **{f"robot2.{k}": v for k, v in obs2.items()},
        }

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast the same action to both robots. Returns robot1's clamped action."""
        sent1 = self._robot1.send_action(action)
        try:
            self._robot2.send_action(action)
        except Exception as e:
            logger.error("robot2 send_action failed: %s — stopping both robots.", e)
            try:
                self.disconnect()
            except Exception:
                pass
            raise
        return sent1
