# config_spot_robot.py
from dataclasses import dataclass, field
from typing import Dict, List

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("spot_robot")
@dataclass
class SpotRobotConfig(RobotConfig):
    """
    Configuration for the Boston Dynamics Spot robot in LeRobot.

    Fields:
        hostname: IP or hostname of Spot.
        username/password: Credentials for Spot's user account.
        update_rate_hz: Target control loop frequency for your control loop.
        use_front_cameras: Whether to stream front stereo pair.
        use_arm_camera: Whether to stream the arm/hand camera.
        image_sources: Optional explicit list of Spot image sources to use.
                       If empty, will be derived from the *_cameras flags.
        image_format: Spot image format (e.g. FORMAT_RGB_U8).
        image_width, image_height: Optional resize; 0 means native size.
    """

    hostname: str = "192.168.80.3"
    username: str = "admin"
    password: str = "password"

    update_rate_hz: float = 10.0

    # High-level camera toggles
    use_front_cameras: bool = True
    use_arm_camera: bool = True

    # Explicit Spot image source names (overrides flags if not empty)
    image_sources: List[str] = field(default_factory=list)

    # Image format and size hints. These map to Spot's ImageRequest fields.
    image_format: int = 0  # Will be interpreted as FORMAT_RGB_U8 in SpotRobot
    image_width: int = 0   # 0 => keep native width
    image_height: int = 0  # 0 => keep native height

    # Enable EAP (Enhanced Vision Package) hand depth cameras (Intel RealSense).
    use_depth_cameras: bool = False

    # Back and side fisheye cameras (standard Spot)
    use_back_camera: bool = True      # back_fisheye
    use_side_cameras: bool = True     # left_fisheye + right_fisheye

    # Proprioceptive sensor groups (all on by default)
    include_joint_states: bool = True        # 19 joints × (pos, vel, acc, load)
    include_foot_states: bool = True         # 4 feet × (contact, pos_x/y/z, friction_mu)
    include_manipulator_state: bool = True   # gripper %, holding, stow, wrench force/torque
    include_power_state: bool = True         # battery charge %, voltage, current, runtime
    include_body_velocity_odom: bool = True  # body vel in odom frame (IMU proxy)

    # You can add more config fields later (e.g. max velocities, arm limits).
    extra: Dict[str, float] = field(default_factory=dict)


@RobotConfig.register_subclass("dual_spot_robot")
@dataclass
class DualSpotRobotConfig(RobotConfig):
    """Configuration for two Spot robots controlled in parallel."""
    robot1: SpotRobotConfig = field(default_factory=SpotRobotConfig)
    robot2: SpotRobotConfig = field(default_factory=SpotRobotConfig)
