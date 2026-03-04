# debug_frames.py
"""
Run this once to find the correct frame name for Spot's body transform.
Then update _get_base_state in spot_robot.py with the right key.
"""
from lerobot_robot_spot import SpotRobot, SpotRobotConfig

HOSTNAME = "128.148.140.22"
USERNAME = "user"
PASSWORD  = "bigbubbabigbubba"

cfg = SpotRobotConfig(
    hostname=HOSTNAME,
    username=USERNAME,
    password=PASSWORD,
    image_sources=[],   # skip cameras, we don't need them here
    image_width=640,
    image_height=480,
)

robot = SpotRobot(cfg)

try:
    robot.connect(calibrate=False)

    state = robot._state_client.get_robot_state()
    tf = state.kinematic_state.transforms_snapshot

    frames = list(tf.child_to_parent_edge_map.keys())
    print(f"\nFound {len(frames)} frames:\n")
    for f in sorted(frames):
        edge = tf.child_to_parent_edge_map[f]
        t = edge.parent_tform_child
        print(f"  '{f}'  parent='{edge.parent_frame_name}'  "
              f"pos=({t.position.x:.3f}, {t.position.y:.3f}, {t.position.z:.3f})")

    # Suggest which one to use
    print("\n--- Suggestion ---")
    candidates = [f for f in frames if f in ("body", "flat_body", "odom", "vision")]
    for c in candidates:
        edge = tf.child_to_parent_edge_map[c]
        t = edge.parent_tform_child
        print(f"  '{c}' -> pos=({t.position.x:.3f}, {t.position.y:.3f})  "
              f"(non-zero = good candidate)")

finally:
    robot.disconnect_keep_powered()



