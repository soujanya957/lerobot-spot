# test_spot_robot.py
"""
Minimal sanity check for SpotRobot + LeRobot integration.

What it does:
  - Connects to Spot using SpotRobotConfig.
  - Reads one observation and prints keys/shapes.
  - Sends a "safe" action:
      * base velocities = 0 (no movement)
      * arm pose = hold current pose
  - Disconnects cleanly.
"""

from lerobot_robot_spot import SpotRobot, SpotRobotConfig


def main():
    # 1) Fill in your robot's credentials and IP/hostname
    cfg = SpotRobotConfig(
        hostname="128.148.140.22",   # TODO: your Spot's IP
        username="user",          # TODO: your Spot username
        password="bigbubbabigbubba",  # TODO: your Spot password
        # Optional: override camera sources explicitly
        #

        image_sources=[
            "frontleft_fisheye_image",
            "frontright_fisheye_image",
            "hand_color_image",
        ],
        image_width=640,
        image_height=480,
    )

    robot = SpotRobot(cfg)

    try:
        print("Connecting to Spot...")
        robot.connect()
        print("Connected:", robot.is_connected)

        # 2) Get one observation
        obs = robot.get_observation()
        print("\nObservation keys:")
        for k in sorted(obs.keys()):
            v = obs[k]
            if hasattr(v, "shape"):
                print(f"  {k}: array with shape {v.shape}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")

        # 3) Build a safe action:
        #    - base stays still
        #    - arm holds current pose from observation
        safe_action = {
            "base.vx": 0.0,
            "base.vy": 0.0,
            "base.vyaw": 0.0,
        }
        for k, v in obs.items():
            if k.startswith("arm.pose."):
                safe_action[k] = float(v)

        print("\nSending safe hold-pose action...")
        sent = robot.send_action(safe_action)
        print("Action actually sent (after clamping):")
        for k in sorted(sent.keys()):
            print(f"  {k}: {sent[k]}")

        print("\nTest complete. Robot should be standing, not moving.")

    finally:
        print("\nDisconnecting...")
        robot.disconnect()
        print("Disconnected. Done.")


if __name__ == "__main__":
    main()
