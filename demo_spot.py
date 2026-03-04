# demo_spot.py
"""
LeRobot demo for Boston Dynamics Spot.

Sequence:
  1. Unstow arm (move from stow to forward-ready pose)
  2. Walk forward (~1 m)
  3. Raise arm up and out in front
  4. Open gripper   <- uses RobotCommandBuilder.claw_gripper_open_command()
  5. Close gripper  <- uses RobotCommandBuilder.claw_gripper_close_command()
  6. Stow arm       <- uses RobotCommandBuilder.arm_stow_command()  + block_until_arm_arrives
  7. Walk back to start

SAFETY: Run in a clear open area (~2 m in front). Keep e-stop ready.
"""

import time
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
from lerobot_robot_spot import SpotRobot, SpotRobotConfig

# ── config ────────────────────────────────────────────────────────────────────

HOSTNAME = "128.148.140.22"
USERNAME = "user"
PASSWORD  = "bigbubbabigbubba"

WALK_SPEED    = 0.4   # m/s  (clamped to max_vx=0.5 in SpotRobot)
WALK_DURATION = 2.5   # seconds  =>  ~1 m at 0.4 m/s
STEP_HZ       = 10    # control loop rate

# ── low-level helpers ─────────────────────────────────────────────────────────

def hold(robot, duration_sec):
    """Hold current position with zero base velocity."""
    obs = robot.get_observation()
    action = {
        "base.vx": 0.0, "base.vy": 0.0, "base.vyaw": 0.0,
        **{k: float(obs[k]) for k in obs if k.startswith("arm.")},
    }
    steps = max(1, int(duration_sec * STEP_HZ))
    for _ in range(steps):
        robot.send_action(action)
        time.sleep(1.0 / STEP_HZ)

def walk(robot, vx, duration_sec):
    """Walk at vx m/s while holding arm in current position."""
    obs = robot.get_observation()
    action = {
        "base.vx": vx, "base.vy": 0.0, "base.vyaw": 0.0,
        **{k: float(obs[k]) for k in obs if k.startswith("arm.")},
    }
    steps = max(1, int(duration_sec * STEP_HZ))
    for _ in range(steps):
        robot.send_action(action)
        time.sleep(1.0 / STEP_HZ)

def unstow_arm(robot):
    """
    Unstow arm using BD SDK arm_ready_command — the correct official unstow.
    block_until_arm_arrives waits for the motion to fully complete instead
    of a fixed sleep, so the next step never starts before the arm is ready.
    """
    cmd = RobotCommandBuilder.arm_ready_command()
    cmd_id = robot._command_client.robot_command(cmd)
    block_until_arm_arrives(robot._command_client, cmd_id, timeout_sec=5.0)

def gripper_open(robot):
    """
    Open gripper using the proper BD SDK gripper command.
    The gripper is a separate actuator — NOT a joint in arm_joint_specs.
    """
    cmd = RobotCommandBuilder.claw_gripper_open_command()
    cmd_id = robot._command_client.robot_command(cmd)
    block_until_arm_arrives(robot._command_client, cmd_id, timeout_sec=3.0)

def gripper_close(robot):
    """Close gripper using the proper BD SDK gripper command."""
    cmd = RobotCommandBuilder.claw_gripper_close_command()
    cmd_id = robot._command_client.robot_command(cmd)
    block_until_arm_arrives(robot._command_client, cmd_id, timeout_sec=3.0)

def stow_arm(robot):
    """
    Stow arm using BD SDK arm_stow_command — plans a collision-free stow path.
    block_until_arm_arrives ensures we don't walk until the arm is fully stowed.
    """
    cmd = RobotCommandBuilder.arm_stow_command()
    cmd_id = robot._command_client.robot_command(cmd)
    block_until_arm_arrives(robot._command_client, cmd_id, timeout_sec=5.0)

# ── demo sequence ─────────────────────────────────────────────────────────────

def run_demo(robot):

    print("\n[1/7]  Unstowing arm (arm_ready_command)...")
    unstow_arm(robot)
    hold(robot, 0.5)

    print("\n[2/7]  Walking forward...")
    print(f"       {WALK_SPEED} m/s x {WALK_DURATION}s = ~{WALK_SPEED*WALK_DURATION:.1f} m")
    walk(robot, vx=WALK_SPEED, duration_sec=WALK_DURATION)
    hold(robot, 0.5)

    print("\n[3/7]  Holding arm level...")
    hold(robot, 1.0)

    print("\n[4/7]  Opening gripper...")
    gripper_open(robot)
    hold(robot, 0.5)

    print("\n[5/7]  Closing gripper...")
    gripper_close(robot)
    hold(robot, 0.5)

    print("\n[6/7]  Stowing arm (arm_stow_command)...")
    stow_arm(robot)
    hold(robot, 0.5)

    print("\n[7/7]  Walking back to start...")
    walk(robot, vx=-WALK_SPEED, duration_sec=WALK_DURATION)
    hold(robot, 1.0)

    print("\nDemo complete! Spot should be back at start with arm stowed.")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = SpotRobotConfig(
        hostname=HOSTNAME,
        username=USERNAME,
        password=PASSWORD,
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
        print(f"Connected: {robot.is_connected}")

        input(
            "\nSpot will:\n"
            "  1. Unstow arm to forward-ready pose\n"
            "  2. Walk forward ~1 m\n"
            "  3. Hold arm level\n"
            "  4. Open gripper (fingers open)\n"
            "  5. Close gripper (fingers close)\n"
            "  6. Stow arm\n"
            "  7. Walk back\n\n"
            "Ensure ~2 m of clear space in front of Spot.\n"
            "Press Enter to start (Ctrl+C to abort at any time)..."
        )

        run_demo(robot)

    except KeyboardInterrupt:
        print("\nAborted — holding position...")
        try:
            hold(robot, 0.3)
        except Exception:
            pass

    finally:
        print("\nDisconnecting (keeping Spot powered)...")
        robot.disconnect_keep_powered()
        print("Done.")

if __name__ == "__main__":
    main()


