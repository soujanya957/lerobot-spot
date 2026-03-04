# test_spot_lerobot.py
"""
LeRobot integration test for SpotRobot.

Tests (all with zero/safe actions — Spot will NOT move):
  1. Feature spec:    observation_features and action_features match LeRobot expectations.
  2. Observation:     get_observation() returns keys + types matching observation_features.
  3. Action:          send_action() accepts action_features-shaped dict, returns clamped dict.
  4. Feature shape:   observation tensors can be converted to torch (what LeRobot does internally).
  5. Dataset record:  Simulates one episode of LeRobot-style data collection (obs -> action loop).
  6. Policy rollout:  Simulates what LeRobot does during inference (feed obs, get action, send).
"""

import time
import traceback
import numpy as np
from lerobot_robot_spot import SpotRobot, SpotRobotConfig

HOSTNAME = "128.148.140.22"
USERNAME = "user"
PASSWORD = "bigbubbabigbubba"
IMAGE_SOURCES = [
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "hand_color_image",
]

# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"{status} {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return condition

# ── test 1: feature spec ─────────────────────────────────────────────────────

def test_feature_spec(robot: SpotRobot):
    section("TEST 1: Feature Spec")

    obs_ft = robot.observation_features
    act_ft = robot.action_features

    check("observation_features is a dict", isinstance(obs_ft, dict))
    check("action_features is a dict",      isinstance(act_ft, dict))
    check("observation_features non-empty", len(obs_ft) > 0,
          f"{len(obs_ft)} keys")
    check("action_features non-empty",      len(act_ft) > 0,
          f"{len(act_ft)} keys")

    # Base keys must exist
    for key in ("base.pos_x", "base.pos_y", "base.yaw",
                "base.vel_x", "base.vel_y", "base.vel_yaw"):
        check(f"obs has {key}", key in obs_ft)

    for key in ("base.vx", "base.vy", "base.vyaw"):
        check(f"act has {key}", key in act_ft)

    # Arm pose keys must be present in both obs and act
    arm_obs = [k for k in obs_ft if k.startswith("arm.")]
    arm_act = [k for k in act_ft if k.startswith("arm.")]
    check("arm pose obs keys present", len(arm_obs) > 0, str(arm_obs))
    check("arm pose act keys present", len(arm_act) > 0, str(arm_act))
    check("arm obs and act keys match", set(arm_obs) == set(arm_act))

    # Camera keys
    non_cam = set(arm_obs) | {
        "base.pos_x", "base.pos_y", "base.yaw",
        "base.vel_x", "base.vel_y", "base.vel_yaw",
    }
    cam_keys = [k for k in obs_ft if k not in non_cam]
    check("camera keys present in obs", len(cam_keys) > 0, str(cam_keys))

    print(f"\n  obs keys  ({len(obs_ft)}): {sorted(obs_ft.keys())}")
    print(f"  act keys  ({len(act_ft)}): {sorted(act_ft.keys())}")

# ── test 2: observation ───────────────────────────────────────────────────────

def test_observation(robot: SpotRobot):
    section("TEST 2: Observation")

    obs = robot.get_observation()
    obs_ft = robot.observation_features

    check("get_observation() returns dict", isinstance(obs, dict))
    check("obs has same keys as observation_features",
          set(obs.keys()) == set(obs_ft.keys()),
          f"extra={set(obs.keys())-set(obs_ft.keys())}  "
          f"missing={set(obs_ft.keys())-set(obs.keys())}")

    all_floats_ok = True
    for k, expected_type in obs_ft.items():
        if expected_type is float:
            ok = isinstance(obs[k], (int, float)) and not np.isnan(obs[k])
            if not ok:
                print(f"{FAIL} scalar obs[{k}] = {obs[k]!r}")
                all_floats_ok = False
    check("all scalar obs values are finite floats", all_floats_ok)

    all_images_ok = True
    for k in obs_ft:
        if isinstance(obs_ft[k], tuple):  # camera feature
            v = obs[k]
            if v is None:
                print(f"{FAIL} image obs[{k}] is None")
                all_images_ok = False
            elif not (isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] == 3):
                print(f"{FAIL} image obs[{k}] bad shape {getattr(v,'shape',type(v))}")
                all_images_ok = False
    check("all camera obs are (H,W,3) uint8 arrays", all_images_ok)

    print("\n  Sample obs values:")
    for k in sorted(obs.keys()):
        v = obs[k]
        if hasattr(v, "shape"):
            print(f"    {k}: array {v.shape} dtype={v.dtype}")
        else:
            print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v!r}")

    return obs

# ── test 3: action ────────────────────────────────────────────────────────────

def test_action(robot: SpotRobot, obs: dict):
    section("TEST 3: send_action (safe zero-velocity hold)")

    act_ft = robot.action_features
    safe_action = {}
    for k in act_ft:
        if k.startswith("arm."):
            obs_key = k  # arm pose keys are identical in obs and act
            safe_action[k] = float(obs.get(obs_key, 0.0))
        else:
            safe_action[k] = 0.0  # zero base velocity

    sent = robot.send_action(safe_action)

    check("send_action returns dict",    isinstance(sent, dict))
    check("sent has base.vx",           "base.vx" in sent)
    check("sent has arm keys",          any(k.startswith("arm.") for k in sent))
    check("base velocities are zero",
          sent["base.vx"] == 0.0 and sent["base.vy"] == 0.0 and sent["base.vyaw"] == 0.0)

    print("\n  Sent action (after clamping):")
    for k in sorted(sent.keys()):
        print(f"    {k}: {sent[k]:.6f}")

    return sent

# ── test 4: torch tensor conversion ──────────────────────────────────────────

def test_torch_conversion(robot: SpotRobot, obs: dict):
    section("TEST 4: Torch Tensor Conversion (LeRobot internal format)")
    try:
        import torch

        obs_ft = robot.observation_features
        tensors = {}

        for k, v in obs.items():
            if isinstance(obs_ft.get(k), tuple):  # image
                # LeRobot normalises images to (C, H, W) float32 in [0, 1]
                arr = v.astype(np.float32) / 255.0
                tensors[k] = torch.from_numpy(arr).permute(2, 0, 1)
            else:
                tensors[k] = torch.tensor(v, dtype=torch.float32)

        check("all obs keys converted to tensors", len(tensors) == len(obs))

        for k, t in tensors.items():
            if isinstance(obs_ft.get(k), tuple):
                ok = t.shape[0] == 3 and t.dtype == torch.float32
                check(f"image {k} is (3,H,W) float32", ok, str(t.shape))
            else:
                ok = t.shape == torch.Size([]) and t.dtype == torch.float32
                check(f"scalar {k} is scalar float32", ok, str(t.shape))

        print("\n  Tensor shapes:")
        for k in sorted(tensors.keys()):
            print(f"    {k}: {tuple(tensors[k].shape)}  dtype={tensors[k].dtype}")

    except ImportError:
        print(f"{SKIP} torch not installed — skipping tensor test")

# ── test 5: dataset record simulation ────────────────────────────────────────

def test_dataset_record_sim(robot: SpotRobot, obs: dict, n_steps: int = 3):
    section(f"TEST 5: Dataset Record Simulation ({n_steps} steps)")
    """
    Mimics what lerobot/scripts/control_robot.py does when recording a dataset:
        for each step:
            obs  = robot.get_observation()
            act  = <policy or human telop>
            sent = robot.send_action(act)
            dataset.add_frame(obs, sent)
    We use a zero/hold action (safe) and just verify the loop runs cleanly.
    """
    episode = []
    act_ft  = robot.action_features

    try:
        for step in range(n_steps):
            t0 = time.time()
            obs_step = robot.get_observation()

            safe_act = {k: float(obs_step.get(k, 0.0)) if k.startswith("arm.")
                        else 0.0 for k in act_ft}
            sent_act = robot.send_action(safe_act)

            episode.append({"obs": obs_step, "action": sent_act})
            dt = time.time() - t0
            print(f"  step {step+1}/{n_steps}  loop_time={dt*1000:.1f} ms")

        check("all steps completed",        len(episode) == n_steps)
        check("each frame has obs + action",
              all("obs" in f and "action" in f for f in episode))
        check("obs keys consistent across steps",
              len({frozenset(f["obs"].keys()) for f in episode}) == 1)

    except Exception as e:
        print(f"{FAIL} dataset record sim crashed: {e}")
        traceback.print_exc()

# ── test 6: policy rollout simulation ────────────────────────────────────────

def test_policy_rollout_sim(robot: SpotRobot, n_steps: int = 3):
    section(f"TEST 6: Policy Rollout Simulation ({n_steps} steps)")
    """
    Mimics what lerobot/scripts/control_robot.py does during inference:
        obs_dict  = robot.get_observation()
        obs_tensor = stack_and_normalise(obs_dict)   # we fake this
        action    = policy(obs_tensor)               # we fake with zeros
        robot.send_action(action)
    """
    try:
        import torch

        act_ft = robot.action_features

        def fake_policy(obs_tensors: dict) -> dict:
            """Returns zero/hold action — safe stand-still."""
            obs_live = robot.get_observation()
            return {k: float(obs_live.get(k, 0.0)) if k.startswith("arm.")
                    else 0.0 for k in act_ft}

        for step in range(n_steps):
            t0 = time.time()

            # 1. get obs
            obs = robot.get_observation()

            # 2. convert to tensors (as LeRobot would)
            obs_tensors = {}
            obs_ft = robot.observation_features
            for k, v in obs.items():
                if isinstance(obs_ft.get(k), tuple):
                    arr = v.astype(np.float32) / 255.0
                    obs_tensors[k] = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                else:
                    obs_tensors[k] = torch.tensor([[v]], dtype=torch.float32)

            # 3. fake policy inference
            action = fake_policy(obs_tensors)

            # 4. send action
            sent = robot.send_action(action)

            dt = time.time() - t0
            print(f"  step {step+1}/{n_steps}  loop_time={dt*1000:.1f} ms  "
                  f"base.vx={sent['base.vx']:.3f}")

        check("policy rollout completed all steps", True)

    except ImportError:
        print(f"{SKIP} torch not installed — skipping rollout test")
    except Exception as e:
        print(f"{FAIL} policy rollout sim crashed: {e}")
        traceback.print_exc()

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = SpotRobotConfig(
        hostname=HOSTNAME,
        username=USERNAME,
        password=PASSWORD,
        image_sources=IMAGE_SOURCES,
        image_width=640,
        image_height=480,
    )

    robot = SpotRobot(cfg)

    try:
        print("Connecting to Spot...")
        robot.connect()
        print(f"Connected: {robot.is_connected}")
        print(f"Calibrated: {robot.is_calibrated}")

        test_feature_spec(robot)
        obs  = test_observation(robot)
        sent = test_action(robot, obs)
        test_torch_conversion(robot, obs)
        test_dataset_record_sim(robot, obs, n_steps=3)
        test_policy_rollout_sim(robot, n_steps=3)

        section("SUMMARY")
        print("  All tests ran. Check [PASS]/[FAIL] lines above.")
        print("  If all green: SpotRobot is ready for LeRobot dataset recording and policy rollout.")

    finally:
        print("\nDisconnecting (keeping Spot powered)...")
        robot.disconnect_keep_powered()
        print("Done.")


if __name__ == "__main__":
    main()

