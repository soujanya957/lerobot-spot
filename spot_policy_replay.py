#!/usr/bin/env python3
"""
Run a trained LeRobot policy on the real Spot robot.

Usage example:
  python3 spot_policy_replay.py \
    --hostname 128.148.138.22 \
    --username user \
    --password bigbubbabigbubba \
    --pretrained-path outputs/train/spot-act/checkpoints/last/pretrained_model \
    --dataset-root data/yourname/spot-pose-demo-images_20260304_113713 \
    --episode-time-s 30

Controls:
  Ctrl+C : stop
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: F401 — registers 'act' type
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

from lerobot_robot_spot import SpotRobot, SpotRobotConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a trained LeRobot policy on Spot")
    p.add_argument("--hostname", required=True)
    p.add_argument("--username", required=True)
    p.add_argument("--password", required=True)
    p.add_argument(
        "--image-sources",
        nargs="+",
        default=["frontleft_fisheye_image", "frontright_fisheye_image", "hand_color_image"],
    )
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument(
        "--pretrained-path",
        required=True,
        help="Path to pretrained model dir, e.g. outputs/train/spot-act/checkpoints/last/pretrained_model",
    )
    p.add_argument(
        "--dataset-path",
        required=True,
        help="Full path to training dataset, e.g. data/yourname/spot-pose-demo-images_20260304_113713",
    )
    p.add_argument("--device", default="cuda", help="Inference device: cuda or cpu")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--episode-time-s", type=float, default=30.0)
    p.add_argument("--task", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load policy ────────────────────────────────────────────────────────────
    print(f"Loading policy from {args.pretrained_path} ...")
    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device

    from pathlib import Path
    dataset_path = Path(args.dataset_path)
    # repo_id is the last two path components: e.g. "yourname/spot-pose-demo-images_20260304_113713"
    repo_id = "/".join(dataset_path.parts[-2:])
    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_path)

    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    device = get_safe_torch_device(args.device)
    ds_features = ds_meta.features

    print(f"Policy loaded. Action dim: {ds_features[ACTION]['shape']}")

    # ── Connect robot ──────────────────────────────────────────────────────────
    cfg = SpotRobotConfig(
        hostname=args.hostname,
        username=args.username,
        password=args.password,
        image_sources=args.image_sources,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    robot = SpotRobot(cfg)
    print("Connecting to Spot ...")
    robot.connect()
    print(f"Connected: {robot.is_connected}")

    input(
        "\nPolicy will control Spot for {:.0f}s.\n"
        "Keep e-stop ready.\n"
        "Press Enter to start (Ctrl+C to abort)...".format(args.episode_time_s)
    )

    loop_dt = 1.0 / float(args.fps)

    try:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        t0 = time.perf_counter()
        step = 0

        while time.perf_counter() - t0 < args.episode_time_s:
            t_start = time.perf_counter()

            # Get observation and build frame
            obs = robot.get_observation()
            obs_frame = build_dataset_frame(ds_features, obs, OBS_STR)

            # Run policy inference (disable AMP — float16 can overflow → NaN)
            with torch.no_grad():
                action_tensor = predict_action(
                    observation=obs_frame,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=False,
                    task=args.task,
                    robot_type=robot.name,
                )

            if step == 0:
                print(f"[debug] raw action tensor: {action_tensor}")
                for k, v in obs_frame.items():
                    print(f"[debug] obs_frame[{k}]: shape={v.shape}, dtype={v.dtype}")

            if torch.any(torch.isnan(action_tensor)):
                print(f"[warn] NaN in action at step {step}, skipping")
                continue

            # Convert tensor → robot action dict
            action_dict = make_robot_action(action_tensor, ds_features)

            # Send to robot
            robot.send_action(action_dict)

            step += 1
            elapsed = time.perf_counter() - t_start
            sleep_t = loop_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

            if step % (args.fps * 5) == 0:
                print(
                    f"t={time.perf_counter()-t0:.1f}s  "
                    f"vx={action_dict.get('base.vx', 0):.3f}  "
                    f"arm.x={action_dict.get('arm.pose.x', 0):.3f}"
                )

        print("\nEpisode complete.")

    except KeyboardInterrupt:
        print("\nAborted.")

    finally:
        print("Sending zero-velocity command and disconnecting ...")
        try:
            obs = robot.get_observation()
            stop_action = {
                "base.vx": 0.0, "base.vy": 0.0, "base.vyaw": 0.0,
                **{k: float(obs[k]) for k in obs if k.startswith("arm.pose.")},
            }
            robot.send_action(stop_action)
        except Exception:
            pass
        robot.disconnect_keep_powered()
        print("Done.")


if __name__ == "__main__":
    main()
