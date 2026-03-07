# LeRobot Spot

## Environment

- Conda environment: `lerobot-spot`
- Activate with: `conda activate lerobot-spot`

## Workflow

1. **Record** teleop episodes: `spot_pose_keyboard_record.py`
2. **Train** ACT policy: `lerobot-train --policy.type=act ...`
3. **Replay** trained policy on robot: `spot_policy_replay.py`

## Key paths

- Training output: `outputs/train/spot-act/checkpoints/last/pretrained_model/`
- Dataset root pattern: `data/<repo-id>/`

## Policy loading pattern

```python
from lerobot.policies.act.modeling_act import ACTPolicy  # registers 'act'
from lerobot.configs.policies import PreTrainedConfig
policy_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
```
`ACTConfig.from_pretrained()` fails due to `type` field in config.json — use `PreTrainedConfig` instead.
