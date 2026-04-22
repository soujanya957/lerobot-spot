# lerobot-spot

## Record

```bash
python spot_pose_keyboard_record.py \
  --hostname <SPOT_IP> --username <USER> --password <PASS> 
  --repo-id <user/dataset-name> \
  --dataset-root ./datasets \
  --num-episodes 10 \
  --episode-time-s 30 \
  --fps 10
```

With cameras as images (default):

```bash
python spot_pose_keyboard_record.py \
  --hostname <SPOT_IP> --username <USER> --password <PASS> \
  --repo-id <user/dataset-name> --dataset-root ./datasets \
  --store-cameras
```

With cameras as video:

```bash
python spot_pose_keyboard_record.py \
  --hostname <SPOT_IP> --username <USER> --password <PASS> \
  --repo-id <user/dataset-name> --dataset-root ./datasets \
  --video
```

Resume existing dataset:

```bash
python spot_pose_keyboard_record.py \
  --hostname <SPOT_IP> --username <USER> --password <PASS> \
  --repo-id <user/dataset-name>
  --dataset-root ./datasets \
  --resume
```

## Train

```bash
python -m lerobot.scripts.train \
  --dataset-repo-id <user/dataset-name> \
  --policy diffusion \
  --output-dir ./outputs/train \
  --job-name spot-pose
```

From local dataset root:

```bash
python -m lerobot.scripts.train \
  --dataset-repo-id <user/dataset-name> \
  --dataset-root ./datasets \
  --policy diffusion \
  --output-dir ./outputs/train
```

## Deploy

```bash
python -m lerobot.scripts.eval \
  --pretrained-policy-name-or-path ./outputs/train/checkpoints/last \
  --env-name spot \
  --repo-id <user/dataset-name>
```
