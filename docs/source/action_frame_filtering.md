# Action-Frame Filtering

This page documents action-frame filtering in the split `lerobot-fork` repository.

The implementation lives in:

- `src/lerobot/datasets/action_frame_filtering.py`
- `src/lerobot/datasets/mixed_dataset.py`

## What it does

Action-frame filtering computes per-anchor activity scores and drops low-signal anchors before sampling.

Supported modes:

- `none`
- `motion`
- `action`
- `both`

Filtering also supports motion-based endpoint trimming (`trim_episode_ends`) and dataset-local cache files:

- `<dataset_root>/meta/hlrp_action_frame_filter_cache/<split>_<camera_tag>_<fingerprint>.npz`

## Where it is applied

In this repo structure, filtering is applied by `MixedLeRobotDataset` logical sources (mix config path).

- Global config entry: `dataset.filtering`
- Per-source override entry: `sources[i].filtering` in your mix YAML

## Feature parity in this repo

Available:

- Motion filtering (`frame_diff`, `two_stage`)
- Action filtering (norm threshold + `exclude_dims` + `delta_dims`)
- Combined mode (`both`)
- Endpoint trimming
- Per-camera aggregation (`aggregate_all_cameras`, `aggregate_reduce`)
- Cache fingerprinting and reuse
- Decision-only cache reuse (`cache=score-hit`) when score-producing params match
- Batched chunked decode (`motion.decode_chunk_size`)
- Optional next-episode decode prefetch (`motion.prefetch_next_episode`)
- Vectorized motion scoring with `motion.device=cpu|cuda` (when CUDA is available)
- TorchCodec decode path with fallback to PyAV
- Safety guard: torchcodec now auto-disables prefetch to avoid known decoder instability

Known limitations:

- `sparse_flow` is currently a stub (returns zeros), same behavior as in your previous codebase.
- Filtering is wired for mixed datasets (`dataset.mix_path`) in this repo. It is not currently applied in the single-dataset code path.
- `plot_action_frame_filter_cache.py` is available in this repo, but it is a manual debug utility (not integrated into training loops).

## Config schema

Filtering config follows this structure:

```yaml
dataset:
  filtering:
    enabled: true
    mode: both                # none | motion | action | both
    apply_at_sampling: true
    trim_episode_ends: true

    motion:
      enabled: true
      method: frame_diff      # frame_diff | sparse_flow | two_stage
      frame_gap: null         # null => inferred from requested image deltas
      decode_backend: null    # null | pyav | torchcodec
      decode_chunk_size: 16
      prefetch_next_episode: true
      prefetch_chunk_size: 16
      device: cpu             # cpu | cuda
      aggregate_all_cameras: true
      aggregate_reduce: mean  # mean | max
      resize_short_side: 224
      blur_kernel: 5
      diff_pixel_threshold: 0.03
      smoothing_window: 5
      consecutive_active_k: 3
      low_threshold: 0.01
      high_threshold: 0.02
      use_hysteresis: true

      sparse_flow:
        enabled: false
        only_on_uncertain: true
        max_corners: 200
        quality_level: 0.01
        min_distance: 5.0
        win_size: 15
        max_level: 2
        min_tracked_fraction: 0.25
        median_flow_threshold: 0.6

    action:
      enabled: true
      method: norm
      threshold: 0.02
      exclude_dims: []
      delta_dims: [6]
      chunk_size: 3
      chunk_reduce: max       # max | mean
      min_nonzero_ratio: 0.0

    cache:
      enabled: true
      reuse_if_config_unchanged: true
      force_recompute: false
```

## Runnable snippets in `lerobot-fork`

Run all commands from repository root:

```bash
cd /path/to/lam-repos/lerobot-fork
```

If your environment cannot run `pip install -e .` due network/DNS constraints,
you can still execute repo-local code by prefixing commands with:

```bash
PYTHONPATH=src
```

### 1) Import + config normalization smoke test

```bash
PYTHONPATH=src python - <<'PY'
from lerobot.datasets.action_frame_filtering import normalize_filtering_config

cfg = normalize_filtering_config(
    global_filtering={
        "enabled": True,
        "mode": "both",
        "motion": {"enabled": True, "decode_chunk_size": 16},
        "action": {"enabled": True, "threshold": 0.02},
        "cache": {"enabled": True},
    },
    source_filtering={"motion": {"prefetch_next_episode": True}},
)
print(cfg["mode"], cfg["motion"]["decode_chunk_size"], cfg["motion"]["prefetch_next_episode"])
PY
```

Expected output includes:

- `both`
- `16`
- `True`

### 2) End-to-end cache generation test (real dataset)

This validates the full scoring + cache pipeline directly in the new module.

```bash
PYTHONPATH=src python - <<'PY'
import numpy as np

from lerobot.datasets.action_frame_filtering import (
    build_action_frame_filter,
    infer_motion_frame_gap,
    normalize_filtering_config,
)
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

# Update these for your environment/dataset:
repo_id = "LSY-lab/simple_tasks_teleop_v1"
root = "cache/huggingface/lerobot"   # or your local dataset root prefix
revision = None

meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root, revision=revision)
episodes = list(meta.episodes[:3])   # small smoke subset

episode_ids = np.asarray([int(ep["episode_index"]) for ep in episodes], dtype=np.int32)
candidate_start = np.asarray([int(ep["dataset_from_index"]) for ep in episodes], dtype=np.int64)
candidate_end = np.asarray([int(ep["dataset_to_index"]) for ep in episodes], dtype=np.int64)

camera_keys = [
    key
    for key, feature in meta.features.items()
    if feature.get("dtype") in {"image", "video"}
]
if not camera_keys:
    raise RuntimeError("No camera key found in dataset features")

filtering_cfg = normalize_filtering_config(
    global_filtering={
        "enabled": True,
        "mode": "both",
        "apply_at_sampling": True,
        "trim_episode_ends": True,
        "motion": {
            "enabled": True,
            "method": "frame_diff",
            "decode_chunk_size": 16,
            "prefetch_next_episode": True,
            "aggregate_all_cameras": False,
            "aggregate_reduce": "mean",
            "low_threshold": 0.01,
            "high_threshold": 0.02,
            "use_hysteresis": True,
        },
        "action": {
            "enabled": True,
            "threshold": 0.02,
            "exclude_dims": [],
            "delta_dims": [6],
            "chunk_size": 3,
            "chunk_reduce": "max",
            "min_nonzero_ratio": 0.0,
        },
        "cache": {
            "enabled": True,
            "reuse_if_config_unchanged": True,
            "force_recompute": False,
        },
    },
    source_filtering=None,
)

request_image_deltas = (-1, 0)
frame_gap = infer_motion_frame_gap(
    request_image_deltas=request_image_deltas,
    configured_frame_gap=filtering_cfg["motion"].get("frame_gap"),
)
candidate_end = np.maximum(candidate_start, candidate_end - int(max(1, frame_gap)))

result = build_action_frame_filter(
    repo_id=repo_id,
    root=root,
    revision=revision,
    video_backend=None,
    tolerance_s=1e-4,
    request_image_deltas=request_image_deltas,
    camera_dataset_keys=(camera_keys[0],),
    camera_aggregate_reduce="mean",
    action_key="action",
    episode_ids=episode_ids,
    candidate_start=candidate_start,
    candidate_end=candidate_end,
    filtering_cfg=filtering_cfg,
    split="train",
)

print("summary:", result.summary)
print("cache:", result.cache_path)
PY
```

Run the same snippet twice:

- first run should typically report `cache: miss`
- second run should report `cache: hit` (or `score-hit` if only decision params changed)

## Practical tuning notes

- Start with `mode=motion` and stabilize motion thresholds first.
- Then switch to `mode=both` and tune action thresholds/chunk behavior.
- Use `delta_dims` for gripper/open-close channels to emphasize transitions.
- Keep `cache.force_recompute=false` for normal runs; enable only while deliberately regenerating caches.

## Stage run commands (`lerobot-train`)

Use `lerobot-train` directly with mix configs for smoke/full checks.

### Stage1 smoke test (first 20 episodes)

```bash
lerobot-train \
  --dataset.repo_id=logical/stage1_teleop_v1_smoke \
  --dataset.mix_path=/home/maxchr/repos/lam-repos/lerobot-fork/configs/mixes/stage1_teleop_v1_smoke_mix.yaml \
  --policy.type=lam_lapa \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir=outputs/train/stage1_filter_smoke \
  --job_name=stage1_filter_smoke \
  --steps=50 \
  --wandb.enable=false
```

### Stage1 full run

```bash
lerobot-train \
  --dataset.repo_id=logical/stage1_teleop_v1 \
  --dataset.mix_path=/home/maxchr/repos/lam-repos/lerobot-fork/configs/mixes/stage1_teleop_v1_mix.yaml \
  --policy.type=lam_lapa \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir=outputs/train/stage1_filter_full \
  --job_name=stage1_filter_full \
  --steps=50 \
  --wandb.enable=false
```

### Stage3 smoke test (first 20 episodes)

```bash
lerobot-train \
  --dataset.repo_id=logical/stage3_teleop_v1_smoke \
  --dataset.mix_path=/home/maxchr/repos/lam-repos/lerobot-fork/configs/mixes/stage3_teleop_v1_smoke_mix.yaml \
  --policy.type=latent_smolvla \
  --policy.training_mode=multitask \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir=outputs/train/stage3_filter_smoke \
  --job_name=stage3_filter_smoke \
  --steps=50 \
  --wandb.enable=false
```

### Stage3 full run

```bash
lerobot-train \
  --dataset.repo_id=logical/stage3_teleop_v1 \
  --dataset.mix_path=/home/maxchr/repos/lam-repos/lerobot-fork/configs/mixes/stage3_teleop_v1_mix.yaml \
  --policy.type=latent_smolvla \
  --policy.training_mode=multitask \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir=outputs/train/stage3_filter_full \
  --job_name=stage3_filter_full \
  --steps=50 \
  --wandb.enable=false
```

Notes:

- Stage1 uses `lam_lapa` and stage3 uses `latent_smolvla` in these examples.
- If your stage3 policy key is custom (e.g. `hlrp_smolvla_shared`), replace
  `--policy.type=latent_smolvla` accordingly.
- If you use a pretrained checkpoint, add `--policy.path=<repo-or-local-policy>`.
- Add your usual stage/policy/training overrides (`policy.*`, `training.*`, etc.).
- Re-running the same command typically transitions from cache recompute to cache reuse
  unless `cache.force_recompute=true`.
- TorchCodec prefetch instability guard is automatic when `decode_backend=torchcodec`.
- Smoke vs full distinction here is dataset coverage (first 20 episodes vs all episodes),
  while both use `--steps=50` for quick turnaround.

## Rank episodes and generate videos (cache-driven)

### 1) Score/rank episodes from cache keep ratio

After a training run creates cache files, rank episodes by kept-anchor ratio:

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
import numpy as np

cache_file = Path("/home/maxchr/.cache/huggingface/lerobot/LSY-lab/simple_tasks_teleop_v1/meta/hlrp_action_frame_filter_cache/train_*.npz")
matches = sorted(cache_file.parent.glob(cache_file.name))
if not matches:
    raise FileNotFoundError(f"No cache files matched: {cache_file}")
path = matches[-1]
payload = np.load(path, allow_pickle=True)

ep_ids = payload["episode_ids"].astype(int)
candidate_start = payload["candidate_offsets_start"].astype(int)
candidate_end = payload["candidate_offsets_end"].astype(int)
kept = payload["kept_counts"].astype(int)
total = np.maximum(candidate_end - candidate_start, 1)
ratio = kept / total

order = np.argsort(ratio)  # lowest kept ratio first
print(f"cache_file: {path}")
print("lowest kept-ratio episodes:")
for rank, idx in enumerate(order[:10], start=1):
    print(
        f"{rank:2d}. episode={int(ep_ids[idx])} "
        f"kept={int(kept[idx])}/{int(total[idx])} "
        f"ratio={float(ratio[idx]):.4f}"
    )
PY
```

### 2) Visualize selected ranked episodes

Use `lerobot-dataset-viz` (same capability as old docs):

```bash
# Single episode
lerobot-dataset-viz \
  --repo-id LSY-lab/simple_tasks_teleop_v1 \
  --display-compressed-images 0 \
  --mode local \
  --save 1 \
  --output-dir /tmp/lerobot_viz_simple_tasks_teleop_v1 \
  --episode-index <EPISODE_ID_FROM_RANKING>
```

```bash
# Batch several ranked episodes quickly
for ep in 9 11 7 3; do
  lerobot-dataset-viz \
    --repo-id LSY-lab/simple_tasks_teleop_v1 \
    --display-compressed-images 0 \
    --mode local \
    --save 1 \
    --output-dir /tmp/lerobot_viz_simple_tasks_teleop_v1 \
    --episode-index "${ep}"
done
```

### Plot the generated cache files

After running either preset:

```bash
python scripts/plot_action_frame_filter_cache.py \
  --cache-dir /home/maxchr/.cache/huggingface/lerobot/LSY-lab/simple_tasks_teleop_v1/meta/hlrp_action_frame_filter_cache \
  --split train \
  --all-files \
  --episode-row 0 \
  --fps 15 \
  --save-path runs/debug/action_frame_filtering_all_train_ep0.png
```

## Debug plots in this repo

The cache plotting helper has been migrated to:

- `scripts/plot_action_frame_filter_cache.py`

Examples:

```bash
# Plot a single cache file, episode row 0
python scripts/plot_action_frame_filter_cache.py \
  /path/to/train_<camera_or_allcams>_<fingerprint>.npz \
  --episode-row 0
```

```bash
# Plot all train cache files from one cache directory
python scripts/plot_action_frame_filter_cache.py \
  --cache-dir /path/to/<dataset_root>/meta/hlrp_action_frame_filter_cache \
  --split train \
  --all-files \
  --fps 15 \
  --episode-row 0 \
  --save-path runs/debug/action_frame_filtering_all_train_ep0.png
```

The plot includes:

- per-camera raw/smoothed motion traces (if available)
- aggregated motion traces + thresholds
- trim start/end markers
- action score + action threshold
- keep mask (`0/1`) overlay

## Quick cache-file discovery (Python)

If you are unsure where cache files are written, run:

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path

# Adjust this to your dataset root:
dataset_root = Path("/home/maxchr/.cache/huggingface/lerobot/LSY-lab/simple_tasks_teleop_v1")
cache_dir = dataset_root / "meta" / "hlrp_action_frame_filter_cache"

print("cache_dir:", cache_dir)
for p in sorted(cache_dir.glob("train_*.npz")):
    print(" ", p)
for p in sorted(cache_dir.glob("val_*.npz")):
    print(" ", p)
PY
```

Training logs print filtering cache usage via `[mixed-filter]` messages, including
cache path and before/after anchor counts.

## Where to change filtering config

Depending on your workflow, configure filtering in one of these places:

- Mixed-dataset training/inference: set `dataset.filtering` and optional per-source
  `sources[i].filtering` in your mix YAML.
- Global defaults in dedicated files: `configs/filtering/stage1_filtering.yaml` and
  `configs/filtering/stage3_filtering.yaml`.
- Source-specific overrides directly in mix files under `sources[i].filtering`.

Starter files in this repo:

- `configs/mixes/stage1_teleop_v1_mix.yaml`
- `configs/mixes/stage3_teleop_v1_mix.yaml`
- `configs/mixes/stage1_teleop_v1_smoke_mix.yaml`
- `configs/mixes/stage3_teleop_v1_smoke_mix.yaml`
- `configs/mixes/human_plus_teleop_template.yaml`
- `configs/mixes/human_only_motion_tuning.yaml`
- `configs/filtering/stage1_filtering.yaml`
- `configs/filtering/stage3_filtering.yaml`

For public portability, these mix files intentionally omit `root`. When `root` is
unset, dataset loading resolves to `HF_LEROBOT_HOME/<repo_id>` (default:
`~/.cache/huggingface/lerobot/<repo_id>`), reusing local cache if present and
downloading missing files otherwise.

## Full training: where to set mix + filtering

For full stage runs, there are two levels:

- dataset mixing (which datasets, weights, episode subsets) via `dataset.mix_path`
- filtering behavior via global `dataset.filtering` and/or per-source `sources[i].filtering`

`lerobot-fork` provides the mixed-dataset/filtering engine; your stage-specific training
entrypoints live in the split stage/policy repos.

### 1) Define which datasets are mixed (`mix_path`)

Create a mix file (example: `configs/mixes/stage_mix.yaml`):

```yaml
sources:
  - name: teleop_primary
    repo_id: LSY-lab/simple_tasks_teleop_v1
    root: /home/maxchr/.cache/huggingface/lerobot/LSY-lab/simple_tasks_teleop_v1
    weight: 3.0
    supervision: latent_only
    video_backend: pyav
    tolerance_s: 0.0001
    episodes: [0, 1, 2, 3]
    filtering:
      mode: both
      motion:
        device: cuda
        decode_backend: pyav

  - name: teleop_aux
    repo_id: LSY-lab/simple_tasks_teleop_v1
    root: /home/maxchr/.cache/huggingface/lerobot/LSY-lab/simple_tasks_teleop_v1
    weight: 1.0
    supervision: multitask
    video_backend: pyav
    tolerance_s: 0.0001
    exclude_episodes: [0, 1, 2, 3]
```

Notes:

- Use either `episodes` or `exclude_episodes` per source (not both).
- `weight` controls sampling share across sources.
- `sources[i].filtering` is a per-source override on top of global filtering.

### 2) Define global filtering defaults for the run

In your stage run config (the config consumed by your training command), set:

```yaml
dataset:
  mix_path: /abs/path/to/configs/mixes/stage_mix.yaml
  filtering:
    enabled: true
    mode: both
    apply_at_sampling: true
    trim_episode_ends: true
    motion:
      enabled: true
      method: frame_diff
      device: cuda
      decode_backend: pyav
      decode_chunk_size: 16
      prefetch_next_episode: true
      prefetch_chunk_size: 16
      aggregate_all_cameras: true
      aggregate_reduce: mean
      low_threshold: 0.01
      high_threshold: 0.02
      use_hysteresis: true
    action:
      enabled: true
      threshold: 0.02
      delta_dims: [6]
      chunk_size: 3
      chunk_reduce: max
    cache:
      enabled: true
      reuse_if_config_unchanged: true
      force_recompute: false
```

### 3) Run full filtering during training

The stage training run automatically filters all selected episodes from the mix config
when filtering is enabled.

## Migration scope notes

This documentation covers action-frame filtering as implemented in `lerobot-fork`.
Some old monorepo files from `robot-learning-from-video` are not migrated 1:1 because those wrappers/config trees are now split into other repositories.

Notable examples:

- old stage wrappers (`scripts/2_train_stage1_lam.py`, `scripts/6_train_lerobot.py`) are not part of `lerobot-fork`
- old config trees under `config/stage3_*` are now repo-specific in your split layout
- debug outputs under `runs/debug/*` are artifacts, not source code