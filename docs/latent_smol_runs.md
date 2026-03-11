# `latent_smol` runs (Stage 1 latent pretrain + Stage 2 action finetune)

This repo is typically run with cached Hugging Face datasets/models and the `lerobhl` conda env.

## Common environment (cached HF + HLRP imports)

```bash
cd /mnt/data/workspace/code/lerobot
conda activate lerobhl

export PYTHONPATH="/mnt/data/workspace/code/high-level-robot-planner/packages:${PYTHONPATH:-}"
export HF_HOME=/mnt/data/tmp/hf
export HF_HUB_CACHE=/mnt/data/tmp/hf/hub
export HF_DATASETS_CACHE=/mnt/data/tmp/hf/datasets
export TRANSFORMERS_CACHE=/mnt/data/tmp/hf/transformers
export HF_TOKEN="$(cat /home/felix/.cache/huggingface/token)"
```

If you prefer a single command (no `conda activate`):

```bash
cd /mnt/data/workspace/code/lerobot && conda run -n lerobhl --no-capture-output bash -lc '...'
```

## Stage 1: latent pretraining (LAQ codes)

Example “real-ish” settings (effective batch size = `batch_size * grad_accum_steps * num_gpus`):

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=latent_smol \
  --policy.head_mode=latent \
  --policy.laq_checkpoint_path=/mnt/data/workspace/code/high-level-robot-planner/laq-stepstep052500.ckpt \
  --policy.lam_camera_keys=observation.images.top \
  --policy.lam_future_seconds=1.0 \
  --policy.load_vlm_weights=false \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.use_imagenet_stats=false \
  --dataset.video_backend=pyav \
  --batch_size=16 \
  --grad_accum_steps=2 \
  --steps=20000 \
  --log_freq=200 \
  --save_freq=2000 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --wandb.entity=felixmin \
  --wandb.mode=online \
  --laq_viz_freq=2000 \
  --laq_viz_num_samples=4 \
  --output_dir=outputs/latent_smol_stage1_bs16_ga2
```

W&B visualization:
- Logs a `train/laq_viz` table every `--laq_viz_freq` steps (only in `head_mode=latent`), showing instruction text + (t=0,t=Δ) image pair + GT/pred LAQ codes.

## Resume Stage 1 from the last checkpoint (keep optimizer/scheduler state)

Use the `train_config.json` inside the last checkpoint’s `pretrained_model/`:

```bash
python -m lerobot.scripts.lerobot_train \
  --resume=true \
  --config_path=outputs/<run_name>/checkpoints/last/pretrained_model/train_config.json \
  --laq_viz_freq=2000 \
  --laq_viz_num_samples=4
```

Notes:
- `--config_path` loads the saved config; CLI flags override it.
- W&B should resume the same run (same run id) when `resume=true` and the checkpoint contains the previous `wandb.run_id`.

## Stage 2: action finetuning (flow-matching)

Start a new run that *loads* the Stage 1 checkpoint weights, but trains the action head:

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=latent_smol \
  --policy.head_mode=action \
  --policy.pretrained_path=outputs/<run_name>/checkpoints/last \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.use_imagenet_stats=false \
  --dataset.video_backend=pyav \
  --batch_size=64 \
  --steps=200000 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --wandb.entity=felixmin \
  --wandb.mode=online \
  --output_dir=outputs/latent_smol_stage2_action
```

## Baseline: train `smolvla` from scratch (action mode)

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=false \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.use_imagenet_stats=false \
  --dataset.video_backend=pyav \
  --batch_size=64 \
  --steps=200000 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --wandb.entity=felixmin \
  --wandb.mode=online \
  --output_dir=outputs/smolvla_scratch_action
```

