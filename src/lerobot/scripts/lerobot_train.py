#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def _sanitize_output_dict_for_logging(output_dict: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(output_dict, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in output_dict.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (int, float)):
            sanitized[key] = value
            continue
        if torch.is_tensor(value):
            if value.numel() == 1:
                sanitized[key] = float(value.detach().cpu().item())
            continue
    return sanitized


def _merge_microbatch_output_dicts(output_dicts: list[dict[str, Any]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    if not output_dicts:
        return merged

    output_sums: dict[str, float] = {}
    output_counts: dict[str, int] = {}
    action_samples = 0.0
    action_denominator = 0.0
    latent_samples = 0.0
    latent_denominator = 0.0

    for output_dict in output_dicts:
        for key, value in output_dict.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (int, float)):
                output_sums[key] = output_sums.get(key, 0.0) + float(value)
                output_counts[key] = output_counts.get(key, 0) + 1
        action_samples += float(output_dict.get("batch_action_supervised_samples", 0.0))
        action_denominator += float(output_dict.get("_action_supervised_denominator", 0.0))
        latent_samples += float(output_dict.get("batch_latent_supervised_samples", 0.0))
        latent_denominator += float(output_dict.get("_latent_supervised_denominator", 0.0))

    merged.update({k: output_sums[k] / max(1, output_counts[k]) for k in output_sums})
    if action_denominator > 0.0:
        merged["batch_action_supervised_samples"] = action_samples
        merged["batch_action_supervised_denominator"] = action_denominator
        merged["batch_action_supervised_fraction"] = action_samples / action_denominator
    if latent_denominator > 0.0:
        merged["batch_latent_supervised_samples"] = latent_samples
        merged["batch_latent_supervised_denominator"] = latent_denominator
        merged["batch_latent_supervised_fraction"] = latent_samples / latent_denominator
    return merged


def _format_supervision_batch_log(output_dict: dict[str, Any] | None) -> str | None:
    if not isinstance(output_dict, dict):
        return None

    parts: list[str] = []
    action_fraction = output_dict.get("batch_action_supervised_fraction")
    action_samples = output_dict.get("batch_action_supervised_samples")
    action_denominator = output_dict.get("batch_action_supervised_denominator")
    if isinstance(action_fraction, (int, float)):
        if isinstance(action_samples, (int, float)) and isinstance(action_denominator, (int, float)):
            parts.append(
                f"action={float(action_fraction):.3f} ({float(action_samples):.1f}/{float(action_denominator):.1f})"
            )
        elif isinstance(action_samples, (int, float)):
            parts.append(f"action={float(action_fraction):.3f} ({float(action_samples):.1f})")
        else:
            parts.append(f"action={float(action_fraction):.3f}")

    latent_fraction = output_dict.get("batch_latent_supervised_fraction")
    latent_samples = output_dict.get("batch_latent_supervised_samples")
    latent_denominator = output_dict.get("batch_latent_supervised_denominator")
    if isinstance(latent_fraction, (int, float)):
        if isinstance(latent_samples, (int, float)) and isinstance(latent_denominator, (int, float)):
            parts.append(
                f"latent={float(latent_fraction):.3f} ({float(latent_samples):.1f}/{float(latent_denominator):.1f})"
            )
        elif isinstance(latent_samples, (int, float)):
            parts.append(f"latent={float(latent_fraction):.3f} ({float(latent_samples):.1f})")
        else:
            parts.append(f"latent={float(latent_fraction):.3f}")

    if not parts:
        return None
    return "batch_sup " + " ".join(parts)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = (
            rabc_weights_provider.compute_batch_weights(batch)
        )

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (
                rabc_batch_weights.sum() + epsilon
            )
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, _sanitize_output_dict_for_logging(output_dict)


def _to_uint8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    """Convert an image tensor [C,H,W] in [0,1] or [-1,1] to uint8 HWC."""
    img = img_chw.detach().float().cpu()
    if img.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(img.shape)}")

    if img.min().item() < 0.0:
        img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    img_hwc = img.permute(1, 2, 0).numpy()
    return (img_hwc * 255.0).round().astype(np.uint8)


def _decode_instructions(
    tokenizer: Any | None,
    token_ids: torch.Tensor | None,
    attn_mask: torch.Tensor | None,
    num_samples: int,
) -> list[str]:
    if tokenizer is None or token_ids is None:
        return [""] * num_samples

    token_ids_cpu = token_ids[:num_samples].detach().cpu()
    mask_cpu = (
        attn_mask[:num_samples].detach().cpu().bool() if attn_mask is not None else None
    )

    texts: list[str] = []
    for i in range(token_ids_cpu.shape[0]):
        ids = token_ids_cpu[i]
        if mask_cpu is not None and mask_cpu.ndim == 2:
            ids = ids[mask_cpu[i]]
        texts.append(tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip())
    while len(texts) < num_samples:
        texts.append("")
    return texts


def _build_lam_viz_table(
    cfg: TrainPipelineConfig,
    batch: dict[str, Any],
    output_dict: dict[str, Any],
    tokenizer: Any | None,
    wandb: Any,
    step: int,
) -> Any:
    """Create a small W&B table for LAM latent pretraining debugging."""
    num_samples = int(max(1, getattr(cfg, "lam_viz_num_samples", 4)))

    # Instruction text
    lang_tokens = batch.get(OBS_LANGUAGE_TOKENS)
    lang_mask = batch.get(OBS_LANGUAGE_ATTENTION_MASK)
    instructions = _decode_instructions(
        tokenizer, lang_tokens, lang_mask, num_samples=num_samples
    )

    # Images: (t=0, t=Δ) pair from lam_camera_key
    camera_key = (
        getattr(cfg.policy, "lam_camera_key", None) if cfg.policy is not None else None
    )
    frames = batch.get(camera_key) if camera_key else None
    images: list[Any] = []
    if isinstance(frames, torch.Tensor) and frames.ndim == 5:
        frames_t = frames[:num_samples]
        # Handle both [B, 2, C, H, W] and [B, 2, H, W, C]
        if frames_t.shape[-1] == 3:
            frames_t = frames_t.permute(0, 1, 4, 2, 3)
        if frames_t.shape[2] == 3 and frames_t.shape[1] >= 2:
            frames_cpu = frames_t.detach().cpu()
            for i in range(frames_cpu.shape[0]):
                img0 = _to_uint8_hwc(frames_cpu[i, 0])
                img1 = _to_uint8_hwc(frames_cpu[i, 1])
                images.append(wandb.Image(np.concatenate([img0, img1], axis=1)))

    # GT + predicted codes
    gt_codes_t = batch.get("lam_codes")
    valid_pair_t = batch.get("lam_valid_pair")
    logits_t = output_dict.get("_logits")

    gt_codes = (
        gt_codes_t[:num_samples].detach().cpu().long().tolist()
        if isinstance(gt_codes_t, torch.Tensor) and gt_codes_t.ndim >= 2
        else [
            [-1] * int(getattr(cfg.policy, "lam_code_seq_len", 4))
            for _ in range(num_samples)
        ]
    )
    valid_pair = (
        valid_pair_t[:num_samples].detach().cpu().bool().tolist()
        if isinstance(valid_pair_t, torch.Tensor) and valid_pair_t.ndim == 1
        else [True] * num_samples
    )

    pred_codes: list[list[int]] = [[-1] * len(gt_codes[0]) for _ in range(num_samples)]
    confidences: list[float] = [0.0 for _ in range(num_samples)]
    if isinstance(logits_t, torch.Tensor) and logits_t.ndim == 3:
        logits_cpu = logits_t[:num_samples].detach().cpu().float()  # [N, S, K]
        pred = logits_cpu.argmax(dim=-1)  # [N, S]
        pred_codes = pred.long().tolist()
        probs = torch.softmax(logits_cpu, dim=-1)  # [N, S, K]
        conf = probs.max(dim=-1).values.mean(dim=-1)  # [N]
        confidences = conf.detach().cpu().tolist()

    table = wandb.Table(
        columns=[
            "step",
            "instruction",
            "image_pair_t0_tΔ",
            "gt_codes",
            "pred_codes",
            "valid_pair",
            "mean_confidence",
        ]
    )

    for i in range(num_samples):
        img_cell = images[i] if i < len(images) else None
        table.add_data(
            step,
            instructions[i],
            img_cell,
            str(gt_codes[i]),
            str(pred_codes[i]),
            bool(valid_pair[i]),
            float(confidences[i]),
        )
    return table


def get_default_peft_configuration(policy_type):
    """Build a basic PEFT configuration for the given policy type assuming that we train a policy from a checkpoint."""

    common_projections = "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"

    if policy_type == "smolvla":
        return {
            "target_modules": rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))",
            "modules_to_save": [],
        }
    elif policy_type in ("pi0", "pi05"):
        return {
            "target_modules": rf"(.*\.gemma_expert\..*\.self_attn.(q|v)_proj|model\.({common_projections}))",
            "modules_to_save": [],
        }

    return {"modules_to_save": None}


def wrap_policy_in_peft_model(cfg, policy):
    from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftType, get_peft_model

    # Disable all gradients because we'll only train the parameters selected by the PEFT method.
    # Layers that should receive gradients anyway need to be listed in `modules_to_save`.
    for p in policy.parameters():
        p.requires_grad_(False)

    if not cfg.policy.pretrained_path:
        raise ValueError(
            "Training from scratch using PEFT. This is unlikely to yield good results. "
            "Supply a `policy.path` to fine-tune an existing model."
        )

    if cfg.policy.type == "smolvla" and not cfg.policy.load_vlm_weights:
        logging.warning(
            "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. Set "
            "`load_vlm_weights=True` to fine-tune the existing policy."
        )

    peft_config_policy = get_default_peft_configuration(cfg.policy.type)
    peft_config_cli = dataclasses.asdict(cfg.peft) if cfg.peft else {}
    peft_config_cli["modules_to_save"] = peft_config_cli[
        "full_training_modules"
    ]  # compatibility with PEFT
    peft_method_type = PeftType[peft_config_cli["method_type"].upper()]
    peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]

    # Handle specific CLI overrides
    for key in ["target_modules", "modules_to_save", "r"]:
        if peft_config_cli[key] is not None:
            peft_config_policy[key] = peft_config_cli[key]

    if "target_modules" not in peft_config_policy:
        raise ValueError(
            f"There is no default `target_modules` value for policy {cfg.policy.type}. Please pass it manually."
        )

    # Init method depends on the used PEFT method, your specific PEFT method
    # might not be considered here, in that case an error is raised.
    if peft_config_cli["init_type"] is not None:
        if peft_method_type == "LORA":
            peft_config_policy["init_lora_weights"] = peft_config_cli["init_type"]
        elif peft_method_type == "MISS":
            peft_config_policy["init_weights"] = peft_config_cli["init_type"]
        else:
            raise ValueError(
                f"Init type {peft_config_cli['init_type']} unknown for PEFT method {peft_method_type}."
            )

    # PEFT uses this attribute to set adapter_config.base_name_or_path which we use for loading the
    # correct base model in `make_policy` since in a PEFT loading setting we only get the path to the
    # adapter, not the base model.
    if policy.config.pretrained_path:
        policy.name_or_path = str(policy.config.pretrained_path)

    # Finally wrap the policy in a PEFT model
    policy = get_peft_model(
        policy,
        peft_config_cls(**peft_config_policy),
    )

    # Make sure that the config is tagged as using PEFT so that the loading code can take the
    # appropriate steps to use the adapter weights and the PEFT config instead of the full model weights.
    policy.config.use_peft = True

    return policy


def make_offline_dataloader(
    cfg: TrainPipelineConfig,
    dataset,
    *,
    device: torch.device,
) -> torch.utils.data.DataLoader:
    """Create the offline-training dataloader.

    Datasets that implement `build_sampler()` own the global sample order. The sampler receives a shared
    seed and must stay rank-agnostic; Accelerate shards the prepared dataloader across processes.
    """

    drop_n_last_frames = int(getattr(cfg.policy, "drop_n_last_frames", 0))
    sampler = None
    shuffle = True

    if hasattr(dataset, "build_sampler"):
        sampler = dataset.build_sampler(
            seed=0 if cfg.seed is None else int(cfg.seed),
            drop_n_last_frames=drop_n_last_frames,
        )
        shuffle = False
    elif hasattr(cfg.policy, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=True,
        )
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(
                colored("Logs will be saved locally.", "yellow", attrs=["bold"])
            )

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(
            cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs
        )

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # Convert CLI peft config to dict for overrides
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (
        cfg.policy.pretrained_path and not cfg.resume
    ) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        if not cfg.resume:
            processor_kwargs["preprocessor_overrides"]["normalizer_processor"] = {
                "stats": dataset.meta.stats,
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
            }
            postprocessor_kwargs["postprocessor_overrides"] = {
                "unnormalizer_processor": {
                    "stats": dataset.meta.stats,
                    "features": policy.config.output_features,
                    "norm_map": policy.config.normalization_mapping,
                },
            }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(
            f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}"
        )
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    # Initialize LAM teacher for latent_smol in latent mode
    lam_teacher = None
    if (
        cfg.policy.type == "latent_smol"
        and getattr(cfg.policy, "head_mode", "action") == "latent"
    ):
        from lerobot.teachers.lam_teacher import LAMTeacher, LAMTeacherConfig

        if cfg.policy.lam_checkpoint_path is None:
            raise ValueError("lam_checkpoint_path required for head_mode='latent'")

        lam_teacher = LAMTeacher(
            LAMTeacherConfig(
                checkpoint_path=cfg.policy.lam_checkpoint_path,
                device=str(device),
            )
        )
        lam_teacher.eval()
        if is_main_process:
            logging.info(
                f"LAM Teacher: K={lam_teacher.codebook_size}, S={lam_teacher.code_seq_len}"
            )

    lam_viz_enabled = (
        is_main_process
        and wandb_logger is not None
        and int(getattr(cfg, "lam_viz_freq", 0)) > 0
        and cfg.policy.type == "latent_smol"
        and getattr(cfg.policy, "head_mode", "action") == "latent"
    )
    lam_viz_wandb = wandb_logger._wandb if wandb_logger is not None else None
    lam_viz_tokenizer = None
    if lam_viz_enabled:
        try:
            # Cache the tokenizer once (used only for visualization)
            lam_viz_tokenizer = policy.model.vlm_with_expert.processor.tokenizer
        except Exception as e:
            logging.warning(f"LAM viz enabled but tokenizer lookup failed: {e}")
            lam_viz_tokenizer = None

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    num_learnable_params = sum(
        p.numel() for p in policy.parameters() if p.requires_grad
    )
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(
            colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}"
        )
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        grad_accum_steps = max(1, int(getattr(cfg, "grad_accum_steps", 1)))
        effective_bs = cfg.batch_size * grad_accum_steps * num_processes
        if grad_accum_steps > 1:
            logging.info(
                f"Effective batch size: {cfg.batch_size} x {grad_accum_steps} x {num_processes} = {effective_bs}"
            )
        else:
            logging.info(
                f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}"
            )
        logging.info(
            f"{num_learnable_params=} ({format_big_number(num_learnable_params)})"
        )
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    dataloader = make_offline_dataloader(cfg, dataset, device=device)

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    grad_accum_steps = max(1, int(getattr(cfg, "grad_accum_steps", 1)))
    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * grad_accum_steps * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    while step < cfg.steps:
        start_time = time.perf_counter()
        policy.train()

        next_step = step + 1
        lam_viz_due = (
            lam_viz_enabled
            and lam_viz_wandb is not None
            and next_step % int(getattr(cfg, "lam_viz_freq", 0)) == 0
        )
        lam_viz_table = None

        dataloading_total_s = 0.0
        output_dict = {}

        if grad_accum_steps > 1:
            microbatch_output_dicts: list[dict[str, Any]] = []
            loss_sum = 0.0

            optimizer.zero_grad()

            for micro_idx in range(grad_accum_steps):
                data_start = time.perf_counter()
                batch = next(dl_iter)
                batch = preprocessor(batch)

                # Generate LAM codes for latent_smol (only when head_mode=latent)
                if lam_teacher is not None:
                    from lerobot.teachers.lam_teacher import valid_pair_from_is_pad

                    # Disable autocast to avoid fp16/bf16 surprises with LAM
                    with (
                        torch.no_grad(),
                        torch.autocast(device_type=device.type, enabled=False),
                    ):
                        camera_key = cfg.policy.lam_camera_key
                        if camera_key not in batch:
                            image_keys = sorted(
                                [
                                    k
                                    for k in batch.keys()
                                    if k.startswith("observation.images.")
                                ]
                            )
                            raise KeyError(
                                f"LAM teacher camera key '{camera_key}' not found in batch. "
                                f"Available image keys: {image_keys}. "
                                "Set --policy.lam_camera_key=<one of the available image keys>."
                            )
                        frames = batch[camera_key]

                        # Handle both [B, 2, C, H, W] and [B, 2, H, W, C] layouts
                        if frames.shape[-1] == 3:  # [B, 2, H, W, C]
                            frames = frames.permute(0, 1, 4, 2, 3)  # -> [B, 2, C, H, W]
                        assert (
                            frames.shape[2] == 3
                        ), f"Expected C=3, got shape {frames.shape}"

                        # Default is_pad on same device as frames
                        is_pad = batch.get(
                            f"{camera_key}_is_pad",
                            torch.zeros(
                                frames.shape[:2], dtype=torch.bool, device=frames.device
                            ),
                        )

                        valid_pair = valid_pair_from_is_pad(is_pad)

                        # Resize to LAM input size (use reshape for non-contiguous safety)
                        B, T, C, H, W = frames.shape
                        resize_hw = cfg.policy.lam_resize_hw
                        frames_flat = frames.reshape(B * T, C, H, W)
                        frames_resized = F.interpolate(
                            frames_flat,
                            size=resize_hw,
                            mode="bilinear",
                            align_corners=False,
                        )
                        frames = frames_resized.reshape(B, T, C, *resize_hw)

                        # frames already on device after preprocessor (DeviceProcessorStep)
                        codes = lam_teacher.codes_from_pair(frames)  # [B, 4]

                        batch["lam_codes"] = codes
                        batch["lam_valid_pair"] = valid_pair.to(
                            codes.device
                        )  # Ensure same device

                dataloading_total_s += time.perf_counter() - data_start

                # Let accelerator handle mixed precision
                with accelerator.autocast():
                    # Get RA-BC weights if enabled
                    rabc_batch_weights = None
                    rabc_batch_stats = None
                    if rabc_weights is not None:
                        rabc_batch_weights, rabc_batch_stats = (
                            rabc_weights.compute_batch_weights(batch)
                        )

                    # Use per-sample loss when RA-BC is enabled for proper weighting
                    if rabc_batch_weights is not None:
                        per_sample_loss, output_dict_local = policy.forward(
                            batch, reduction="none"
                        )
                        epsilon = 1e-6
                        loss = (per_sample_loss * rabc_batch_weights).sum() / (
                            rabc_batch_weights.sum() + epsilon
                        )
                        output_dict_local["rabc_mean_weight"] = rabc_batch_stats[
                            "raw_mean_weight"
                        ]
                        output_dict_local["rabc_num_zero_weight"] = rabc_batch_stats[
                            "num_zero_weight"
                        ]
                        output_dict_local["rabc_num_full_weight"] = rabc_batch_stats[
                            "num_full_weight"
                        ]
                    else:
                        loss, output_dict_local = policy.forward(batch)

                if lam_viz_due and lam_viz_table is None:
                    try:
                        lam_viz_table = _build_lam_viz_table(
                            cfg=cfg,
                            batch=batch,
                            output_dict=output_dict_local,
                            tokenizer=lam_viz_tokenizer,
                            wandb=lam_viz_wandb,
                            step=next_step,
                        )
                    except Exception as e:
                        logging.warning(f"Failed to build LAM viz table: {e}")
                        lam_viz_table = None

                loss_sum += float(loss.item())
                accelerator.backward(loss / grad_accum_steps)

                if output_dict_local:
                    microbatch_output_dicts.append(output_dict_local)

            # Clip gradients if specified
            if cfg.optimizer.grad_clip_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(
                    policy.parameters(), cfg.optimizer.grad_clip_norm
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float("inf"), error_if_nonfinite=False
                )

            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if has_method(
                accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"
            ):
                accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

            train_tracker.loss = loss_sum / grad_accum_steps
            train_tracker.grad_norm = grad_norm.item()
            train_tracker.lr = optimizer.param_groups[0]["lr"]
            train_tracker.update_s = time.perf_counter() - start_time
            train_tracker.dataloading_s = dataloading_total_s

            output_dict = _merge_microbatch_output_dicts(microbatch_output_dicts)
        else:
            batch = next(dl_iter)
            batch = preprocessor(batch)

            # Generate LAM codes for latent_smol (only when head_mode=latent)
            if lam_teacher is not None:
                from lerobot.teachers.lam_teacher import valid_pair_from_is_pad

                # Disable autocast to avoid fp16/bf16 surprises with LAM
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type, enabled=False),
                ):
                    camera_key = cfg.policy.lam_camera_key
                    if camera_key not in batch:
                        image_keys = sorted(
                            [
                                k
                                for k in batch.keys()
                                if k.startswith("observation.images.")
                            ]
                        )
                        raise KeyError(
                            f"LAM teacher camera key '{camera_key}' not found in batch. "
                            f"Available image keys: {image_keys}. "
                            "Set --policy.lam_camera_key=<one of the available image keys>."
                        )
                    frames = batch[camera_key]

                    # Handle both [B, 2, C, H, W] and [B, 2, H, W, C] layouts
                    if frames.shape[-1] == 3:  # [B, 2, H, W, C]
                        frames = frames.permute(0, 1, 4, 2, 3)  # -> [B, 2, C, H, W]
                    assert (
                        frames.shape[2] == 3
                    ), f"Expected C=3, got shape {frames.shape}"

                    # Default is_pad on same device as frames
                    is_pad = batch.get(
                        f"{camera_key}_is_pad",
                        torch.zeros(
                            frames.shape[:2], dtype=torch.bool, device=frames.device
                        ),
                    )

                    valid_pair = valid_pair_from_is_pad(is_pad)

                    # Resize to LAM input size (use reshape for non-contiguous safety)
                    B, T, C, H, W = frames.shape
                    resize_hw = cfg.policy.lam_resize_hw
                    frames_flat = frames.reshape(B * T, C, H, W)
                    frames_resized = F.interpolate(
                        frames_flat,
                        size=resize_hw,
                        mode="bilinear",
                        align_corners=False,
                    )
                    frames = frames_resized.reshape(B, T, C, *resize_hw)

                    # frames already on device after preprocessor (DeviceProcessorStep)
                    codes = lam_teacher.codes_from_pair(frames)  # [B, 4]

                    batch["lam_codes"] = codes
                    batch["lam_valid_pair"] = valid_pair.to(
                        codes.device
                    )  # Ensure same device

            train_tracker.dataloading_s = time.perf_counter() - start_time

            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                accelerator=accelerator,
                lr_scheduler=lr_scheduler,
                rabc_weights_provider=rabc_weights,
            )

            if lam_viz_due and lam_viz_table is None:
                try:
                    lam_viz_table = _build_lam_viz_table(
                        cfg=cfg,
                        batch=batch,
                        output_dict=output_dict,
                        tokenizer=lam_viz_tokenizer,
                        wandb=lam_viz_wandb,
                        step=next_step,
                    )
                except Exception as e:
                    logging.warning(f"Failed to build LAM viz table: {e}")
                    lam_viz_table = None

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        if lam_viz_table is not None and lam_viz_wandb is not None:
            lam_viz_wandb.log({"train/lam_viz": lam_viz_table}, step=step)
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            supervision_log = _format_supervision_batch_log(output_dict)
            if supervision_log is not None:
                logging.info(supervision_log)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    # Filter out internal tensors (prefixed with _) from scalar logging
                    scalar_dict = {
                        k: v for k, v in output_dict.items() if not k.startswith("_")
                    }
                    wandb_log_dict.update(scalar_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)

            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(
                    cfg.output_dir, cfg.steps, step
                )
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(
                        eval_info["overall"]["video_paths"][0], step, mode="eval"
                    )

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
