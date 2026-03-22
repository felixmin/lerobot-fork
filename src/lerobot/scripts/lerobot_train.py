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
import datetime as dt
import logging
import time
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

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
    inside_slurm,
)


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
    action_loss_numerator = 0.0
    action_loss_denominator = 0.0
    latent_loss_numerator = 0.0
    latent_loss_denominator = 0.0

    for output_dict in output_dicts:
        for key, value in output_dict.items():
            if key.startswith("_"):
                continue
            if key in {"action_loss", "latent_loss"}:
                continue
            if isinstance(value, (int, float)):
                output_sums[key] = output_sums.get(key, 0.0) + float(value)
                output_counts[key] = output_counts.get(key, 0) + 1
        action_samples += float(output_dict.get("batch_action_supervised_samples", 0.0))
        action_denominator += float(output_dict.get("_action_supervised_denominator", 0.0))
        latent_samples += float(output_dict.get("batch_latent_supervised_samples", 0.0))
        latent_denominator += float(output_dict.get("_latent_supervised_denominator", 0.0))
        action_loss_denom_local = float(output_dict.get("_action_loss_denominator_exact", 0.0))
        latent_loss_denom_local = float(output_dict.get("_latent_loss_denominator_exact", 0.0))
        if "action_loss" in output_dict and action_loss_denom_local > 0.0:
            action_loss_numerator += float(output_dict["action_loss"]) * action_loss_denom_local
            action_loss_denominator += action_loss_denom_local
        if "latent_loss" in output_dict and latent_loss_denom_local > 0.0:
            latent_loss_numerator += float(output_dict["latent_loss"]) * latent_loss_denom_local
            latent_loss_denominator += latent_loss_denom_local

    merged.update({k: output_sums[k] / max(1, output_counts[k]) for k in output_sums})
    if action_loss_denominator > 0.0:
        merged["action_loss"] = action_loss_numerator / action_loss_denominator
    if latent_loss_denominator > 0.0:
        merged["latent_loss"] = latent_loss_numerator / latent_loss_denominator
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


def _gather_accumulation_denominator_totals(
    *,
    accelerator: Accelerator,
    action_total: float,
    latent_total: float,
) -> tuple[float, float]:
    totals = torch.tensor(
        [action_total, latent_total],
        device=accelerator.device,
        dtype=torch.float32,
    )
    gathered = accelerator.gather(totals)
    if gathered.ndim == 1:
        return float(gathered[0].item()), float(gathered[1].item())
    summed = gathered.reshape(-1, 2).sum(dim=0)
    return float(summed[0].item()), float(summed[1].item())


def _build_exact_scaled_loss(
    *,
    output_dict: dict[str, Any],
    action_total: float,
    latent_total: float,
    action_weight: float,
    latent_weight: float,
) -> torch.Tensor:
    scaled_loss: torch.Tensor | None = None

    action_loss = output_dict.get("_action_loss_tensor")
    action_denom = float(output_dict.get("_action_loss_denominator_exact", 0.0))
    if isinstance(action_loss, torch.Tensor) and action_denom > 0.0 and action_total > 0.0:
        scaled_loss = action_loss * (action_weight * action_denom / action_total)

    latent_loss = output_dict.get("_latent_loss_tensor")
    latent_denom = float(output_dict.get("_latent_loss_denominator_exact", 0.0))
    if isinstance(latent_loss, torch.Tensor) and latent_denom > 0.0 and latent_total > 0.0:
        term = latent_loss * (latent_weight * latent_denom / latent_total)
        scaled_loss = term if scaled_loss is None else (scaled_loss + term)

    if scaled_loss is None:
        for key in ("_action_loss_tensor", "_latent_loss_tensor"):
            tensor = output_dict.get(key)
            if isinstance(tensor, torch.Tensor):
                return tensor * 0.0
        raise RuntimeError("Exact accumulation requested but no supervised loss terms were available.")
    return scaled_loss


def _accelerator_kwargs_handlers(cfg: TrainPipelineConfig) -> list[Any]:
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs

    kwargs_handlers: list[Any] = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    if cfg.distributed_timeout_s is not None:
        kwargs_handlers.append(
            InitProcessGroupKwargs(
                timeout=dt.timedelta(seconds=float(cfg.distributed_timeout_s))
            )
        )
    return kwargs_handlers


def _compute_loss_and_output_dict(
    policy: PreTrainedPolicy,
    batch: Any,
    accelerator: Accelerator,
    rabc_weights_provider=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Computes the microbatch loss and raw output dict for one forward pass.
    """
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = (
            rabc_weights_provider.compute_batch_weights(batch)
        )

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        if rabc_batch_weights is not None:
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (
                rabc_batch_weights.sum() + epsilon
            )
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)
    return loss, output_dict


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    dl_iter,
    preprocessor,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict[str, float]]:
    """Performs one optimizer update, including internal gradient accumulation when configured."""
    start_time = time.perf_counter()
    policy.train()
    optimizer.zero_grad()

    unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    supports_exact_accumulation = all(
        has_method(unwrapped_policy, name)
        for name in ("begin_training_step", "end_training_step", "get_accumulation_denominators")
    )

    dataloading_total_s = 0.0
    prefetched_batches: list[Any] = []
    prefetched_denominators: list[dict[str, float]] = []
    microbatch_output_dicts: list[dict[str, Any]] = []
    logged_loss_value = 0.0
    grad_norm_value = 0.0

    accum_steps = max(1, int(accelerator.gradient_accumulation_steps))
    if supports_exact_accumulation:
        unwrapped_policy.begin_training_step()
    try:
        for _ in range(accum_steps):
            data_start = time.perf_counter()
            batch = next(dl_iter)
            batch = preprocessor(batch)
            dataloading_total_s += time.perf_counter() - data_start
            prefetched_batches.append(batch)
            if supports_exact_accumulation:
                prefetched_denominators.append(unwrapped_policy.get_accumulation_denominators(batch))

        action_total = 0.0
        latent_total = 0.0
        if supports_exact_accumulation:
            action_total = sum(item["action"] for item in prefetched_denominators)
            latent_total = sum(item["latent"] for item in prefetched_denominators)
            if accelerator.num_processes > 1:
                action_total, latent_total = _gather_accumulation_denominator_totals(
                    accelerator=accelerator,
                    action_total=action_total,
                    latent_total=latent_total,
                )

        total_loss_value = 0.0
        for batch in prefetched_batches:
            with accelerator.accumulate(policy):
                loss, output_dict_local = _compute_loss_and_output_dict(
                    policy,
                    batch,
                    accelerator=accelerator,
                    rabc_weights_provider=rabc_weights_provider,
                )
                if supports_exact_accumulation:
                    scaled_loss = _build_exact_scaled_loss(
                        output_dict=output_dict_local,
                        action_total=action_total,
                        latent_total=latent_total,
                        action_weight=float(getattr(unwrapped_policy.config, "action_loss_weight", 1.0)),
                        latent_weight=float(getattr(unwrapped_policy.config, "latent_loss_weight", 1.0)),
                    )
                    total_loss_value += float(scaled_loss.detach().item())
                    accelerator.backward(scaled_loss)
                else:
                    total_loss_value += float(loss.item())
                    accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if grad_clip_norm > 0:
                        grad_norm = accelerator.clip_grad_norm_(
                            policy.parameters(), grad_clip_norm
                        )
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            policy.parameters(), float("inf"), error_if_nonfinite=False
                        )
                    grad_norm_value = float(grad_norm.item())
                    optimizer.step()
                    optimizer.zero_grad()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    if has_method(unwrapped_policy, "update"):
                        unwrapped_policy.update()

            if output_dict_local:
                stored_output_dict = dict(output_dict_local)
                stored_output_dict.pop("_action_loss_tensor", None)
                stored_output_dict.pop("_latent_loss_tensor", None)
                microbatch_output_dicts.append(stored_output_dict)

        if not accelerator.sync_gradients:
            raise RuntimeError("Gradient accumulation window ended without an optimizer step.")
        logged_loss_value = total_loss_value if supports_exact_accumulation else (total_loss_value / float(accum_steps))
    finally:
        if supports_exact_accumulation:
            unwrapped_policy.end_training_step()

    train_metrics.loss = logged_loss_value
    train_metrics.grad_norm = grad_norm_value
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    train_metrics.dataloading_s = dataloading_total_s
    return train_metrics, _merge_microbatch_output_dicts(microbatch_output_dicts)


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
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            gradient_accumulation_steps=max(1, int(getattr(cfg, "grad_accum_steps", 1))),
            step_scheduler_with_optimizer=False,
            kwargs_handlers=_accelerator_kwargs_handlers(cfg),
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
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
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
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            dl_iter,
            preprocessor,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
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

    if is_main_process:
        progbar.close()

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
