#!/usr/bin/env python

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import make_dataset


def _make_default_policy() -> SimpleNamespace:
    return SimpleNamespace(
        observation_delta_indices=None,
        action_delta_indices=None,
        reward_delta_indices=None,
        image_size=None,
    )


def _build_dataset(
    *,
    mix_path: str,
    mix_implementation: str,
    tolerance_s: float,
    policy_path: str | None,
):
    policy: Any
    if policy_path is None:
        policy = _make_default_policy()
    else:
        policy = PreTrainedConfig.from_pretrained(policy_path)

    cfg = SimpleNamespace(
        dataset=DatasetConfig(
            repo_id="debug/mixed_sampler_inspection",
            mix_path=mix_path,
            mix_implementation=mix_implementation,
        ),
        policy=policy,
        tolerance_s=float(tolerance_s),
    )
    return make_dataset(cfg)


def _build_training_style_sampler(
    dataset: Any,
    *,
    seed: int,
    batch_size: int,
    drop_n_last_frames: int,
):
    if not hasattr(dataset, "build_sampler"):
        raise TypeError(
            f"Dataset {type(dataset).__name__} does not implement build_sampler()."
        )

    loader_hints = (
        dataset.loader_hints()
        if hasattr(dataset, "loader_hints") and callable(dataset.loader_hints)
        else {}
    )

    sampler_kwargs: dict[str, Any] = {
        "seed": int(seed),
        "drop_n_last_frames": int(drop_n_last_frames),
    }
    if loader_hints.get("sampler_mode") == "source_block":
        sampler_kwargs["source_block_size"] = max(1, int(batch_size))
    if loader_hints.get("pass_batch_size_to_sampler"):
        sampler_kwargs["batch_size"] = max(1, int(batch_size))

    sampler = dataset.build_sampler(**sampler_kwargs)
    return sampler, loader_hints, sampler_kwargs


def _sample_index_summary(sample_index: Any) -> dict[str, Any]:
    if isinstance(sample_index, tuple) and len(sample_index) == 2:
        return {
            "kind": "tuple",
            "source_id": int(sample_index[0]),
            "anchor_abs_index": int(sample_index[1]),
        }

    if hasattr(sample_index, "source_id") and hasattr(sample_index, "anchor_abs_index"):
        payload = {
            "kind": type(sample_index).__name__,
            "source_id": int(sample_index.source_id),
            "anchor_abs_index": int(sample_index.anchor_abs_index),
        }
        if hasattr(sample_index, "episode_id"):
            payload["episode_id"] = int(sample_index.episode_id)
        return payload

    return {"kind": type(sample_index).__name__, "index": int(sample_index)}


def _value_summary(value: Any) -> str:
    if torch.is_tensor(value):
        return f"tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, tuple):
        return f"tuple(len={len(value)})"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (int, float, bool)):
        return repr(value)
    return type(value).__name__


def _item_summary(item: dict[str, Any]) -> dict[str, str]:
    preferred_keys = [
        "dataset_source_name",
        "dataset_source_repo_id",
        "dataset_source_root",
        "dataset_source_revision",
        "dataset_source_index",
        "action_supervision",
        "latent_supervision",
        "index",
        "episode_index",
        "task",
        "action",
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
    ]
    summary: dict[str, str] = {}
    for key in preferred_keys:
        if key in item:
            summary[key] = _value_summary(item[key])

    for key in sorted(item):
        if key in summary:
            continue
        if key.endswith("_is_pad"):
            summary[key] = _value_summary(item[key])

    return summary


def _default_debug_root() -> Path:
    return Path(
        "/mnt/data/workspace/runs_root/runs_lerobot/outputs/debug/mixed_sampler_inspect"
    )


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    return _default_debug_root() / stamp


def _to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().float()
    if array.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape {tuple(array.shape)}")

    if array.shape[0] in {1, 3}:
        array = array.permute(1, 2, 0)

    out = array.numpy()
    if out.shape[-1] == 1:
        out = np.repeat(out, 3, axis=-1)

    finite = np.isfinite(out)
    if not np.any(finite):
        return np.zeros((*out.shape[:2], 3), dtype=np.uint8)

    min_value = float(np.min(out[finite]))
    max_value = float(np.max(out[finite]))
    if min_value >= 0.0 and max_value <= 1.0:
        scaled = out
    elif min_value >= 0.0 and max_value <= 255.0:
        scaled = out / 255.0
    elif max_value > min_value:
        scaled = (out - min_value) / (max_value - min_value)
    else:
        scaled = np.zeros_like(out)

    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _image_panels(item: dict[str, Any]) -> list[tuple[str, Image.Image]]:
    panels: list[tuple[str, Image.Image]] = []
    for key in ("observation.images.image", "observation.images.image2"):
        if key not in item or not torch.is_tensor(item[key]):
            continue

        value = item[key]
        if value.ndim == 3:
            panels.append((key, Image.fromarray(_to_hwc_uint8(value))))
            continue

        if value.ndim == 4:
            max_frames = min(2, int(value.shape[0]))
            for frame_idx in range(max_frames):
                panels.append(
                    (
                        f"{key}[{frame_idx}]",
                        Image.fromarray(_to_hwc_uint8(value[frame_idx])),
                    )
                )

    return panels


def _item_caption(item: dict[str, Any], sample_pos: int) -> str:
    source_name = item.get("dataset_source_name", "<missing>")
    task = item.get("task", "<missing>")
    episode_index = item.get("episode_index")
    sample_index = item.get("index")

    def _scalar_repr(value: Any) -> str:
        if torch.is_tensor(value) and value.numel() == 1:
            return str(value.item())
        return str(value)

    return (
        f"sample={sample_pos} "
        f"source={source_name} "
        f"task={task} "
        f"episode={_scalar_repr(episode_index)} "
        f"index={_scalar_repr(sample_index)}"
    )


def _save_item_plot(item: dict[str, Any], sample_pos: int, plot_dir: Path) -> Path | None:
    panels = _image_panels(item)
    if not panels:
        return None

    font = ImageFont.load_default()
    margin = 16
    label_height = 18
    title_height = 26
    panel_widths = [panel.size[0] for _, panel in panels]
    panel_heights = [panel.size[1] for _, panel in panels]
    canvas_width = margin * (len(panels) + 1) + sum(panel_widths)
    canvas_height = margin * 2 + title_height + label_height + max(panel_heights)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), _item_caption(item, sample_pos), fill=(0, 0, 0), font=font)

    x = margin
    y = margin + title_height
    for label, panel in panels:
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
        canvas.paste(panel, (x, y + label_height))
        x += panel.size[0] + margin

    output_path = plot_dir / f"sample_{sample_pos:04d}.png"
    canvas.save(output_path)
    return output_path


def _print_source_table(dataset: Any) -> None:
    print("Declared sources:")
    for source in getattr(dataset, "sources", []):
        episodes = getattr(source, "selected_episodes", None)
        if episodes is None:
            ep_text = "unknown"
        else:
            ep_text = str(len(episodes))
        print(
            "  - "
            f"name={source.name} "
            f"repo_id={source.repo_id} "
            f"weight={source.weight} "
            f"episodes={ep_text} "
            f"action_supervision={source.action_supervision} "
            f"latent_supervision={source.latent_supervision}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect how a mixed dataset sampler behaves. "
            "Loads the mix through the standard dataset factory, builds the sampler "
            "with the same loader hints training uses, and prints sampled items."
        )
    )
    parser.add_argument("mix_path", help="Path to the mix yaml file.")
    parser.add_argument(
        "--mix-implementation",
        default="current",
        choices=["legacy", "current", "compact_manifest"],
        help="Mixed dataset implementation to inspect.",
    )
    parser.add_argument(
        "--policy-path",
        default=None,
        help=(
            "Optional policy config or checkpoint path. "
            "Pass this if you want policy-specific delta timestamps and image size "
            "to match training exactly."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument(
        "--show-items",
        type=int,
        default=8,
        help="How many fetched sampled items to print in detail.",
    )
    parser.add_argument(
        "--drop-n-last-frames",
        type=int,
        default=0,
        help="Extra frames to drop at the end of each episode when building the sampler.",
    )
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional directory for the text report. "
            "If omitted, output stays on stdout unless --save-image-plots is enabled."
        ),
    )
    parser.add_argument(
        "--save-image-plots",
        action="store_true",
        help="Save one PNG per sampled item when observation image tensors are present.",
    )
    parser.add_argument(
        "--plot-items",
        type=int,
        default=None,
        help="How many sampled items to render as PNGs. Defaults to --show-items.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional plot directory. Defaults to <output_dir>/plots.",
    )
    args = parser.parse_args()

    dataset = _build_dataset(
        mix_path=args.mix_path,
        mix_implementation=args.mix_implementation,
        tolerance_s=args.tolerance_s,
        policy_path=args.policy_path,
    )
    sampler, loader_hints, sampler_kwargs = _build_training_style_sampler(
        dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        drop_n_last_frames=args.drop_n_last_frames,
    )

    sampled_indices = list(islice(iter(sampler), int(args.num_samples)))

    show_count = min(int(args.show_items), len(sampled_indices))
    plot_count = 0
    if args.save_image_plots:
        plot_count = min(
            int(args.plot_items if args.plot_items is not None else args.show_items),
            len(sampled_indices),
        )

    output_dir: Path | None = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.save_image_plots:
        output_dir = _default_output_dir()

    plot_dir: Path | None = None
    if args.save_image_plots:
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else (
            output_dir / "plots" if output_dir is not None else _default_output_dir() / "plots"
        )
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_dir = plot_dir.parent if output_dir is None else output_dir

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    saved_detail_items: list[tuple[Any, dict[str, str]]] = []
    saved_plot_paths: list[Path] = []

    def emit(line: str = "") -> None:
        print(line)
        lines.append(line)

    inspect_count = max(show_count, plot_count)
    for sample_pos, sample_index in enumerate(sampled_indices[:inspect_count]):
        item = dataset[sample_index]

        if sample_pos < show_count:
            saved_detail_items.append((sample_index, _item_summary(item)))

        if plot_dir is not None and sample_pos < plot_count:
            saved_path = _save_item_plot(item, sample_pos, plot_dir)
            if saved_path is not None:
                saved_plot_paths.append(saved_path)

    emit(f"Dataset type: {type(dataset).__name__}")
    emit(f"Num frames: {len(dataset)}")
    emit(f"Num sampled indices requested: {len(sampled_indices)}")
    emit(f"Loader hints: {loader_hints}")
    emit(f"Sampler kwargs: {sampler_kwargs}")
    emit("Declared sources:")
    for source in getattr(dataset, "sources", []):
        episodes = getattr(source, "selected_episodes", None)
        ep_text = "unknown" if episodes is None else str(len(episodes))
        emit(
            "  - "
            f"name={source.name} "
            f"repo_id={source.repo_id} "
            f"weight={source.weight} "
            f"episodes={ep_text} "
            f"action_supervision={source.action_supervision} "
            f"latent_supervision={source.latent_supervision}"
        )

    source_id_counts = Counter()
    for sample_index in sampled_indices:
        summary = _sample_index_summary(sample_index)
        if "source_id" in summary:
            source_id_counts[int(summary["source_id"])] += 1

    source_name_counts = Counter()
    if source_id_counts and hasattr(dataset, "sources"):
        for source_id, count in source_id_counts.items():
            source = dataset.sources[source_id]
            source_name_counts[str(getattr(source, "name", source_id))] += count

    emit("\nSampled source counts by name:")
    for name, count in source_name_counts.most_common():
        emit(f"  - {name}: {count}")

    if source_id_counts:
        emit("\nSampled source counts by source_id:")
        for source_id, count in sorted(source_id_counts.items()):
            emit(f"  - {source_id}: {count}")

    emit(f"\nFirst {show_count} sampled items:")
    for sample_pos, (sample_index, item_summary) in enumerate(saved_detail_items):
        emit(f"\n[{sample_pos}] sampler_index={_sample_index_summary(sample_index)}")
        for key, value in item_summary.items():
            emit(f"  {key}: {value}")

    if output_dir is not None:
        report_path = output_dir / "report.txt"
        report_path.write_text("\n".join(lines) + "\n")
        emit(f"\nSaved text report to: {report_path}")

    if plot_dir is not None:
        if saved_plot_paths:
            emit(f"Saved {len(saved_plot_paths)} image plots to: {plot_dir}")
        else:
            emit(f"No image tensors found for plotting in the first {plot_count} sampled items.")


if __name__ == "__main__":
    main()
