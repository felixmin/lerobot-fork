#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _episode_slice(offset_start: np.ndarray, offset_end: np.ndarray, index: int) -> slice:
    start = int(offset_start[index])
    end = int(offset_end[index])
    return slice(start, end)


def _resolve_cache_files(
    *,
    cache_file: Path | None,
    cache_dir: Path | None,
    split: str,
    all_files: bool,
) -> list[Path]:
    if cache_file is not None:
        return [cache_file]
    if cache_dir is None:
        raise ValueError("Provide either cache_file or --cache-dir.")
    pattern = f"{split}_*.npz"
    files = sorted(cache_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No cache files found matching {pattern} in {cache_dir}")
    if all_files:
        return files
    return [files[-1]]


def _build_x_ticks(n: int, fps: float) -> tuple[np.ndarray, list[str]]:
    if n <= 0:
        return np.asarray([], dtype=np.int64), []
    tick_count = min(10, n)
    ticks = np.linspace(0, n - 1, num=tick_count).round().astype(np.int64)
    ticks = np.unique(ticks)
    labels = [f"{idx}\n{(float(idx) / fps):.3f}s" for idx in ticks.tolist()]
    return ticks, labels


def _load_episode_payload(payload: Any, row: int) -> dict[str, Any]:
    ep_ids = payload["episode_ids"]
    if row < 0 or row >= int(ep_ids.shape[0]):
        raise IndexError(f"episode-row out of range: {row}, total rows={int(ep_ids.shape[0])}")

    sl = _episode_slice(payload["candidate_offsets_start"], payload["candidate_offsets_end"], row)
    return {
        "episode_id": int(ep_ids[row]),
        "camera_dataset_key": str(payload["camera_dataset_key"]) if "camera_dataset_key" in payload else None,
        "camera_dataset_keys": [str(x) for x in payload["camera_dataset_keys"]]
        if "camera_dataset_keys" in payload
        else None,
        "motion_aggregate_reduce": str(payload["motion_aggregate_reduce"])
        if "motion_aggregate_reduce" in payload
        else None,
        "motion_raw": payload["motion_raw"][sl],
        "motion_smooth": payload["motion_smooth"][sl],
        "motion_raw_per_camera": payload["motion_raw_per_camera"][:, sl]
        if "motion_raw_per_camera" in payload
        else None,
        "motion_smooth_per_camera": payload["motion_smooth_per_camera"][:, sl]
        if "motion_smooth_per_camera" in payload
        else None,
        "action_score": payload["action_score"][sl],
        "keep_mask": payload["keep_mask"][sl].astype(bool),
        "motion_low": float(payload["motion_low_threshold"])
        if "motion_low_threshold" in payload
        else None,
        "motion_high": float(payload["motion_high_threshold"])
        if "motion_high_threshold" in payload
        else None,
        "action_thr": float(payload["action_threshold"])
        if "action_threshold" in payload
        else None,
        "trim_start_local": int(payload["trim_start_local"][row]),
        "trim_end_local": int(payload["trim_end_local"][row]),
    }


def _plot_one_cache(axes: np.ndarray, data: dict[str, Any], title: str, fps: float) -> None:
    motion_raw = data["motion_raw"]
    motion_smooth = data["motion_smooth"]
    motion_raw_per_camera = data["motion_raw_per_camera"]
    motion_smooth_per_camera = data["motion_smooth_per_camera"]
    camera_dataset_keys = data["camera_dataset_keys"]
    motion_aggregate_reduce = data["motion_aggregate_reduce"]
    action_score = data["action_score"]
    keep_mask = data["keep_mask"]
    motion_low = data["motion_low"]
    motion_high = data["motion_high"]
    action_thr = data["action_thr"]
    trim_start_local = data["trim_start_local"]
    trim_end_local = data["trim_end_local"]

    x = np.arange(int(motion_raw.shape[0]))

    if motion_raw_per_camera is not None and camera_dataset_keys is not None:
        for cam_idx, camera_key in enumerate(camera_dataset_keys):
            cam_label = str(camera_key).split(".")[-1]
            axes[0].plot(
                x,
                motion_raw_per_camera[cam_idx],
                linewidth=0.9,
                alpha=0.35,
                label=f"raw:{cam_label}",
            )
        if motion_smooth_per_camera is not None:
            for cam_idx, camera_key in enumerate(camera_dataset_keys):
                cam_label = str(camera_key).split(".")[-1]
                axes[0].plot(
                    x,
                    motion_smooth_per_camera[cam_idx],
                    linewidth=0.9,
                    alpha=0.6,
                    linestyle="--",
                    label=f"smooth:{cam_label}",
                )

    agg_label = "motion_smooth"
    if motion_aggregate_reduce is not None:
        agg_label = f"motion_smooth({motion_aggregate_reduce})"

    axes[0].plot(x, motion_raw, label="motion_raw(agg)", alpha=0.7, linewidth=1.1, color="tab:blue")
    axes[0].plot(x, motion_smooth, label=agg_label, linewidth=2.0, color="tab:red")
    if motion_low is not None:
        axes[0].axhline(motion_low, color="tab:purple", linestyle=":", label="motion_low")
    if motion_high is not None:
        axes[0].axhline(motion_high, color="tab:brown", linestyle=":", label="motion_high")
    axes[0].axvline(trim_start_local, color="tab:orange", linestyle="--", label="trim_start")
    axes[0].axvline(trim_end_local, color="tab:red", linestyle="--", label="trim_end")
    axes[0].set_ylabel("motion score")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[0].grid(alpha=0.2)

    finite_action = np.isfinite(action_score)
    if finite_action.any():
        axes[1].plot(x, action_score, label="action_score", color="tab:green")
        if action_thr is not None:
            axes[1].axhline(action_thr, color="tab:olive", linestyle=":", label="action_threshold")

        # Mark the two lowest action-score anchors for quick threshold debugging.
        finite_idx = np.flatnonzero(finite_action)
        finite_vals = action_score[finite_action]
        low_k = min(2, int(finite_vals.shape[0]))
        order = np.argsort(finite_vals)[:low_k]
        low_idx = finite_idx[order]
        low_vals = action_score[low_idx]
        axes[1].scatter(low_idx, low_vals, color="tab:red", s=40, zorder=5, label="lowest_action")
        for rank, (idx, val) in enumerate(zip(low_idx.tolist(), low_vals.tolist()), start=1):
            t_sec = float(idx) / float(max(1.0e-6, fps))
            offset_y = 10 if rank == 1 else -14
            axes[1].annotate(
                f"low{rank}: i={idx}, t={t_sec:.2f}s, s={val:.4f}",
                xy=(idx, val),
                xytext=(8, offset_y),
                textcoords="offset points",
                fontsize=8,
                color="tab:red",
                arrowprops={"arrowstyle": "-", "color": "tab:red", "lw": 0.8},
            )

    keep_axis = axes[1].twinx()
    keep_axis.plot(x, keep_mask.astype(np.float32), label="keep_mask", color="black", linewidth=1.2)
    keep_axis.set_ylim(-0.05, 1.05)
    keep_axis.set_yticks([0.0, 1.0])
    keep_axis.set_ylabel("keep_mask (0/1)")

    ticks, labels = _build_x_ticks(int(x.shape[0]), fps)
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("anchor index / time")
    axes[1].set_ylabel("action score")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper left", fontsize=8)
    keep_axis.legend(loc="upper right", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot action-frame-filter scores from cache .npz")
    parser.add_argument("cache_file", type=Path, nargs="?", help="Path to cache file (*.npz)")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory containing cache files. Use with --all-files to compare all cameras/sources.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Split name used when selecting files from --cache-dir.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Plot all matching <split>_*.npz files from --cache-dir.",
    )
    parser.add_argument("--episode-row", type=int, default=0, help="Row index in cached episode arrays")
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="FPS used to convert anchor index to seconds for x-axis labels.",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help=(
            "Save the plot to disk. If --save-path is not provided, uses "
            "runs/debug/action_frame_filtering_ep<episode_row>.png."
        ),
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional full output path for the plot image (png/pdf/etc). Overrides default path.",
    )
    args = parser.parse_args()

    cache_files = _resolve_cache_files(
        cache_file=args.cache_file,
        cache_dir=args.cache_dir,
        split=args.split,
        all_files=bool(args.all_files),
    )

    n_files = len(cache_files)
    fig, all_axes = plt.subplots(
        2 * n_files,
        1,
        figsize=(14, 8 * n_files),
        sharex=False,
        constrained_layout=True,
    )
    if n_files == 1:
        all_axes = np.asarray(all_axes).reshape(2,)

    for file_idx, cache_path in enumerate(cache_files):
        payload = np.load(cache_path, allow_pickle=False)
        row = int(args.episode_row)
        data = _load_episode_payload(payload, row=row)
        title = (
            (
                f"{cache_path.name} | camera={data['camera_dataset_key']} | "
                f"episode row={row}, episode_id={data['episode_id']}"
                if data["camera_dataset_key"] is not None
                else f"{cache_path.name} | episode row={row}, episode_id={data['episode_id']}"
            )
            if n_files > 1
            else (
                f"camera={data['camera_dataset_key']} | episode row={row}, episode_id={data['episode_id']}"
                if data["camera_dataset_key"] is not None
                else f"Episode row={row}, episode_id={data['episode_id']}"
            )
        )

        axes_pair = all_axes[2 * file_idx : 2 * file_idx + 2]
        _plot_one_cache(axes_pair, data, title=title, fps=float(args.fps))

    fig.set_constrained_layout_pads(h_pad=0.25, w_pad=0.1, hspace=0.2, wspace=0.1)
    output_path: Path | None = None
    if args.save_path is not None:
        output_path = args.save_path
    elif bool(args.save_plot):
        output_path = Path(f"runs/debug/action_frame_filtering_ep{int(args.episode_row)}.png")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        print(f"Saved plot to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
