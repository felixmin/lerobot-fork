"""Action-frame filtering utilities for LeRobot-v3 anchor sampling.

This module computes per-anchor motion/action scores, trims low-activity episode
endpoints, and materializes keep/drop masks used by training-time samplers.
Scores and decisions are cached in dataset-local sidecars with config
fingerprinting to avoid recomputation.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoDecoderCache
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.datasets.video_utils import decode_video_frames_torchcodec

logger = logging.getLogger(__name__)
_TORCHCODEC_FAILED = False


@dataclass(frozen=True)
class ActionFrameFilterResult:
    kept_anchor_values: np.ndarray
    kept_offsets_start: np.ndarray
    kept_offsets_end: np.ndarray
    kept_counts: np.ndarray
    kept_range_start: np.ndarray
    kept_range_end: np.ndarray
    cache_path: str | None
    summary: dict[str, Any]


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _as_list_of_int(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    raise TypeError(f"Expected list/tuple for integer list, got {type(value)}")


def _deep_merge_dict(
    base: Mapping[str, Any], override: Mapping[str, Any] | None
) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    if override is None:
        return out
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def normalize_filtering_config(
    *,
    global_filtering: Mapping[str, Any] | None,
    source_filtering: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge global/source filtering config into a normalized runtime schema."""
    base = {} if global_filtering is None else dict(global_filtering)
    merged = _deep_merge_dict(base, source_filtering)
    enabled = _to_bool(merged.get("enabled"), default=False)
    mode = str(merged.get("mode", "none"))
    if not enabled or mode == "none":
        return None

    motion_cfg = dict(merged.get("motion", {}))
    sparse_flow_cfg = dict(motion_cfg.get("sparse_flow", {}))
    motion_cfg["sparse_flow"] = sparse_flow_cfg

    action_cfg = dict(merged.get("action", {}))
    cache_cfg = dict(merged.get("cache", {}))

    normalized = {
        "enabled": True,
        "mode": mode,
        "apply_at_sampling": _to_bool(merged.get("apply_at_sampling"), True),
        "trim_episode_ends": _to_bool(merged.get("trim_episode_ends"), True),
        "motion": {
            "enabled": _to_bool(motion_cfg.get("enabled"), mode in {"motion", "both"}),
            "method": str(motion_cfg.get("method", "frame_diff")),
            "frame_gap": motion_cfg.get("frame_gap"),
            "decode_backend": motion_cfg.get("decode_backend"),
            "decode_chunk_size": _to_int(motion_cfg.get("decode_chunk_size"), 16),
            "prefetch_next_episode": _to_bool(
                motion_cfg.get("prefetch_next_episode"), False
            ),
            "prefetch_chunk_size": _to_int(motion_cfg.get("prefetch_chunk_size"), 16),
            "device": str(motion_cfg.get("device", "cpu")),
            "aggregate_all_cameras": _to_bool(
                motion_cfg.get("aggregate_all_cameras"), False
            ),
            "aggregate_reduce": str(motion_cfg.get("aggregate_reduce", "mean")),
            "resize_short_side": _to_int(motion_cfg.get("resize_short_side"), 224),
            "grayscale": _to_bool(motion_cfg.get("grayscale"), True),
            "blur_kernel": _to_int(motion_cfg.get("blur_kernel"), 5),
            "diff_pixel_threshold": _to_float(
                motion_cfg.get("diff_pixel_threshold"), 0.03
            ),
            "smoothing_window": _to_int(motion_cfg.get("smoothing_window"), 5),
            "consecutive_active_k": _to_int(motion_cfg.get("consecutive_active_k"), 3),
            "low_threshold": _to_float(motion_cfg.get("low_threshold"), 0.01),
            "high_threshold": _to_float(motion_cfg.get("high_threshold"), 0.02),
            "use_hysteresis": _to_bool(motion_cfg.get("use_hysteresis"), True),
            "sparse_flow": {
                "enabled": _to_bool(sparse_flow_cfg.get("enabled"), False),
                "only_on_uncertain": _to_bool(
                    sparse_flow_cfg.get("only_on_uncertain"), True
                ),
                "max_corners": _to_int(sparse_flow_cfg.get("max_corners"), 200),
                "quality_level": _to_float(sparse_flow_cfg.get("quality_level"), 0.01),
                "min_distance": _to_float(sparse_flow_cfg.get("min_distance"), 5.0),
                "win_size": _to_int(sparse_flow_cfg.get("win_size"), 15),
                "max_level": _to_int(sparse_flow_cfg.get("max_level"), 2),
                "min_tracked_fraction": _to_float(
                    sparse_flow_cfg.get("min_tracked_fraction"), 0.25
                ),
                "median_flow_threshold": _to_float(
                    sparse_flow_cfg.get("median_flow_threshold"), 0.6
                ),
            },
        },
        "action": {
            "enabled": _to_bool(action_cfg.get("enabled"), mode in {"action", "both"}),
            "method": str(action_cfg.get("method", "norm")),
            "threshold": _to_float(action_cfg.get("threshold"), 0.02),
            "exclude_dims": _as_list_of_int(action_cfg.get("exclude_dims")),
            "delta_dims": _as_list_of_int(action_cfg.get("delta_dims")),
            "chunk_size": _to_int(action_cfg.get("chunk_size"), 1),
            "chunk_reduce": str(action_cfg.get("chunk_reduce", "max")),
            "min_nonzero_ratio": _to_float(action_cfg.get("min_nonzero_ratio"), 0.0),
        },
        "cache": {
            "enabled": _to_bool(cache_cfg.get("enabled"), True),
            "reuse_if_config_unchanged": _to_bool(
                cache_cfg.get("reuse_if_config_unchanged"), True
            ),
            "force_recompute": _to_bool(cache_cfg.get("force_recompute"), False),
        },
    }
    return normalized


def infer_motion_frame_gap(
    *,
    request_image_deltas: tuple[int, ...],
    configured_frame_gap: int | None,
) -> int:
    """Infer temporal gap used for motion scoring from requested image deltas."""
    if configured_frame_gap is not None:
        return max(1, int(configured_frame_gap))
    if len(request_image_deltas) == 0:
        return 1
    return max(1, int(max(request_image_deltas) - min(request_image_deltas)))


def _to_gray_float_pair(
    pair: torch.Tensor,
    *,
    resize_short_side: int,
    blur_kernel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pair.ndim != 4:
        raise ValueError(
            f"Expected image pair [T,C,H,W] or [T,H,W,C], got {tuple(pair.shape)}"
        )
    if int(pair.shape[0]) != 2:
        raise ValueError(
            f"Expected exactly 2 time steps for pair, got T={int(pair.shape[0])}"
        )

    if int(pair.shape[1]) == 3:
        x = pair
    elif int(pair.shape[-1]) == 3:
        x = pair.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            f"Unsupported pair layout for RGB conversion: {tuple(pair.shape)}"
        )

    x = x.to(torch.float32)
    if x.max() > 1.0:
        x = x / 255.0

    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    h = int(gray.shape[-2])
    w = int(gray.shape[-1])
    short = min(h, w)
    if resize_short_side > 0 and short != resize_short_side:
        scale = float(resize_short_side) / float(short)
        new_h = max(8, int(round(h * scale)))
        new_w = max(8, int(round(w * scale)))
        gray = F.interpolate(
            gray, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

    if blur_kernel > 1:
        k = int(blur_kernel)
        if k % 2 == 0:
            k += 1
        gray = F.avg_pool2d(gray, kernel_size=k, stride=1, padding=k // 2)

    return gray[0, 0], gray[1, 0]


def motion_score_from_pair(
    pair: torch.Tensor,
    *,
    resize_short_side: int,
    blur_kernel: int,
    diff_pixel_threshold: float,
) -> float:
    frame0, frame1 = _to_gray_float_pair(
        pair,
        resize_short_side=resize_short_side,
        blur_kernel=blur_kernel,
    )
    diff = torch.abs(frame1 - frame0)
    active = (diff >= float(diff_pixel_threshold)).to(torch.float32)
    return float(active.mean().item())


def motion_scores_from_pairs(
    frame0: torch.Tensor,
    frame1: torch.Tensor,
    *,
    resize_short_side: int,
    blur_kernel: int,
    diff_pixel_threshold: float,
    device: torch.device,
) -> torch.Tensor:
    """Vectorized motion score over frame pairs [B,C,H,W] or [B,H,W,C]."""
    if frame0.ndim != 4 or frame1.ndim != 4:
        raise ValueError("Expected 4D frame batches")

    if int(frame0.shape[1]) == 3:
        x0 = frame0
        x1 = frame1
    elif int(frame0.shape[-1]) == 3:
        x0 = frame0.permute(0, 3, 1, 2)
        x1 = frame1.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unsupported frame layout: {tuple(frame0.shape)}")

    x0 = x0.to(device=device, dtype=torch.float32)
    x1 = x1.to(device=device, dtype=torch.float32)
    if x0.max() > 1.0:
        x0 = x0 / 255.0
    if x1.max() > 1.0:
        x1 = x1 / 255.0

    g0 = 0.299 * x0[:, 0:1] + 0.587 * x0[:, 1:2] + 0.114 * x0[:, 2:3]
    g1 = 0.299 * x1[:, 0:1] + 0.587 * x1[:, 1:2] + 0.114 * x1[:, 2:3]

    h = int(g0.shape[-2])
    w = int(g0.shape[-1])
    short = min(h, w)
    if resize_short_side > 0 and short != resize_short_side:
        scale = float(resize_short_side) / float(short)
        new_h = max(8, int(round(h * scale)))
        new_w = max(8, int(round(w * scale)))
        g0 = F.interpolate(
            g0, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        g1 = F.interpolate(
            g1, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

    if blur_kernel > 1:
        k = int(blur_kernel)
        if k % 2 == 0:
            k += 1
        g0 = F.avg_pool2d(g0, kernel_size=k, stride=1, padding=k // 2)
        g1 = F.avg_pool2d(g1, kernel_size=k, stride=1, padding=k // 2)

    diff = torch.abs(g1 - g0)
    active = (diff >= float(diff_pixel_threshold)).to(torch.float32)
    return active.mean(dim=(1, 2, 3)).to(device="cpu")


def _absolute_to_relative_indices(ds: LeRobotDataset, indices: np.ndarray) -> list[int]:
    if ds._absolute_to_relative_idx is None:
        return [int(x) for x in indices.tolist()]
    return [int(ds._absolute_to_relative_idx[int(x)]) for x in indices.tolist()]


def _timestamps_for_indices(ds: LeRobotDataset, indices: np.ndarray) -> list[float]:
    rel_indices = _absolute_to_relative_indices(ds, indices)
    timestamps = ds.hf_dataset[rel_indices]["timestamp"]
    return [float(ts) for ts in timestamps]


def _decode_frames_for_filter(
    *,
    video_path: Path,
    query_timestamps: list[float],
    tolerance_s: float,
    decode_backend: str | None,
    decoder_cache: VideoDecoderCache | None,
) -> torch.Tensor:
    global _TORCHCODEC_FAILED
    if decode_backend == "torchcodec" and not _TORCHCODEC_FAILED:
        try:
            return decode_video_frames_torchcodec(
                video_path,
                query_timestamps,
                tolerance_s,
                decoder_cache=decoder_cache,
            )
        except Exception as exc:
            _TORCHCODEC_FAILED = True
            logger.warning(
                "TorchCodec decode unavailable; falling back to pyav (%s)", exc
            )

    fallback_backend = "pyav" if decode_backend == "torchcodec" else decode_backend
    return decode_video_frames(
        video_path, query_timestamps, tolerance_s, fallback_backend
    )


def _prefetch_episode_decode_warmup(
    *,
    ds: LeRobotDataset,
    episode_index: int,
    anchors_start: int,
    anchors_end: int,
    frame_gap: int,
    camera_dataset_keys: tuple[str, ...],
    decode_backend: str | None,
    decoder_cache: VideoDecoderCache | None,
    prefetch_chunk_size: int,
) -> None:
    n = max(0, int(anchors_end) - int(anchors_start))
    if n <= 0:
        return

    b = min(int(max(1, prefetch_chunk_size)), n)
    anchors = np.arange(int(anchors_start), int(anchors_start) + b, dtype=np.int64)
    anchors_next = anchors + int(frame_gap)
    ts0 = _timestamps_for_indices(ds, anchors)
    ts1 = _timestamps_for_indices(ds, anchors_next)
    episode_meta = ds.meta.episodes[int(episode_index)]

    for camera_dataset_key in camera_dataset_keys:
        video_path = ds.root / ds.meta.get_video_file_path(
            int(episode_index), camera_dataset_key
        )
        from_timestamp = float(
            episode_meta[f"videos/{camera_dataset_key}/from_timestamp"]
        )
        query_ts = [from_timestamp + float(ts) for ts in ts0]
        query_ts.extend(from_timestamp + float(ts) for ts in ts1)
        _ = _decode_frames_for_filter(
            video_path=video_path,
            query_timestamps=query_ts,
            tolerance_s=ds.tolerance_s,
            decode_backend=decode_backend,
            decoder_cache=decoder_cache,
        )


def _motion_scores_for_episode_chunked(
    *,
    ds: LeRobotDataset,
    episode_index: int,
    anchors: np.ndarray,
    frame_gap: int,
    camera_dataset_keys: tuple[str, ...],
    motion_cfg: Mapping[str, Any],
    aggregate_reduce: str,
    motion_device: torch.device,
    decode_backend: str | None,
    decoder_cache: VideoDecoderCache | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n = int(anchors.shape[0])
    per_cam = np.zeros((len(camera_dataset_keys), n), dtype=np.float32)
    decode_s = 0.0
    motion_s = 0.0
    chunk_size = int(max(1, motion_cfg.get("decode_chunk_size", 16)))
    episode_meta = ds.meta.episodes[int(episode_index)]

    for offset in range(0, n, chunk_size):
        chunk = anchors[offset : offset + chunk_size]
        chunk_next = chunk + int(frame_gap)
        ts0 = _timestamps_for_indices(ds, chunk)
        ts1 = _timestamps_for_indices(ds, chunk_next)

        for cam_idx, camera_dataset_key in enumerate(camera_dataset_keys):
            video_path = ds.root / ds.meta.get_video_file_path(
                int(episode_index), camera_dataset_key
            )
            from_timestamp = float(
                episode_meta[f"videos/{camera_dataset_key}/from_timestamp"]
            )
            query_ts = [from_timestamp + float(ts) for ts in ts0]
            query_ts.extend(from_timestamp + float(ts) for ts in ts1)

            t0 = time.perf_counter()
            frames = _decode_frames_for_filter(
                video_path=video_path,
                query_timestamps=query_ts,
                tolerance_s=ds.tolerance_s,
                decode_backend=decode_backend,
                decoder_cache=decoder_cache,
            )
            decode_s += time.perf_counter() - t0

            b = int(chunk.shape[0])
            f0 = frames[:b]
            f1 = frames[b : 2 * b]

            t1 = time.perf_counter()
            scores = motion_scores_from_pairs(
                f0,
                f1,
                resize_short_side=int(motion_cfg.get("resize_short_side", 224)),
                blur_kernel=int(motion_cfg.get("blur_kernel", 5)),
                diff_pixel_threshold=float(
                    motion_cfg.get("diff_pixel_threshold", 0.03)
                ),
                device=motion_device,
            )
            motion_s += time.perf_counter() - t1
            per_cam[cam_idx, offset : offset + b] = scores.numpy().astype(np.float32)

    if aggregate_reduce == "max":
        aggregate = np.max(per_cam, axis=0).astype(np.float32)
    else:
        aggregate = np.mean(per_cam, axis=0).astype(np.float32)
    return aggregate, per_cam, decode_s, motion_s


def _action_scores_for_episode_batched(
    *,
    ds: LeRobotDataset,
    anchors: np.ndarray,
    action_key: str,
    action_chunk_size: int,
    exclude_dims: list[int],
    delta_dims: list[int],
    reduce: str,
) -> tuple[np.ndarray, np.ndarray]:
    action_series: list[torch.Tensor] = []
    for delta in range(int(action_chunk_size)):
        rel_indices = _absolute_to_relative_indices(ds, anchors + int(delta))
        values = ds.hf_dataset[rel_indices][action_key]
        action_series.append(
            torch.stack(
                [v if torch.is_tensor(v) else torch.as_tensor(v) for v in values],
                dim=0,
            ).to(torch.float32)
        )

    action = torch.stack(action_series, dim=1)  # [N, T, A]
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [N,T,A], got {tuple(action.shape)}")

    # For selected dimensions, score temporal change within the chunk rather
    # than absolute level (useful for binary open/close channels).
    if len(delta_dims) > 0:
        valid_delta_dims = [
            int(idx) for idx in delta_dims if 0 <= int(idx) < int(action.shape[2])
        ]
        if len(valid_delta_dims) > 0 and int(action.shape[1]) > 1:
            delta = action[:, 1:, valid_delta_dims] - action[:, :-1, valid_delta_dims]
            action[:, 0, valid_delta_dims] = 0.0
            action[:, 1:, valid_delta_dims] = delta

    keep_dims = [
        idx for idx in range(int(action.shape[2])) if idx not in set(exclude_dims)
    ]
    vec = action if len(keep_dims) == 0 else action[:, :, keep_dims]
    norms = torch.linalg.norm(vec, dim=2)  # [N, T]
    ratio = (norms > 1.0e-8).to(torch.float32).mean(dim=1)
    if str(reduce) == "mean":
        score = norms.mean(dim=1)
    else:
        score = norms.max(dim=1).values
    return score.numpy().astype(np.float32), ratio.numpy().astype(np.float32)


@dataclass(frozen=True)
class _EpisodeDecision:
    keep_mask: np.ndarray
    trim_start: int
    trim_end: int
    trimmed_count: int
    motion_only_removed: int
    action_only_removed: int


def _decide_episode_keep(
    *,
    n: int,
    motion_smooth: np.ndarray,
    action_score: np.ndarray,
    action_ratio: np.ndarray,
    flow_score: np.ndarray | None,
    flow_valid: np.ndarray | None,
    mode: str,
    method: str,
    motion_enabled: bool,
    action_enabled: bool,
    trim_enabled: bool,
    motion_cfg: Mapping[str, Any],
    action_cfg: Mapping[str, Any],
    low_thr: float,
    high_thr: float,
    use_hysteresis: bool,
) -> _EpisodeDecision:
    """Shared decision logic: motion/action keep, mode combine, trim, stats."""
    motion_keep = np.ones((n,), dtype=np.bool_)
    if motion_enabled:
        if method == "sparse_flow":
            if flow_score is not None and flow_valid is not None:
                flow_thr = float(
                    motion_cfg["sparse_flow"].get("median_flow_threshold", 0.6)
                )
                min_valid = float(
                    motion_cfg["sparse_flow"].get("min_tracked_fraction", 0.25)
                )
                motion_keep = (flow_score >= flow_thr) & (flow_valid >= min_valid)
            else:
                motion_keep[:] = False
        else:
            if use_hysteresis and high_thr > low_thr:
                keep_hi = motion_smooth >= high_thr
                drop_lo = motion_smooth < low_thr
                uncertain = (~keep_hi) & (~drop_lo)
                motion_keep = keep_hi | uncertain
            else:
                thr = high_thr if high_thr > 0 else low_thr
                motion_keep = motion_smooth >= thr

    action_keep = np.ones((n,), dtype=np.bool_)
    if action_enabled and np.isfinite(action_score).any():
        action_keep = (
            np.nan_to_num(action_score, nan=0.0)
            >= float(action_cfg.get("threshold", 0.02))
        ) & (
            np.nan_to_num(action_ratio, nan=0.0)
            >= float(action_cfg.get("min_nonzero_ratio", 0.0))
        )

    if mode == "motion":
        keep = motion_keep
    elif mode == "action":
        keep = action_keep
    else:
        keep = motion_keep & action_keep

    trim_start = 0
    trim_end = n
    trimmed_count = 0
    if trim_enabled:
        active_thr = high_thr if high_thr > 0 else low_thr
        active_mask = motion_smooth >= active_thr
        k = int(motion_cfg.get("consecutive_active_k", 3))
        first_active = _first_consecutive_true(active_mask, k)
        last_active = _last_consecutive_true(active_mask, k)
        if (
            first_active is not None
            and last_active is not None
            and last_active >= first_active
        ):
            trim_start = int(first_active)
            trim_end = int(last_active) + 1
            outside = np.ones((n,), dtype=np.bool_)
            outside[trim_start:trim_end] = False
            trimmed_count = int(outside.sum())
            keep[outside] = False

    keep_mask = keep.astype(np.bool_)
    m_only = 0
    a_only = 0
    if mode == "both":
        dropped = ~keep_mask
        m_only = int((dropped & (~motion_keep) & action_keep).sum())
        a_only = int((dropped & motion_keep & (~action_keep)).sum())

    return _EpisodeDecision(
        keep_mask=keep_mask,
        trim_start=trim_start,
        trim_end=trim_end,
        trimmed_count=trimmed_count,
        motion_only_removed=m_only,
        action_only_removed=a_only,
    )


def _recompute_decisions_from_cached_scores(
    *,
    payload: Any,
    episode_ids: np.ndarray,
    candidate_start: np.ndarray,
    candidate_end: np.ndarray,
    camera_dataset_keys: tuple[str, ...],
    filtering_cfg: Mapping[str, Any],
) -> dict[str, Any] | None:
    required_keys = {
        "candidate_offsets_start",
        "candidate_offsets_end",
        "motion_raw",
        "action_score",
        "action_nonzero_ratio",
        "keep_mask",
    }
    if not all(key in payload for key in required_keys):
        return None

    mode = str(filtering_cfg["mode"])
    motion_cfg = dict(filtering_cfg["motion"])
    action_cfg = dict(filtering_cfg["action"])
    motion_enabled = bool(motion_cfg.get("enabled", False)) and mode in {
        "motion",
        "both",
    }
    action_enabled = bool(action_cfg.get("enabled", False)) and mode in {
        "action",
        "both",
    }
    trim_enabled = bool(filtering_cfg.get("trim_episode_ends", True)) and motion_enabled

    method = str(motion_cfg.get("method", "frame_diff"))
    low_thr = float(motion_cfg.get("low_threshold", 0.01))
    high_thr = float(motion_cfg.get("high_threshold", 0.02))
    use_hysteresis = bool(motion_cfg.get("use_hysteresis", True))

    payload_episode_ids = payload["episode_ids"].astype(np.int32)
    if payload_episode_ids.shape != episode_ids.astype(np.int32).shape:
        return None
    if not np.array_equal(payload_episode_ids, episode_ids.astype(np.int32)):
        return None
    if not np.array_equal(
        payload["candidate_start"].astype(np.int64), candidate_start.astype(np.int64)
    ):
        return None
    if not np.array_equal(
        payload["candidate_end"].astype(np.int64), candidate_end.astype(np.int64)
    ):
        return None

    candidate_offsets_start = payload["candidate_offsets_start"].astype(np.int64)
    candidate_offsets_end = payload["candidate_offsets_end"].astype(np.int64)
    motion_raw_all = payload["motion_raw"].astype(np.float32)
    action_score_all = payload["action_score"].astype(np.float32)
    action_ratio_all = payload["action_nonzero_ratio"].astype(np.float32)
    flow_score_all = (
        payload["flow_score"].astype(np.float32) if "flow_score" in payload else None
    )
    flow_valid_all = (
        payload["flow_valid_fraction"].astype(np.float32)
        if "flow_valid_fraction" in payload
        else None
    )
    motion_raw_per_camera_all = (
        payload["motion_raw_per_camera"].astype(np.float32)
        if "motion_raw_per_camera" in payload
        else None
    )

    kept_anchor_values_parts: list[np.ndarray] = []
    kept_offsets_start: list[int] = []
    kept_offsets_end: list[int] = []
    kept_counts: list[int] = []
    kept_range_start: list[int] = []
    kept_range_end: list[int] = []
    keep_mask_parts: list[np.ndarray] = []
    trim_start_local: list[int] = []
    trim_end_local: list[int] = []
    motion_smooth_parts: list[np.ndarray] = []
    motion_smooth_per_camera_parts: list[np.ndarray] = []

    total_before = 0
    total_after = 0
    total_trimmed = 0
    motion_only_removed = 0
    action_only_removed = 0
    running_kept_offset = 0

    for row_idx in range(int(episode_ids.shape[0])):
        start = int(candidate_start[row_idx])
        end = int(candidate_end[row_idx])
        n = max(0, end - start)
        sl = slice(
            int(candidate_offsets_start[row_idx]), int(candidate_offsets_end[row_idx])
        )
        if int(candidate_offsets_end[row_idx] - candidate_offsets_start[row_idx]) != n:
            return None

        anchors = np.arange(start, end, dtype=np.int64)
        total_before += int(n)

        motion_raw = motion_raw_all[sl]
        motion_smooth = _smooth_1d(
            np.nan_to_num(motion_raw, nan=0.0),
            int(motion_cfg.get("smoothing_window", 5)),
        )
        motion_smooth_parts.append(motion_smooth.astype(np.float32))

        if motion_raw_per_camera_all is not None:
            per_cam_raw = motion_raw_per_camera_all[:, sl]
            if int(per_cam_raw.shape[0]) != len(camera_dataset_keys):
                return None
            per_cam_smooth = np.stack(
                [
                    _smooth_1d(
                        np.nan_to_num(per_cam_raw[cam_idx], nan=0.0),
                        int(motion_cfg.get("smoothing_window", 5)),
                    )
                    for cam_idx in range(len(camera_dataset_keys))
                ],
                axis=0,
            ).astype(np.float32)
        else:
            per_cam_smooth = np.zeros((len(camera_dataset_keys), n), dtype=np.float32)
        motion_smooth_per_camera_parts.append(per_cam_smooth)

        action_score = action_score_all[sl]
        action_ratio = action_ratio_all[sl]

        ep_flow_score = flow_score_all[sl] if flow_score_all is not None else None
        ep_flow_valid = flow_valid_all[sl] if flow_valid_all is not None else None
        if (
            motion_enabled
            and method == "sparse_flow"
            and (ep_flow_score is None or ep_flow_valid is None)
        ):
            return None

        decision = _decide_episode_keep(
            n=n,
            motion_smooth=motion_smooth,
            action_score=action_score,
            action_ratio=action_ratio,
            flow_score=ep_flow_score,
            flow_valid=ep_flow_valid,
            mode=mode,
            method=method,
            motion_enabled=motion_enabled,
            action_enabled=action_enabled,
            trim_enabled=trim_enabled,
            motion_cfg=motion_cfg,
            action_cfg=action_cfg,
            low_thr=low_thr,
            high_thr=high_thr,
            use_hysteresis=use_hysteresis,
        )

        total_trimmed += decision.trimmed_count
        motion_only_removed += decision.motion_only_removed
        action_only_removed += decision.action_only_removed

        kept = anchors[decision.keep_mask]
        total_after += int(kept.shape[0])

        kept_offsets_start.append(running_kept_offset)
        running_kept_offset += int(kept.shape[0])
        kept_offsets_end.append(running_kept_offset)
        kept_counts.append(int(kept.shape[0]))
        kept_range_start.append(int(kept[0]) if kept.shape[0] > 0 else start)
        kept_range_end.append(int(kept[-1]) + 1 if kept.shape[0] > 0 else start)
        trim_start_local.append(decision.trim_start)
        trim_end_local.append(decision.trim_end)

        kept_anchor_values_parts.append(kept)
        keep_mask_parts.append(decision.keep_mask)

    kept_anchor_values = (
        np.concatenate(kept_anchor_values_parts, axis=0)
        if kept_anchor_values_parts
        else np.zeros((0,), dtype=np.int64)
    )
    anchors_before = int(total_before)
    trimmed_fraction = float(total_trimmed) / float(max(1, anchors_before))

    return {
        "kept_anchor_values": kept_anchor_values,
        "kept_offsets_start": np.asarray(kept_offsets_start, dtype=np.int64),
        "kept_offsets_end": np.asarray(kept_offsets_end, dtype=np.int64),
        "kept_counts": np.asarray(kept_counts, dtype=np.int32),
        "kept_range_start": np.asarray(kept_range_start, dtype=np.int64),
        "kept_range_end": np.asarray(kept_range_end, dtype=np.int64),
        "keep_mask": np.concatenate(keep_mask_parts, axis=0).astype(np.bool_),
        "trim_start_local": np.asarray(trim_start_local, dtype=np.int32),
        "trim_end_local": np.asarray(trim_end_local, dtype=np.int32),
        "motion_smooth": np.concatenate(motion_smooth_parts, axis=0).astype(np.float32),
        "motion_smooth_per_camera": np.concatenate(
            motion_smooth_per_camera_parts, axis=1
        ).astype(np.float32),
        "anchors_before": anchors_before,
        "anchors_after": int(total_after),
        "trimmed_fraction": float(trimmed_fraction),
        "motion_only_removed": int(motion_only_removed),
        "action_only_removed": int(action_only_removed),
    }


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = int(max(1, window))
    if window <= 1 or values.size <= 1:
        return values.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    padded = np.pad(values.astype(np.float32), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _first_consecutive_true(mask: np.ndarray, k: int) -> int | None:
    k = int(max(1, k))
    run = 0
    for idx, value in enumerate(mask.tolist()):
        run = run + 1 if value else 0
        if run >= k:
            return idx - k + 1
    return None


def _last_consecutive_true(mask: np.ndarray, k: int) -> int | None:
    if mask.size == 0:
        return None
    rev = mask[::-1]
    first_rev = _first_consecutive_true(rev, k)
    if first_rev is None:
        return None
    end_inclusive = int(mask.size) - first_rev - 1
    return end_inclusive


def _extract_action_tensor(
    raw: Mapping[str, Any], action_key: str
) -> torch.Tensor | None:
    if action_key not in raw:
        return None
    value = raw[action_key]
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected action tensor rank 1/2, got {tuple(tensor.shape)}")
    return tensor.to(torch.float32)


def action_score_from_tensor(
    action: torch.Tensor,
    *,
    exclude_dims: list[int],
    reduce: str,
) -> tuple[float, float]:
    if action.ndim != 2:
        raise ValueError(f"Expected action shape [T,A], got {tuple(action.shape)}")
    keep_dims = [
        idx for idx in range(int(action.shape[1])) if idx not in set(exclude_dims)
    ]
    if len(keep_dims) == 0:
        vec = action
    else:
        vec = action[:, keep_dims]
    norms = torch.linalg.norm(vec, dim=1)
    nonzero_ratio = float((norms > 1.0e-8).to(torch.float32).mean().item())
    reduce = str(reduce)
    if reduce == "mean":
        score = float(norms.mean().item())
    else:
        score = float(norms.max().item())
    return score, nonzero_ratio


def _compute_sparse_flow(
    pair: torch.Tensor,
    *,
    resize_short_side: int,
    blur_kernel: int,
    max_corners: int,
    quality_level: float,
    min_distance: float,
    win_size: int,
    max_level: int,
) -> tuple[float, float]:
    del pair
    del resize_short_side
    del blur_kernel
    del max_corners
    del quality_level
    del min_distance
    del win_size
    del max_level
    return 0.0, 0.0


def _build_filter_cache_path(
    *,
    dataset_root: Path,
    split: str,
    camera_tag: str,
    fingerprint: str,
) -> Path:
    cache_dir = dataset_root / "meta" / "hlrp_action_frame_filter_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{split}_{camera_tag}_{fingerprint}.npz"


def _camera_name_token(camera_key: str) -> str:
    token = str(camera_key).split(".")[-1].strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in token).strip("_")
    return safe or "camera"


def _camera_tag(camera_keys: tuple[str, ...]) -> str:
    if len(camera_keys) == 1:
        return _camera_name_token(camera_keys[0])
    return f"allcams{len(camera_keys)}"


def _fingerprint_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _score_fingerprint_payload_from_full_payload(
    full_payload: Mapping[str, Any],
) -> dict[str, Any]:
    filtering = dict(full_payload.get("filtering", {}))
    motion = dict(filtering.get("motion", {}))
    sparse_flow = dict(motion.get("sparse_flow", {}))
    action = dict(filtering.get("action", {}))
    return {
        "version": 1,
        "repo_id": full_payload.get("repo_id"),
        "revision": full_payload.get("revision"),
        "split": full_payload.get("split"),
        "camera_dataset_keys": full_payload.get("camera_dataset_keys"),
        "camera_aggregate_reduce": full_payload.get("camera_aggregate_reduce"),
        "action_key": full_payload.get("action_key"),
        "request_image_deltas": full_payload.get("request_image_deltas"),
        "episode_ids": full_payload.get("episode_ids"),
        "candidate_start": full_payload.get("candidate_start"),
        "candidate_end": full_payload.get("candidate_end"),
        "score_filtering": {
            "motion": {
                "enabled": motion.get("enabled"),
                "method": motion.get("method"),
                "frame_gap": motion.get("frame_gap"),
                "decode_backend": motion.get("decode_backend"),
                "aggregate_all_cameras": motion.get("aggregate_all_cameras"),
                "aggregate_reduce": motion.get("aggregate_reduce"),
                "resize_short_side": motion.get("resize_short_side"),
                "grayscale": motion.get("grayscale"),
                "blur_kernel": motion.get("blur_kernel"),
                "diff_pixel_threshold": motion.get("diff_pixel_threshold"),
                "sparse_flow": {
                    "enabled": sparse_flow.get("enabled"),
                    "only_on_uncertain": sparse_flow.get("only_on_uncertain"),
                    "max_corners": sparse_flow.get("max_corners"),
                    "quality_level": sparse_flow.get("quality_level"),
                    "min_distance": sparse_flow.get("min_distance"),
                    "win_size": sparse_flow.get("win_size"),
                    "max_level": sparse_flow.get("max_level"),
                    "min_tracked_fraction": sparse_flow.get("min_tracked_fraction"),
                    "median_flow_threshold": sparse_flow.get("median_flow_threshold"),
                },
            },
            "action": {
                "enabled": action.get("enabled"),
                "method": action.get("method"),
                "exclude_dims": action.get("exclude_dims"),
                "delta_dims": action.get("delta_dims"),
                "chunk_size": action.get("chunk_size"),
                "chunk_reduce": action.get("chunk_reduce"),
            },
        },
    }


def _score_fingerprint_from_cache_payload(payload: Any) -> str | None:
    if "score_fingerprint" in payload:
        return str(payload["score_fingerprint"])
    if "fingerprint_payload_json" not in payload:
        return None
    try:
        full_payload = json.loads(str(payload["fingerprint_payload_json"]))
    except Exception:
        return None
    score_payload = _score_fingerprint_payload_from_full_payload(full_payload)
    return _fingerprint_payload(score_payload)


def build_action_frame_filter(
    *,
    repo_id: str,
    root: str | None,
    revision: str | None,
    video_backend: str | None,
    tolerance_s: float | None,
    request_image_deltas: tuple[int, ...],
    camera_dataset_keys: tuple[str, ...],
    camera_aggregate_reduce: str,
    action_key: str | None,
    episode_ids: np.ndarray,
    candidate_start: np.ndarray,
    candidate_end: np.ndarray,
    filtering_cfg: Mapping[str, Any],
    split: str,
) -> ActionFrameFilterResult:
    """Compute and cache action-frame filtering decisions for one source split."""
    mode = str(filtering_cfg["mode"])
    motion_cfg = dict(filtering_cfg["motion"])
    action_cfg = dict(filtering_cfg["action"])
    cache_cfg = dict(filtering_cfg["cache"])

    motion_enabled = bool(motion_cfg.get("enabled", False)) and mode in {
        "motion",
        "both",
    }
    action_enabled = bool(action_cfg.get("enabled", False)) and mode in {
        "action",
        "both",
    }
    trim_enabled = bool(filtering_cfg.get("trim_episode_ends", True)) and motion_enabled

    frame_gap = infer_motion_frame_gap(
        request_image_deltas=request_image_deltas,
        configured_frame_gap=motion_cfg.get("frame_gap"),
    )

    action_chunk_size = int(action_cfg.get("chunk_size", 1))
    action_deltas = (
        list(range(max(1, action_chunk_size))) if action_enabled and action_key else []
    )

    if len(camera_dataset_keys) == 0:
        raise ValueError("camera_dataset_keys must contain at least one camera key")

    delta_timestamps: dict[str, list[float]] = {
        str(camera_key): [0.0, float(frame_gap)] for camera_key in camera_dataset_keys
    }
    if action_enabled and action_key is not None:
        delta_timestamps[action_key] = [float(delta) for delta in action_deltas]

    ds = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        revision=revision,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
        tolerance_s=(1.0e-4 if tolerance_s is None else float(tolerance_s)),
    )

    action_lookahead = max(0, int(action_chunk_size) - 1) if action_enabled else 0
    temporal_lookahead = max(int(frame_gap), action_lookahead)
    max_anchor_end_exclusive = max(0, int(len(ds.hf_dataset)) - int(temporal_lookahead))
    raw_candidate_start = candidate_start.astype(np.int64, copy=False)
    raw_candidate_end = candidate_end.astype(np.int64, copy=False)
    candidate_start = np.clip(raw_candidate_start, 0, max_anchor_end_exclusive).astype(
        np.int64
    )
    candidate_end = np.clip(raw_candidate_end, 0, max_anchor_end_exclusive).astype(
        np.int64
    )
    candidate_end = np.maximum(candidate_end, candidate_start).astype(np.int64)
    if not (
        np.array_equal(candidate_start, raw_candidate_start)
        and np.array_equal(candidate_end, raw_candidate_end)
    ):
        logger.warning(
            "Action-frame filtering clamped candidate ranges split=%s source=%s max_anchor_end=%d",
            split,
            repo_id,
            max_anchor_end_exclusive,
        )

    logger.info(
        "Action-frame filtering dataset resolved source=%s split=%s root=%s total_episodes=%d",
        repo_id,
        split,
        str(ds.meta.root),
        int(ds.meta.total_episodes),
    )

    source_root = Path(ds.meta.root)

    action_ds: LeRobotDataset | None = None
    if action_enabled and action_key is not None:
        action_ds = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            revision=revision,
            delta_timestamps={action_key: [float(delta) for delta in action_deltas]},
            video_backend=video_backend,
            tolerance_s=(1.0e-4 if tolerance_s is None else float(tolerance_s)),
        )
    fingerprint_payload = {
        "version": 1,
        "repo_id": str(repo_id),
        "revision": None if revision is None else str(revision),
        "split": str(split),
        "camera_dataset_keys": [str(key) for key in camera_dataset_keys],
        "camera_aggregate_reduce": str(camera_aggregate_reduce),
        "action_key": None if action_key is None else str(action_key),
        "request_image_deltas": [int(x) for x in request_image_deltas],
        "episode_ids": [int(x) for x in episode_ids.tolist()],
        "candidate_start": [int(x) for x in candidate_start.tolist()],
        "candidate_end": [int(x) for x in candidate_end.tolist()],
        "filtering": filtering_cfg,
    }
    score_fingerprint_payload = _score_fingerprint_payload_from_full_payload(
        fingerprint_payload
    )
    score_fingerprint = _fingerprint_payload(score_fingerprint_payload)
    fingerprint = _fingerprint_payload(fingerprint_payload)
    camera_tag = _camera_tag(camera_dataset_keys)
    cache_path = _build_filter_cache_path(
        dataset_root=source_root,
        split=split,
        camera_tag=camera_tag,
        fingerprint=fingerprint,
    )
    aggregate_reduce = str(camera_aggregate_reduce).lower()
    if aggregate_reduce not in {"mean", "max"}:
        raise ValueError(
            f"Unsupported camera_aggregate_reduce={camera_aggregate_reduce!r}"
        )
    low_thr = float(motion_cfg.get("low_threshold", 0.01))
    high_thr = float(motion_cfg.get("high_threshold", 0.02))

    can_reuse = (
        bool(cache_cfg.get("enabled", True))
        and bool(cache_cfg.get("reuse_if_config_unchanged", True))
        and not bool(cache_cfg.get("force_recompute", False))
        and cache_path.is_file()
    )

    if can_reuse:
        payload = np.load(cache_path, allow_pickle=False)
        cached_fingerprint = (
            str(payload["fingerprint"]) if "fingerprint" in payload else ""
        )
        if cached_fingerprint == fingerprint:
            summary = {
                "cache": "hit",
                "cache_path": str(cache_path),
                "fingerprint": fingerprint,
                "anchors_before": int(payload["anchors_before"]),
                "anchors_after": int(payload["anchors_after"]),
                "trimmed_fraction": float(payload["trimmed_fraction"]),
                "motion_only_removed": int(payload["motion_only_removed"]),
                "action_only_removed": int(payload["action_only_removed"]),
            }
            return ActionFrameFilterResult(
                kept_anchor_values=payload["kept_anchor_values"].astype(np.int64),
                kept_offsets_start=payload["kept_offsets_start"].astype(np.int64),
                kept_offsets_end=payload["kept_offsets_end"].astype(np.int64),
                kept_counts=payload["kept_counts"].astype(np.int32),
                kept_range_start=payload["kept_range_start"].astype(np.int64),
                kept_range_end=payload["kept_range_end"].astype(np.int64),
                cache_path=str(cache_path),
                summary=summary,
            )

    if (
        bool(cache_cfg.get("enabled", True))
        and bool(cache_cfg.get("reuse_if_config_unchanged", True))
        and not bool(cache_cfg.get("force_recompute", False))
    ):
        cache_dir = cache_path.parent
        pattern = f"{split}_{camera_tag}_*.npz"
        for score_cache_path in sorted(
            cache_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            if score_cache_path == cache_path:
                continue
            try:
                payload = np.load(score_cache_path, allow_pickle=False)
            except Exception:
                continue
            candidate_score_fingerprint = _score_fingerprint_from_cache_payload(payload)
            if candidate_score_fingerprint != score_fingerprint:
                continue

            recomputed = _recompute_decisions_from_cached_scores(
                payload=payload,
                episode_ids=episode_ids,
                candidate_start=candidate_start,
                candidate_end=candidate_end,
                camera_dataset_keys=camera_dataset_keys,
                filtering_cfg=filtering_cfg,
            )
            if recomputed is None:
                continue

            logger.info(
                "Action-frame filtering cache score-hit split=%s source=%s from=%s to=%s",
                split,
                repo_id,
                str(score_cache_path),
                str(cache_path),
            )

            if bool(cache_cfg.get("enabled", True)):
                fingerprint_payload_json = json.dumps(
                    fingerprint_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                filtering_config_json = json.dumps(
                    filtering_cfg,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                np.savez_compressed(
                    cache_path,
                    fingerprint=np.asarray(fingerprint),
                    score_fingerprint=np.asarray(score_fingerprint),
                    score_fingerprint_payload_json=np.asarray(
                        json.dumps(
                            score_fingerprint_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                    fingerprint_payload_json=np.asarray(fingerprint_payload_json),
                    filtering_config_json=np.asarray(filtering_config_json),
                    camera_dataset_key=np.asarray(str(camera_dataset_keys[0])),
                    camera_dataset_keys=np.asarray(list(camera_dataset_keys)),
                    motion_aggregate_reduce=np.asarray(aggregate_reduce),
                    kept_anchor_values=recomputed["kept_anchor_values"].astype(
                        np.int64
                    ),
                    kept_offsets_start=recomputed["kept_offsets_start"].astype(
                        np.int64
                    ),
                    kept_offsets_end=recomputed["kept_offsets_end"].astype(np.int64),
                    kept_counts=recomputed["kept_counts"].astype(np.int32),
                    kept_range_start=recomputed["kept_range_start"].astype(np.int64),
                    kept_range_end=recomputed["kept_range_end"].astype(np.int64),
                    episode_ids=episode_ids.astype(np.int32),
                    candidate_start=candidate_start.astype(np.int64),
                    candidate_end=candidate_end.astype(np.int64),
                    candidate_offsets_start=payload["candidate_offsets_start"].astype(
                        np.int64
                    ),
                    candidate_offsets_end=payload["candidate_offsets_end"].astype(
                        np.int64
                    ),
                    motion_raw=payload["motion_raw"].astype(np.float32),
                    motion_smooth=recomputed["motion_smooth"].astype(np.float32),
                    motion_raw_per_camera=(
                        payload["motion_raw_per_camera"].astype(np.float32)
                        if "motion_raw_per_camera" in payload
                        else np.zeros(
                            (
                                len(camera_dataset_keys),
                                int(payload["motion_raw"].shape[0]),
                            ),
                            dtype=np.float32,
                        )
                    ),
                    motion_smooth_per_camera=recomputed[
                        "motion_smooth_per_camera"
                    ].astype(np.float32),
                    action_score=payload["action_score"].astype(np.float32),
                    action_nonzero_ratio=payload["action_nonzero_ratio"].astype(
                        np.float32
                    ),
                    flow_score=(
                        payload["flow_score"].astype(np.float32)
                        if "flow_score" in payload
                        else np.full(
                            (int(payload["motion_raw"].shape[0]),),
                            np.nan,
                            dtype=np.float32,
                        )
                    ),
                    flow_valid_fraction=(
                        payload["flow_valid_fraction"].astype(np.float32)
                        if "flow_valid_fraction" in payload
                        else np.full(
                            (int(payload["motion_raw"].shape[0]),),
                            np.nan,
                            dtype=np.float32,
                        )
                    ),
                    keep_mask=recomputed["keep_mask"].astype(np.bool_),
                    trim_start_local=recomputed["trim_start_local"].astype(np.int32),
                    trim_end_local=recomputed["trim_end_local"].astype(np.int32),
                    anchors_before=np.asarray(
                        recomputed["anchors_before"], dtype=np.int64
                    ),
                    anchors_after=np.asarray(
                        recomputed["anchors_after"], dtype=np.int64
                    ),
                    trimmed_fraction=np.asarray(
                        recomputed["trimmed_fraction"], dtype=np.float32
                    ),
                    motion_only_removed=np.asarray(
                        recomputed["motion_only_removed"],
                        dtype=np.int64,
                    ),
                    action_only_removed=np.asarray(
                        recomputed["action_only_removed"],
                        dtype=np.int64,
                    ),
                    motion_low_threshold=np.asarray(low_thr, dtype=np.float32),
                    motion_high_threshold=np.asarray(high_thr, dtype=np.float32),
                    action_threshold=np.asarray(
                        float(action_cfg.get("threshold", 0.02)),
                        dtype=np.float32,
                    ),
                )

            summary = {
                "cache": "score-hit",
                "cache_path": str(cache_path),
                "fingerprint": fingerprint,
                "score_fingerprint": score_fingerprint,
                "anchors_before": int(recomputed["anchors_before"]),
                "anchors_after": int(recomputed["anchors_after"]),
                "trimmed_fraction": float(recomputed["trimmed_fraction"]),
                "motion_only_removed": int(recomputed["motion_only_removed"]),
                "action_only_removed": int(recomputed["action_only_removed"]),
            }
            return ActionFrameFilterResult(
                kept_anchor_values=recomputed["kept_anchor_values"].astype(np.int64),
                kept_offsets_start=recomputed["kept_offsets_start"].astype(np.int64),
                kept_offsets_end=recomputed["kept_offsets_end"].astype(np.int64),
                kept_counts=recomputed["kept_counts"].astype(np.int32),
                kept_range_start=recomputed["kept_range_start"].astype(np.int64),
                kept_range_end=recomputed["kept_range_end"].astype(np.int64),
                cache_path=str(cache_path),
                summary=summary,
            )

    total_before = 0
    total_after = 0
    total_trimmed = 0
    motion_only_removed = 0
    action_only_removed = 0
    total_episodes = int(episode_ids.shape[0])

    kept_anchor_values_parts: list[np.ndarray] = []
    kept_offsets_start: list[int] = []
    kept_offsets_end: list[int] = []
    kept_counts: list[int] = []
    kept_range_start: list[int] = []
    kept_range_end: list[int] = []

    motion_raw_parts: list[np.ndarray] = []
    motion_smooth_parts: list[np.ndarray] = []
    motion_raw_per_camera_parts: list[np.ndarray] = []
    motion_smooth_per_camera_parts: list[np.ndarray] = []
    action_score_parts: list[np.ndarray] = []
    action_ratio_parts: list[np.ndarray] = []
    flow_score_parts: list[np.ndarray] = []
    flow_valid_parts: list[np.ndarray] = []
    keep_mask_parts: list[np.ndarray] = []
    candidate_offsets_start: list[int] = []
    candidate_offsets_end: list[int] = []
    trim_start_local: list[int] = []
    trim_end_local: list[int] = []

    running_kept_offset = 0
    running_candidate_offset = 0
    total_candidate_anchors = int(np.maximum(candidate_end - candidate_start, 0).sum())
    start_time = time.monotonic()
    next_episode_milestone_pct = 10
    timed_fetch_s = 0.0
    timed_motion_s = 0.0
    timed_action_s = 0.0

    method = str(motion_cfg.get("method", "frame_diff"))
    use_hysteresis = bool(motion_cfg.get("use_hysteresis", True))
    motion_device_cfg = str(motion_cfg.get("device", "cpu")).lower()
    motion_device = torch.device(
        "cuda"
        if motion_device_cfg.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    decode_backend_raw = motion_cfg.get("decode_backend")
    decode_backend = (
        str(decode_backend_raw) if decode_backend_raw is not None else ds.video_backend
    )
    if decode_backend is not None:
        decode_backend = str(decode_backend)
    prefetch_next_episode = bool(motion_cfg.get("prefetch_next_episode", False))
    if decode_backend == "torchcodec" and prefetch_next_episode:
        logger.warning(
            "Action-frame filtering disabling prefetch_next_episode for torchcodec backend to avoid decoder instability."
        )
        prefetch_next_episode = False
    prefetch_chunk_size = int(max(1, motion_cfg.get("prefetch_chunk_size", 16)))
    decoder_cache = VideoDecoderCache() if decode_backend == "torchcodec" else None
    prefetch_executor: ThreadPoolExecutor | None = (
        ThreadPoolExecutor(max_workers=1) if prefetch_next_episode else None
    )
    prefetch_future: Future[None] | None = None

    logger.info(
        "Action-frame filtering decode setup split=%s source=%s backend=%s motion_device=%s prefetch_next_episode=%s",
        split,
        repo_id,
        str(decode_backend),
        str(motion_device),
        prefetch_next_episode,
    )

    for row_idx in range(int(episode_ids.shape[0])):
        if prefetch_future is not None and prefetch_future.done():
            prefetch_future.result()
            prefetch_future = None

        start = int(candidate_start[row_idx])
        end = int(candidate_end[row_idx])
        n = max(0, end - start)

        candidate_offsets_start.append(running_candidate_offset)
        candidate_offsets_end.append(running_candidate_offset + n)
        running_candidate_offset += n

        if n == 0:
            kept_offsets_start.append(running_kept_offset)
            kept_offsets_end.append(running_kept_offset)
            kept_counts.append(0)
            kept_range_start.append(start)
            kept_range_end.append(start)
            trim_start_local.append(0)
            trim_end_local.append(0)

            motion_raw_parts.append(np.zeros((0,), dtype=np.float32))
            motion_smooth_parts.append(np.zeros((0,), dtype=np.float32))
            motion_raw_per_camera_parts.append(
                np.zeros((len(camera_dataset_keys), 0), dtype=np.float32)
            )
            motion_smooth_per_camera_parts.append(
                np.zeros((len(camera_dataset_keys), 0), dtype=np.float32)
            )
            action_score_parts.append(np.zeros((0,), dtype=np.float32))
            action_ratio_parts.append(np.zeros((0,), dtype=np.float32))
            flow_score_parts.append(np.zeros((0,), dtype=np.float32))
            flow_valid_parts.append(np.zeros((0,), dtype=np.float32))
            keep_mask_parts.append(np.zeros((0,), dtype=np.bool_))
            continue

        anchors = np.arange(start, end, dtype=np.int64)
        total_before += int(n)

        if prefetch_executor is not None and row_idx + 1 < int(episode_ids.shape[0]):
            if prefetch_future is None:
                prefetch_future = prefetch_executor.submit(
                    _prefetch_episode_decode_warmup,
                    ds=ds,
                    episode_index=int(episode_ids[row_idx + 1]),
                    anchors_start=int(candidate_start[row_idx + 1]),
                    anchors_end=int(candidate_end[row_idx + 1]),
                    frame_gap=frame_gap,
                    camera_dataset_keys=camera_dataset_keys,
                    decode_backend=decode_backend,
                    decoder_cache=decoder_cache,
                    prefetch_chunk_size=prefetch_chunk_size,
                )

        motion_raw = np.full((n,), np.nan, dtype=np.float32)
        motion_raw_per_camera = np.full(
            (len(camera_dataset_keys), n),
            np.nan,
            dtype=np.float32,
        )
        action_score = np.full((n,), np.nan, dtype=np.float32)
        action_ratio = np.full((n,), np.nan, dtype=np.float32)

        need_motion = motion_enabled or trim_enabled
        need_action = action_enabled and action_key is not None

        if need_motion and method in {"frame_diff", "two_stage"}:
            motion_raw, motion_raw_per_camera, decode_s, vector_motion_s = (
                _motion_scores_for_episode_chunked(
                    ds=ds,
                    episode_index=int(episode_ids[row_idx]),
                    anchors=anchors,
                    frame_gap=frame_gap,
                    camera_dataset_keys=camera_dataset_keys,
                    motion_cfg=motion_cfg,
                    aggregate_reduce=aggregate_reduce,
                    motion_device=motion_device,
                    decode_backend=decode_backend,
                    decoder_cache=decoder_cache,
                )
            )
            timed_fetch_s += decode_s
            timed_motion_s += vector_motion_s

        for local_idx, anchor in enumerate(anchors.tolist()):
            if need_motion and method in {"frame_diff", "two_stage"}:
                pass
            else:
                fetch_t0 = time.perf_counter()
                raw = ds[int(anchor)]
                timed_fetch_s += time.perf_counter() - fetch_t0

                if need_motion:
                    motion_t0 = time.perf_counter()
                    per_camera_scores: list[float] = []
                    for cam_idx, camera_dataset_key in enumerate(camera_dataset_keys):
                        pair = raw[camera_dataset_key]
                        pair_t = (
                            pair if torch.is_tensor(pair) else torch.as_tensor(pair)
                        )
                        score = motion_score_from_pair(
                            pair_t,
                            resize_short_side=int(
                                motion_cfg.get("resize_short_side", 224)
                            ),
                            blur_kernel=int(motion_cfg.get("blur_kernel", 5)),
                            diff_pixel_threshold=float(
                                motion_cfg.get("diff_pixel_threshold", 0.03)
                            ),
                        )
                        motion_raw_per_camera[cam_idx, local_idx] = score
                        per_camera_scores.append(score)
                    if aggregate_reduce == "max":
                        motion_raw[local_idx] = float(np.max(per_camera_scores))
                    else:
                        motion_raw[local_idx] = float(np.mean(per_camera_scores))
                    timed_motion_s += time.perf_counter() - motion_t0

            if need_action:
                if action_ds is not None:
                    # Filled once below in batched mode.
                    pass
                else:
                    action_t0 = time.perf_counter()
                    action_score[local_idx] = 0.0
                    action_ratio[local_idx] = 0.0
                    timed_action_s += time.perf_counter() - action_t0

        if need_action and action_ds is not None:
            action_t0 = time.perf_counter()
            batched_score, batched_ratio = _action_scores_for_episode_batched(
                ds=action_ds,
                anchors=anchors,
                action_key=action_key,
                action_chunk_size=action_chunk_size,
                exclude_dims=list(action_cfg.get("exclude_dims", [])),
                delta_dims=list(action_cfg.get("delta_dims", [])),
                reduce=str(action_cfg.get("chunk_reduce", "max")),
            )
            action_score[:] = batched_score
            action_ratio[:] = batched_ratio
            timed_action_s += time.perf_counter() - action_t0

        processed_anchors = int(running_candidate_offset)
        if processed_anchors > 0 and processed_anchors % 2000 == 0:
            elapsed = max(1.0e-6, time.monotonic() - start_time)
            anchors_per_s = float(processed_anchors) / elapsed
            remaining = max(0, int(total_candidate_anchors - processed_anchors))
            eta_s = float(remaining) / max(1.0e-6, anchors_per_s)
            logger.info(
                "Action-frame filtering progress split=%s source=%s anchors=%d/%d (%.1f%%) rate=%.1f/s eta=%.0fs",
                split,
                repo_id,
                processed_anchors,
                total_candidate_anchors,
                100.0 * float(processed_anchors) / max(1, total_candidate_anchors),
                anchors_per_s,
                eta_s,
            )

        motion_smooth = _smooth_1d(
            np.nan_to_num(motion_raw, nan=0.0),
            int(motion_cfg.get("smoothing_window", 5)),
        )
        motion_smooth_per_camera = np.stack(
            [
                _smooth_1d(
                    np.nan_to_num(motion_raw_per_camera[cam_idx], nan=0.0),
                    int(motion_cfg.get("smoothing_window", 5)),
                )
                for cam_idx in range(len(camera_dataset_keys))
            ],
            axis=0,
        ).astype(np.float32)

        flow_score = np.full((n,), np.nan, dtype=np.float32)
        flow_valid = np.full((n,), np.nan, dtype=np.float32)

        if motion_enabled and method == "sparse_flow":
            for local_idx, anchor in enumerate(anchors.tolist()):
                raw = ds[int(anchor)]
                pair = raw[camera_dataset_keys[0]]
                pair_t = pair if torch.is_tensor(pair) else torch.as_tensor(pair)
                try:
                    score, valid_frac = _compute_sparse_flow(
                        pair_t,
                        resize_short_side=int(motion_cfg.get("resize_short_side", 224)),
                        blur_kernel=int(motion_cfg.get("blur_kernel", 5)),
                        max_corners=int(
                            motion_cfg["sparse_flow"].get("max_corners", 200)
                        ),
                        quality_level=float(
                            motion_cfg["sparse_flow"].get("quality_level", 0.01)
                        ),
                        min_distance=float(
                            motion_cfg["sparse_flow"].get("min_distance", 5.0)
                        ),
                        win_size=int(motion_cfg["sparse_flow"].get("win_size", 15)),
                        max_level=int(motion_cfg["sparse_flow"].get("max_level", 2)),
                    )
                except Exception:
                    score = 0.0
                    valid_frac = 0.0
                flow_score[local_idx] = score
                flow_valid[local_idx] = valid_frac

        if (
            motion_enabled
            and method == "two_stage"
            and bool(motion_cfg["sparse_flow"].get("enabled", False))
        ):
            logger.warning(
                "Sparse-flow refinement requested but torch-only mode is active; skipping refinement."
            )

        decision = _decide_episode_keep(
            n=n,
            motion_smooth=motion_smooth,
            action_score=action_score,
            action_ratio=action_ratio,
            flow_score=(
                flow_score if motion_enabled and method == "sparse_flow" else None
            ),
            flow_valid=(
                flow_valid if motion_enabled and method == "sparse_flow" else None
            ),
            mode=mode,
            method=method,
            motion_enabled=motion_enabled,
            action_enabled=action_enabled and action_key is not None,
            trim_enabled=trim_enabled,
            motion_cfg=motion_cfg,
            action_cfg=action_cfg,
            low_thr=low_thr,
            high_thr=high_thr,
            use_hysteresis=use_hysteresis,
        )

        total_trimmed += decision.trimmed_count
        motion_only_removed += decision.motion_only_removed
        action_only_removed += decision.action_only_removed

        kept = anchors[decision.keep_mask]
        total_after += int(kept.shape[0])

        kept_offsets_start.append(running_kept_offset)
        running_kept_offset += int(kept.shape[0])
        kept_offsets_end.append(running_kept_offset)
        kept_counts.append(int(kept.shape[0]))
        kept_range_start.append(int(kept[0]) if kept.shape[0] > 0 else start)
        kept_range_end.append(int(kept[-1]) + 1 if kept.shape[0] > 0 else start)
        trim_start_local.append(decision.trim_start)
        trim_end_local.append(decision.trim_end)

        kept_anchor_values_parts.append(kept)
        motion_raw_parts.append(motion_raw)
        motion_smooth_parts.append(motion_smooth)
        motion_raw_per_camera_parts.append(motion_raw_per_camera)
        motion_smooth_per_camera_parts.append(motion_smooth_per_camera)
        action_score_parts.append(action_score)
        action_ratio_parts.append(action_ratio)
        flow_score_parts.append(flow_score)
        flow_valid_parts.append(flow_valid)
        keep_mask_parts.append(decision.keep_mask)

        episodes_done = row_idx + 1
        episodes_pct = int((100 * episodes_done) / max(1, total_episodes))
        if episodes_done == 1 or episodes_pct >= next_episode_milestone_pct:
            elapsed = max(1.0e-6, time.monotonic() - start_time)
            anchors_done = int(total_before)
            anchors_per_s = float(anchors_done) / elapsed if anchors_done > 0 else 0.0
            anchors_remaining = max(0, int(total_candidate_anchors - anchors_done))
            eta_s = (
                float(anchors_remaining) / max(1.0e-6, anchors_per_s)
                if anchors_per_s > 0.0
                else float("inf")
            )
            timed_total_s = timed_fetch_s + timed_motion_s + timed_action_s
            fetch_ms_per_anchor = 1000.0 * timed_fetch_s / max(1, anchors_done)
            motion_ms_per_anchor = 1000.0 * timed_motion_s / max(1, anchors_done)
            action_ms_per_anchor = 1000.0 * timed_action_s / max(1, anchors_done)
            fetch_pct = 100.0 * timed_fetch_s / max(1.0e-6, timed_total_s)
            motion_pct = 100.0 * timed_motion_s / max(1.0e-6, timed_total_s)
            action_pct = 100.0 * timed_action_s / max(1.0e-6, timed_total_s)
            logger.info(
                "Action-frame filtering episode-progress split=%s source=%s episodes=%d/%d (%d%%) anchors=%d/%d rate=%.1f/s eta=%s timing_ms_per_anchor(fetch=%.1f,motion=%.1f,action=%.3f) timing_pct(fetch=%.1f,motion=%.1f,action=%.1f)",
                split,
                repo_id,
                episodes_done,
                total_episodes,
                episodes_pct,
                anchors_done,
                total_candidate_anchors,
                anchors_per_s,
                "inf" if not np.isfinite(eta_s) else f"{eta_s:.0f}s",
                fetch_ms_per_anchor,
                motion_ms_per_anchor,
                action_ms_per_anchor,
                fetch_pct,
                motion_pct,
                action_pct,
            )
            while next_episode_milestone_pct <= episodes_pct:
                next_episode_milestone_pct += 10

    if prefetch_future is not None:
        prefetch_future.result()
    if prefetch_executor is not None:
        prefetch_executor.shutdown(wait=True)

    kept_anchor_values = (
        np.concatenate(kept_anchor_values_parts, axis=0)
        if kept_anchor_values_parts
        else np.zeros((0,), dtype=np.int64)
    )

    anchors_before = int(total_before)
    anchors_after = int(total_after)
    trimmed_fraction = float(total_trimmed) / float(max(1, anchors_before))

    if bool(cache_cfg.get("enabled", True)):
        fingerprint_payload_json = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        filtering_config_json = json.dumps(
            filtering_cfg,
            sort_keys=True,
            separators=(",", ":"),
        )
        np.savez_compressed(
            cache_path,
            fingerprint=np.asarray(fingerprint),
            score_fingerprint=np.asarray(score_fingerprint),
            score_fingerprint_payload_json=np.asarray(
                json.dumps(
                    score_fingerprint_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            fingerprint_payload_json=np.asarray(fingerprint_payload_json),
            filtering_config_json=np.asarray(filtering_config_json),
            camera_dataset_key=np.asarray(str(camera_dataset_keys[0])),
            camera_dataset_keys=np.asarray(list(camera_dataset_keys)),
            motion_aggregate_reduce=np.asarray(aggregate_reduce),
            kept_anchor_values=kept_anchor_values.astype(np.int64),
            kept_offsets_start=np.asarray(kept_offsets_start, dtype=np.int64),
            kept_offsets_end=np.asarray(kept_offsets_end, dtype=np.int64),
            kept_counts=np.asarray(kept_counts, dtype=np.int32),
            kept_range_start=np.asarray(kept_range_start, dtype=np.int64),
            kept_range_end=np.asarray(kept_range_end, dtype=np.int64),
            episode_ids=episode_ids.astype(np.int32),
            candidate_start=candidate_start.astype(np.int64),
            candidate_end=candidate_end.astype(np.int64),
            candidate_offsets_start=np.asarray(candidate_offsets_start, dtype=np.int64),
            candidate_offsets_end=np.asarray(candidate_offsets_end, dtype=np.int64),
            motion_raw=np.concatenate(motion_raw_parts, axis=0).astype(np.float32),
            motion_smooth=np.concatenate(motion_smooth_parts, axis=0).astype(
                np.float32
            ),
            motion_raw_per_camera=np.concatenate(
                motion_raw_per_camera_parts, axis=1
            ).astype(np.float32),
            motion_smooth_per_camera=np.concatenate(
                motion_smooth_per_camera_parts, axis=1
            ).astype(np.float32),
            action_score=np.concatenate(action_score_parts, axis=0).astype(np.float32),
            action_nonzero_ratio=np.concatenate(action_ratio_parts, axis=0).astype(
                np.float32
            ),
            flow_score=np.concatenate(flow_score_parts, axis=0).astype(np.float32),
            flow_valid_fraction=np.concatenate(flow_valid_parts, axis=0).astype(
                np.float32
            ),
            keep_mask=np.concatenate(keep_mask_parts, axis=0).astype(np.bool_),
            trim_start_local=np.asarray(trim_start_local, dtype=np.int32),
            trim_end_local=np.asarray(trim_end_local, dtype=np.int32),
            anchors_before=np.asarray(anchors_before, dtype=np.int64),
            anchors_after=np.asarray(anchors_after, dtype=np.int64),
            trimmed_fraction=np.asarray(trimmed_fraction, dtype=np.float32),
            motion_only_removed=np.asarray(motion_only_removed, dtype=np.int64),
            action_only_removed=np.asarray(action_only_removed, dtype=np.int64),
            motion_low_threshold=np.asarray(low_thr, dtype=np.float32),
            motion_high_threshold=np.asarray(high_thr, dtype=np.float32),
            action_threshold=np.asarray(
                float(action_cfg.get("threshold", 0.02)), dtype=np.float32
            ),
        )

    summary = {
        "cache": "miss",
        "cache_path": str(cache_path),
        "fingerprint": fingerprint,
        "anchors_before": anchors_before,
        "anchors_after": anchors_after,
        "trimmed_fraction": trimmed_fraction,
        "motion_only_removed": int(motion_only_removed),
        "action_only_removed": int(action_only_removed),
    }

    return ActionFrameFilterResult(
        kept_anchor_values=kept_anchor_values,
        kept_offsets_start=np.asarray(kept_offsets_start, dtype=np.int64),
        kept_offsets_end=np.asarray(kept_offsets_end, dtype=np.int64),
        kept_counts=np.asarray(kept_counts, dtype=np.int32),
        kept_range_start=np.asarray(kept_range_start, dtype=np.int64),
        kept_range_end=np.asarray(kept_range_end, dtype=np.int64),
        cache_path=str(cache_path),
        summary=summary,
    )
