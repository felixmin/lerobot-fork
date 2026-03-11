from types import SimpleNamespace

import numpy as np
import torch
from accelerate.data_loader import prepare_data_loader

from lerobot.datasets.sampler import WeightedSourceSampler
from lerobot.scripts.lerobot_train import make_offline_dataloader


class _FixedSampler(torch.utils.data.Sampler[int]):
    def __iter__(self):
        yield from [3, 1, 2, 0]

    def __len__(self) -> int:
        return 4


class _DatasetWithSampler(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.called_with = None
        self.episodes = None
        self._sampler = _FixedSampler()

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"index": torch.tensor(index, dtype=torch.int64)}

    def build_sampler(self, *, seed: int, drop_n_last_frames: int):
        self.called_with = (seed, drop_n_last_frames)
        return self._sampler


class _FixedTupleSampler(torch.utils.data.Sampler[tuple[int, int]]):
    def __iter__(self):
        yield from [(0, 10), (1, 20), (0, 30), (1, 40)]

    def __len__(self) -> int:
        return 4


class _TupleDatasetWithSampler(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self._sampler = _FixedTupleSampler()

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: tuple[int, int]) -> dict[str, torch.Tensor]:
        source_id, anchor = index
        return {
            "source_id": torch.tensor(source_id, dtype=torch.int64),
            "anchor": torch.tensor(anchor, dtype=torch.int64),
        }

    def build_sampler(self, *, seed: int, drop_n_last_frames: int):
        return self._sampler


def _make_test_cfg(batch_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=batch_size,
        num_workers=0,
        seed=17,
        dataset=SimpleNamespace(streaming=False),
        policy=SimpleNamespace(drop_n_last_frames=3),
    )


def _shard_batches(dataloader: torch.utils.data.DataLoader, *, num_processes: int, process_index: int):
    shard = prepare_data_loader(
        dataloader,
        num_processes=num_processes,
        process_index=process_index,
        split_batches=False,
        put_on_device=False,
        even_batches=True,
    )
    return list(shard)


def test_make_offline_dataloader_prefers_dataset_owned_sampler():
    dataset = _DatasetWithSampler()
    cfg = _make_test_cfg()

    dataloader = make_offline_dataloader(cfg, dataset, device=torch.device("cpu"))
    batch = next(iter(dataloader))

    assert dataset.called_with == (17, 3)
    assert batch["index"].tolist() == [3, 1]


def test_accelerate_shards_dataset_owned_sampler_without_cross_rank_overlap():
    cfg = _make_test_cfg()

    rank0_loader = make_offline_dataloader(
        cfg, _DatasetWithSampler(), device=torch.device("cpu")
    )
    rank1_loader = make_offline_dataloader(
        cfg, _DatasetWithSampler(), device=torch.device("cpu")
    )

    rank0_batches = _shard_batches(rank0_loader, num_processes=2, process_index=0)
    rank1_batches = _shard_batches(rank1_loader, num_processes=2, process_index=1)

    rank0_indices = {
        int(index) for batch in rank0_batches for index in batch["index"].tolist()
    }
    rank1_indices = {
        int(index) for batch in rank1_batches for index in batch["index"].tolist()
    }

    assert len(rank0_batches) == len(rank1_batches) == 1
    assert rank0_indices.isdisjoint(rank1_indices)
    assert rank0_indices | rank1_indices == {0, 1, 2, 3}


def test_accelerate_shards_tuple_sampler_for_mixed_dataset_style_indices():
    cfg = _make_test_cfg()

    rank0_loader = make_offline_dataloader(
        cfg, _TupleDatasetWithSampler(), device=torch.device("cpu")
    )
    rank1_loader = make_offline_dataloader(
        cfg, _TupleDatasetWithSampler(), device=torch.device("cpu")
    )

    rank0_batches = _shard_batches(rank0_loader, num_processes=2, process_index=0)
    rank1_batches = _shard_batches(rank1_loader, num_processes=2, process_index=1)

    rank0_pairs = {
        (int(source_id), int(anchor))
        for batch in rank0_batches
        for source_id, anchor in zip(
            batch["source_id"].tolist(), batch["anchor"].tolist(), strict=True
        )
    }
    rank1_pairs = {
        (int(source_id), int(anchor))
        for batch in rank1_batches
        for source_id, anchor in zip(
            batch["source_id"].tolist(), batch["anchor"].tolist(), strict=True
        )
    }

    assert len(rank0_batches) == len(rank1_batches) == 1
    assert rank0_pairs.isdisjoint(rank1_pairs)
    assert rank0_pairs | rank1_pairs == {
        (0, 10),
        (1, 20),
        (0, 30),
        (1, 40),
    }


class _WeightedSource:
    def __init__(self, starts: list[int], lengths: list[int]) -> None:
        self.index = SimpleNamespace(dataset_from_index=np.asarray(starts, dtype=np.int64))
        self._lengths = np.asarray(lengths, dtype=np.int64)

    def get_effective_lengths(self, drop_n_last_frames: int = 0) -> np.ndarray:
        return np.maximum(self._lengths - int(drop_n_last_frames), 0)


def test_weighted_source_sampler_set_epoch_changes_sequence_deterministically():
    sources = [
        _WeightedSource([0, 10], [4, 4]),
        _WeightedSource([100], [8]),
    ]
    sampler = WeightedSourceSampler(
        sources=sources,
        source_weights=np.asarray([1.0, 1.0], dtype=np.float64),
        num_samples=8,
        seed=23,
    )

    sampler.set_epoch(0)
    epoch0 = list(sampler)
    sampler.set_epoch(0)
    epoch0_repeat = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)

    assert epoch0 == epoch0_repeat
    assert epoch1 != epoch0
