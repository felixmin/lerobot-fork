from types import SimpleNamespace

import numpy as np
import torch

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


def test_make_offline_dataloader_prefers_dataset_owned_sampler():
    dataset = _DatasetWithSampler()
    cfg = SimpleNamespace(
        batch_size=2,
        num_workers=0,
        seed=17,
        dataset=SimpleNamespace(streaming=False),
        policy=SimpleNamespace(drop_n_last_frames=3),
    )

    dataloader = make_offline_dataloader(cfg, dataset, device=torch.device("cpu"))
    batch = next(iter(dataloader))

    assert dataset.called_with == (17, 3)
    assert batch["index"].tolist() == [3, 1]


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
