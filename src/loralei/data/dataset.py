"""Dataset and sampler over the pre-computed caches.

Nothing here touches the VAE or the text encoder - by this point training only
reads latent distributions and prompt embeddings off disk, which is what keeps
the transformer's 12+ GiB the only real consumer of VRAM.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import Dataset, Sampler

from ..cache.latents import load_latent
from ..cache.store import CacheLayout
from ..cache.text import load_text_embedding
from ..job import Item, Plan


@dataclass
class Sample:
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    embedding: dict[str, torch.Tensor]
    """Everything the caption was encoded to; which tensors depends on the model family."""
    caption: str
    bucket: tuple[int, int]
    name: str


class CachedDataset(Dataset):
    def __init__(
        self,
        items: list[Item],
        plan: Plan,
        layout: CacheLayout,
        repeats: int = 1,
        caption_dropout: float = 0.0,
    ) -> None:
        self.items = list(items) * max(1, repeats)
        self.plan = plan
        self.layout = layout
        self.caption_dropout = caption_dropout

    def __len__(self) -> int:
        return len(self.items)

    def bucket_of(self, index: int) -> tuple[int, int]:
        return self.plan.bucket_for(self.items[index])

    def __getitem__(self, index: int) -> Sample:
        item = self.items[index]
        bucket = self.plan.bucket_for(item)
        latent = load_latent(self.layout, item.key, bucket)
        caption = item.caption
        if self.caption_dropout and random.random() < self.caption_dropout:
            caption = ""
        return Sample(
            latent_mean=latent["mean"],
            latent_std=latent["std"],
            embedding=load_text_embedding(self.layout, caption),
            caption=caption,
            bucket=bucket,
            name=item.image.name,
        )


class BucketBatchSampler(Sampler[list[int]]):
    """Yield batches whose members share a bucket - mixed shapes cannot be stacked."""

    def __init__(
        self,
        dataset: CachedDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.seed = seed

        self.by_bucket: dict[tuple[int, int], list[int]] = {}
        for index in range(len(dataset)):
            self.by_bucket.setdefault(dataset.bucket_of(index), []).append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self.by_bucket.values():
            order = list(indices)
            if self.shuffle:
                rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for indices in self.by_bucket.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def collate(samples: list[Sample]) -> dict:
    """Latents stack (same bucket); embeddings stay a list (variable token count, and
    more than one tensor per caption for some models)."""
    return {
        "latent_mean": torch.stack([s.latent_mean for s in samples]),
        "latent_std": torch.stack([s.latent_std for s in samples]),
        "embeddings": [s.embedding for s in samples],
        "captions": [s.caption for s in samples],
        "names": [s.name for s in samples],
        "bucket": samples[0].bucket,
    }
