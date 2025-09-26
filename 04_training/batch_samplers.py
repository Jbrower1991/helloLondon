"""Batch sampling utilities for training runs.

Provides a registry-driven interface so training scripts can
switch between the stock uniform sampler and experimental
alternatives (e.g. a variance-aware sampler) through
configuration alone. The variance-aware sampler implemented
here is a lightweight approximation that oversamples high
variance windows – it is intended to be a drop-in scaffold
that you can evolve into a full PBit implementation without
having to touch the trainers again.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SamplerSpec:
    """Simple spec used by :func:`build_batch_sampler`."""

    name: str
    kwargs: Dict[str, object]


class BaseBatchSampler:
    """Abstract base class for batch samplers."""

    def sample(
        self,
        data: np.memmap,
        block_size: int,
        batch_size: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return the starting offsets for ``batch_size`` sequences."""

        raise NotImplementedError


class UniformBatchSampler(BaseBatchSampler):
    """Replicates the original ``torch.randint`` sampling logic."""

    def sample(
        self,
        data: np.memmap,
        block_size: int,
        batch_size: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        upper = max(1, len(data) - block_size)
        if generator is None:
            return torch.randint(upper, (batch_size,))
        return torch.randint(upper, (batch_size,), generator=generator)


class PBitVarianceAwareSampler(BaseBatchSampler):
    """Simple variance-aware sampler scaffolding.

    The sampler oversamples candidate windows with high token
    variance to approximate a PBit-style variance-aware
    strategy. It draws a pool of candidate offsets uniformly,
    scores them by variance, and then samples without
    replacement weighted by those scores.
    """

    def __init__(
        self,
        oversample_factor: int = 4,
        min_candidate_windows: int = 64,
        temperature: float = 1.0,
        epsilon: float = 1e-8,
    ) -> None:
        if oversample_factor < 1:
            raise ValueError("oversample_factor must be >= 1")
        self.oversample_factor = oversample_factor
        self.min_candidate_windows = min_candidate_windows
        self.temperature = max(temperature, 1e-6)
        self.epsilon = epsilon
        logger.debug(
            "Initialised variance-aware sampler with oversample_factor=%s, min_candidate_windows=%s",
            oversample_factor,
            min_candidate_windows,
        )

    def _select_candidate_offsets(
        self,
        total_windows: int,
        batch_size: int,
        *,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        num_candidates = min(
            total_windows,
            max(batch_size * self.oversample_factor, self.min_candidate_windows),
        )
        if generator is None:
            return torch.randint(total_windows, (num_candidates,))
        return torch.randint(total_windows, (num_candidates,), generator=generator)

    def _score_windows(
        self,
        data: np.memmap,
        block_size: int,
        offsets: Iterable[int],
    ) -> torch.Tensor:
        scores = []
        for offset in offsets:
            window = np.asarray(data[offset : offset + block_size], dtype=np.float32)
            if window.size == 0:
                scores.append(0.0)
                continue
            variance = float(window.var())
            scores.append(variance)
        return torch.tensor(scores, dtype=torch.float32)

    def sample(
        self,
        data: np.memmap,
        block_size: int,
        batch_size: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        total_windows = max(1, len(data) - block_size)
        candidates = self._select_candidate_offsets(
            total_windows, batch_size, generator=generator
        )
        scores = self._score_windows(data, block_size, candidates.tolist())
        # Stabilise and normalise scores into sampling weights
        adjusted_scores = (scores + self.epsilon) / self.temperature
        weights = torch.softmax(adjusted_scores, dim=0)
        num_samples = min(batch_size, candidates.numel())
        # ``torch.multinomial`` requires CPU tensors
        cpu_weights = weights.cpu()
        cpu_candidates = candidates.cpu()
        if generator is None:
            selection = torch.multinomial(cpu_weights, num_samples, replacement=False)
        else:
            selection = torch.multinomial(
                cpu_weights, num_samples, replacement=False, generator=generator
            )
        chosen = cpu_candidates[selection]
        if chosen.numel() < batch_size:
            fallback = UniformBatchSampler().sample(
                data, block_size, batch_size - chosen.numel(), generator=generator
            )
            chosen = torch.cat([chosen, fallback.cpu()])
        return chosen.to(torch.long)


SAMPLER_REGISTRY: Dict[str, type[BaseBatchSampler]] = {
    "uniform": UniformBatchSampler,
    "variance_aware": PBitVarianceAwareSampler,
    "pbit": PBitVarianceAwareSampler,
}


def build_batch_sampler(name: Optional[str], **kwargs: object) -> BaseBatchSampler:
    """Instantiate a batch sampler by name.

    Parameters
    ----------
    name:
        Registry key for the sampler. Defaults to ``"uniform"`` when
        ``None`` is provided so existing runs stay identical.
    kwargs:
        Keyword arguments forwarded to the sampler constructor.
    """

    if not name:
        name = "uniform"
    key = name.lower()
    if key not in SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown sampler '{name}'. Available options: {sorted(SAMPLER_REGISTRY)}"
        )
    sampler_cls = SAMPLER_REGISTRY[key]
    return sampler_cls(**kwargs)


def get_available_sampler_names() -> List[str]:
    """Return the sorted list of registered sampler names."""

    return sorted(SAMPLER_REGISTRY.keys())


__all__ = [
    "BaseBatchSampler",
    "UniformBatchSampler",
    "PBitVarianceAwareSampler",
    "SamplerSpec",
    "build_batch_sampler",
    "SAMPLER_REGISTRY",
    "get_available_sampler_names",
]
