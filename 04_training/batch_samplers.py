"""Batch sampling utilities for training runs.

Provides a registry-driven interface so training scripts can
switch between the stock uniform sampler and a rich PBit-inspired
variance-aware sampler through configuration alone.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SamplerSpec:
    """Simple spec used by :func:`build_batch_sampler`."""

    name: str
    kwargs: Dict[str, object]


@dataclass
class PBitConfig:
    """Configuration knobs for the variance-aware sampler."""

    heavy_refresh_interval: int = 50
    shortlist_seed_size: int = 128
    shortlist_knn: int = 8
    shortlist_cap: int = 1024
    sketch_dim: int = 64
    rarity_bucket_count: int = 4096
    rarity_ema: float = 0.01
    feature_weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    diversity_strength: float = 0.5
    epsilon: float = 1e-8
    temperature: float = 1.0
    random_seed: Optional[int] = None
    log_interval: int = 100
    group_quotas: Optional[Dict[int, int]] = None


@dataclass
class SamplerState:
    """Bookkeeping that persists across sampling steps."""

    step: int = 0
    last_refresh_step: int = -1
    rarity_buckets: torch.Tensor | None = None
    feature_tensor: torch.Tensor | None = None
    rank_features: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    shortlist_indices: torch.Tensor | None = None
    shortlist_scores: torch.Tensor | None = None
    shortlist_similarity: torch.Tensor | None = None
    group_ids: torch.Tensor | None = None


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
    """Variance-aware sampler with shortlist, rarity, and diversity logic."""

    def __init__(
        self,
        *,
        heavy_refresh_interval: int = 50,
        shortlist_seed_size: int = 128,
        shortlist_knn: int = 8,
        shortlist_cap: int = 1024,
        sketch_dim: int = 64,
        rarity_bucket_count: int = 4096,
        rarity_ema: float = 0.01,
        feature_weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        diversity_strength: float = 0.5,
        epsilon: float = 1e-8,
        temperature: float = 1.0,
        random_seed: Optional[int] = None,
        log_interval: int = 100,
        group_quotas: Optional[Dict[int, int]] = None,
    ) -> None:
        self.config = PBitConfig(
            heavy_refresh_interval=heavy_refresh_interval,
            shortlist_seed_size=shortlist_seed_size,
            shortlist_knn=shortlist_knn,
            shortlist_cap=shortlist_cap,
            sketch_dim=sketch_dim,
            rarity_bucket_count=rarity_bucket_count,
            rarity_ema=rarity_ema,
            feature_weights=tuple(feature_weights[:4])
            if len(feature_weights) >= 4
            else (0.25, 0.25, 0.25, 0.25),
            diversity_strength=float(np.clip(diversity_strength, 0.0, 1.0)),
            epsilon=max(epsilon, 1e-10),
            temperature=max(temperature, 1e-6),
            random_seed=random_seed,
            log_interval=log_interval,
            group_quotas=group_quotas,
        )
        self.state = SamplerState()
        self._projection_matrix: torch.Tensor | None = None
        logger.debug(
            "Initialised PBitVarianceAwareSampler with config=%s",
            self.config,
        )

    # ------------------------------------------------------------------
    # Feature computation utilities
    # ------------------------------------------------------------------
    def _ensure_projection(self, feature_dim: int) -> torch.Tensor:
        if self._projection_matrix is not None:
            return self._projection_matrix
        torch_rng = torch.Generator()
        if self.config.random_seed is not None:
            torch_rng.manual_seed(self.config.random_seed)
        matrix = torch.randn(
            feature_dim, self.config.sketch_dim, generator=torch_rng
        )
        matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-6)
        self._projection_matrix = matrix
        return matrix

    def _initialise_state(self, total_windows: int) -> None:
        if self.state.rarity_buckets is None:
            self.state.rarity_buckets = torch.ones(
                self.config.rarity_bucket_count, dtype=torch.float32
            )
        if self.state.group_ids is None or self.state.group_ids.numel() != total_windows:
            self.state.group_ids = torch.zeros(total_windows, dtype=torch.long)

    def _compute_window_features(
        self, data: np.memmap, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_windows = max(1, len(data) - block_size)
        lengths = torch.zeros(total_windows, dtype=torch.float32)
        entropy = torch.zeros_like(lengths)
        mi_proxy = torch.zeros_like(lengths)
        variance = torch.zeros_like(lengths)

        for idx in range(total_windows):
            window = np.asarray(data[idx : idx + block_size], dtype=np.int64)
            valid_len = window.size
            if valid_len == 0:
                continue
            lengths[idx] = float(valid_len)
            variance[idx] = float(np.var(window))
            values, counts = np.unique(window, return_counts=True)
            probs = counts / counts.sum()
            entropy[idx] = float(-(probs * np.log2(np.clip(probs, 1e-12, 1))).sum())
            if valid_len > 1:
                diffs = np.diff(window.astype(np.float32))
                mi_proxy[idx] = float(np.mean(np.abs(diffs)))

        return lengths, entropy, mi_proxy, variance

    def _rarity_scores(self, data: np.memmap, block_size: int) -> torch.Tensor:
        total_windows = max(1, len(data) - block_size)
        rarity = torch.ones(total_windows, dtype=torch.float32)
        bucket_state = self.state.rarity_buckets
        assert bucket_state is not None
        for idx in range(total_windows):
            token = int(data[idx]) if idx < len(data) else 0
            bucket = token % self.config.rarity_bucket_count
            rarity[idx] = 1.0 / bucket_state[bucket].clamp_min(self.config.epsilon)
        return rarity

    def _rank_normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(tensor)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(len(tensor), dtype=torch.float32)
        if tensor.numel() == 1:
            return torch.ones_like(tensor, dtype=torch.float32)
        return ranks / (tensor.numel() - 1)

    def _combine_scores(self, rank_features: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor(
            self.config.feature_weights, dtype=torch.float32, device=rank_features.device
        )
        weights = weights / weights.sum().clamp_min(self.config.epsilon)
        return (rank_features * weights).sum(dim=1)

    def _build_shortlist(
        self, scores: torch.Tensor, rank_features: torch.Tensor, total_windows: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        top_k = min(self.config.shortlist_seed_size, total_windows)
        if top_k == 0:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, 0, dtype=torch.float32),
            )
        _, seed_indices = torch.topk(scores, k=top_k)

        sketches = self._compute_sketches(rank_features)
        shortlist = set(seed_indices.tolist())

        for seed in seed_indices.tolist():
            neighbours = self._knn(seed, sketches)
            shortlist.update(neighbours)
            if len(shortlist) >= self.config.shortlist_cap:
                break

        shortlist_indices = torch.tensor(sorted(shortlist), dtype=torch.long)
        shortlist_scores = scores[shortlist_indices]
        if shortlist_indices.numel() > self.config.shortlist_cap:
            top_scores, keep_idx = torch.topk(
                shortlist_scores,
                k=self.config.shortlist_cap,
            )
            shortlist_indices = shortlist_indices[keep_idx]
            shortlist_scores = top_scores
        shortlist_features = rank_features[shortlist_indices]
        similarity = self._cosine_matrix(shortlist_features)
        return shortlist_indices, shortlist_scores, similarity

    def _compute_sketches(self, rank_features: torch.Tensor) -> torch.Tensor:
        projection = self._ensure_projection(rank_features.size(1))
        sketches = rank_features @ projection
        return torch.sign(sketches)

    def _knn(
        self,
        seed_index: int,
        sketches: torch.Tensor,
    ) -> List[int]:
        if sketches.numel() == 0:
            return []
        seed_sketch = sketches[seed_index]
        similarities = (sketches @ seed_sketch) / sketches.size(1)
        topk = torch.topk(similarities, k=min(self.config.shortlist_knn, sketches.size(0)))
        neighbours = [idx for idx in topk.indices.tolist() if idx != seed_index]
        return neighbours

    def _cosine_matrix(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return torch.empty(0, 0, dtype=torch.float32)
        normed = features / features.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return normed @ normed.t()

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------
    def _maybe_refresh(
        self,
        data: np.memmap,
        block_size: int,
    ) -> None:
        total_windows = max(1, len(data) - block_size)
        self._initialise_state(total_windows)
        if self.state.step == 0 or (
            self.state.step - self.state.last_refresh_step
        ) >= self.config.heavy_refresh_interval:
            self._heavy_refresh(data, block_size)

    def _heavy_refresh(self, data: np.memmap, block_size: int) -> None:
        total_windows = max(1, len(data) - block_size)
        lengths, entropy, mi_proxy, variance = self._compute_window_features(
            data, block_size
        )
        rarity = self._rarity_scores(data, block_size)
        features = torch.stack([lengths, entropy, mi_proxy, rarity], dim=1)

        rank_features = torch.stack(
            [self._rank_normalise(col) for col in features.t()], dim=1
        )
        scores = self._combine_scores(rank_features)
        shortlist_indices, shortlist_scores, similarity = self._build_shortlist(
            scores, rank_features, total_windows
        )

        self.state.feature_tensor = features
        self.state.rank_features = rank_features
        self.state.scores = scores
        self.state.shortlist_indices = shortlist_indices
        self.state.shortlist_scores = shortlist_scores
        self.state.shortlist_similarity = similarity
        self.state.last_refresh_step = self.state.step

        if self.config.group_quotas:
            group_ids = self._resolve_groups(data, block_size, total_windows)
            self.state.group_ids = group_ids

        logger.debug(
            "Heavy refresh complete: shortlist=%s entries", shortlist_indices.numel()
        )

    def _resolve_groups(
        self, data: np.memmap, block_size: int, total_windows: int
    ) -> torch.Tensor:
        quotas = self.config.group_quotas
        if not quotas:
            return torch.zeros(total_windows, dtype=torch.long)
        num_groups = len(quotas)
        group_ids = torch.zeros(total_windows, dtype=torch.long)
        for idx in range(total_windows):
            token = int(data[idx]) if idx < len(data) else 0
            group_ids[idx] = token % num_groups
        return group_ids

    def _select_diverse(self, batch_size: int) -> torch.Tensor:
        shortlist_indices = self.state.shortlist_indices
        shortlist_scores = self.state.shortlist_scores
        similarity = self.state.shortlist_similarity
        if shortlist_indices is None or shortlist_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        num_select = min(batch_size, shortlist_indices.numel())
        scores = shortlist_scores.clone()
        scaled = scores / self.config.temperature
        norm_scores = torch.softmax(scaled, dim=0)
        selected: List[int] = []
        remaining = set(range(shortlist_indices.numel()))

        best_idx = int(torch.argmax(scores))
        selected.append(best_idx)
        remaining.remove(best_idx)

        while remaining and len(selected) < num_select:
            best_gain = -float("inf")
            best_candidate = None
            for candidate in list(remaining):
                diversity = 0.0
                if similarity.numel() > 0:
                    existing = torch.tensor(selected, dtype=torch.long)
                    candidate_sim = similarity[candidate, existing]
                    diversity = float(1.0 - candidate_sim.max().item())
                score_component = float(norm_scores[candidate].item())
                gain = (
                    self.config.diversity_strength * diversity
                    + (1 - self.config.diversity_strength) * score_component
                )
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate
            if best_candidate is None:
                break
            selected.append(best_candidate)
            remaining.remove(best_candidate)

        chosen = shortlist_indices[torch.tensor(selected, dtype=torch.long)]
        return chosen

    def _apply_quotas(self, offsets: torch.Tensor) -> torch.Tensor:
        quotas = self.config.group_quotas
        group_ids = self.state.group_ids
        if not quotas or group_ids is None or offsets.numel() == 0:
            return offsets

        desired = torch.tensor(list(quotas.values()), dtype=torch.long)
        group_keys = list(quotas.keys())
        counts = torch.zeros_like(desired)
        for off in offsets:
            gid = int(group_ids[off])
            if gid in group_keys:
                pos = group_keys.index(gid)
                counts[pos] += 1

        deficit_mask = counts < desired
        if not torch.any(deficit_mask):
            return offsets

        deficits = {group_keys[i]: int(desired[i] - counts[i]) for i in range(len(group_keys))}
        available = []
        shortlist = self.state.shortlist_indices
        if shortlist is None:
            return offsets
        existing = set(offsets.tolist())
        for idx, off in enumerate(shortlist.tolist()):
            gid = int(group_ids[off])
            if gid in deficits and deficits[gid] > 0 and off not in existing:
                available.append((gid, idx, off))

        offsets_list = offsets.tolist()
        existing = set(offsets_list)
        for gid, idx, off in available:
            if deficits[gid] <= 0:
                continue
            if off in existing:
                continue
            offsets_list.append(off)
            existing.add(off)
            deficits[gid] -= 1
            if len(offsets_list) >= len(offsets):
                break

        return torch.tensor(offsets_list[: len(offsets)], dtype=torch.long)

    def _update_rarity_counts(self, offsets: torch.Tensor, data: np.memmap) -> None:
        if offsets.numel() == 0:
            return
        bucket_state = self.state.rarity_buckets
        assert bucket_state is not None
        rate = self.config.rarity_ema
        for off in offsets.tolist():
            token = int(data[off]) if off < len(data) else 0
            bucket = token % self.config.rarity_bucket_count
            bucket_state[bucket] = (1 - rate) * bucket_state[bucket] + rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample(
        self,
        data: np.memmap,
        block_size: int,
        batch_size: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        self._maybe_refresh(data, block_size)
        if self.state.shortlist_indices is None or self.state.shortlist_indices.numel() == 0:
            logger.warning("Shortlist empty – falling back to uniform sampling")
            return UniformBatchSampler().sample(
                data, block_size, batch_size, generator=generator
            )

        chosen = self._select_diverse(batch_size)
        chosen = self._apply_quotas(chosen)

        if chosen.numel() < batch_size:
            fallback = UniformBatchSampler().sample(
                data, block_size, batch_size - chosen.numel(), generator=generator
            )
            chosen = torch.cat([chosen, fallback])

        self._update_rarity_counts(chosen, data)

        self.state.step += 1
        if self.state.step % self.config.log_interval == 0:
            logger.info(
                "PBit sampler step %s – shortlist size=%s", self.state.step, chosen.numel()
            )

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
