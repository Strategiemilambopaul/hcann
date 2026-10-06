"""Découpage du flux en épisodes à partir de la nouveauté mémoire."""

from __future__ import annotations


class EventSegmenter:
    """Frontière si la nouveauté dépasse l'EMA de k écarts-types et un seuil absolu."""

    def __init__(self, config):
        self.k = float(getattr(config, "seg_k", 1.5))
        self.seg_ema = float(getattr(config, "seg_ema", 0.9))
        self.min_len = int(getattr(config, "min_event_len", 3))
        self.threshold = float(getattr(config, "mem_novelty_threshold", 0.25))
        self._ema: float | None = None
        self._var = 0.0
        self._since = 0

    def update(self, novelty: float) -> bool:
        self._since += 1
        value = float(novelty)
        boundary = False
        if self._ema is not None:
            std = self._var ** 0.5
            boundary = (
                self._since >= self.min_len
                and value > self._ema + self.k * std
                and value > self.threshold
            )
        if self._ema is None:
            self._ema = value
            self._var = 0.0
        else:
            delta = value - self._ema
            keep = self.seg_ema
            self._var = keep * self._var + (1.0 - keep) * delta * delta
            self._ema = keep * self._ema + (1.0 - keep) * value
        if boundary:
            self._since = 1
        return boundary


class EpisodeIndex:
    """Index des segments. Un segment = suite d'événements entre deux frontières."""

    def __init__(self, config):
        self.segmenter = EventSegmenter(config)
        self._seg_count = 0
        self._current = "seg_0"
        self._event_idx = 0
        self._segments: dict[str, list[str]] = {self._current: []}

    def add_event(self, node_id: str, novelty_mem: float) -> dict:
        boundary = self.segmenter.update(novelty_mem)
        if boundary:
            self._seg_count += 1
            self._current = f"seg_{self._seg_count}"
            self._segments[self._current] = []
            self._event_idx = 0
        self._segments[self._current].append(str(node_id))
        info = {
            "segment_id": self._current,
            "event_idx": self._event_idx,
            "boundary": boundary,
        }
        self._event_idx += 1
        return info

    def members(self, segment_id: str) -> list[str]:
        return list(self._segments.get(segment_id, []))

    def __len__(self) -> int:
        return len(self._segments)
