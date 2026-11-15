"""Reference implementation (NumPy) of the OOD score and event extraction.

Shared by the edge runtime, CARLA threshold tuning and evaluation.
  pixel score   E(x) = -T * logsumexp(logits / T)   (higher = more OOD)
  frame score   mean of the top-k% pixel scores
  event         run of above-threshold frames + margins
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OodConfig:
    temperature: float
    top_k_percent: float
    threshold: float
    margin_sec: float
    min_frames: int

    @classmethod
    def load(cls, path: Path = ROOT / "configs/phase1_ood.yaml") -> "OodConfig":
        d = yaml.safe_load(path.read_text())
        return cls(d["score"]["temperature"], d["score"]["top_k_percent"], d["event"]["threshold"],
                   d["event"]["margin_sec"], d["event"]["min_frames"])


@dataclass(frozen=True)
class Event:
    start_us: int
    end_us: int
    score: float  # maximum frame score inside the event


def pixel_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """logits: (C, H, W) -> (H, W), using a numerically stable logsumexp."""
    z = logits.astype(np.float64) / temperature
    m = z.max(axis=0)
    return -temperature * (m + np.log(np.exp(z - m).sum(axis=0)))


def frame_score(energy: np.ndarray, top_k_percent: float) -> float:
    flat = energy.ravel()
    n = max(1, int(np.ceil(flat.size * top_k_percent / 100.0)))
    return float(np.partition(flat, flat.size - n)[flat.size - n:].mean())


def extract_events(timestamps_us: list[int], scores: list[float], cfg: OodConfig) -> list[Event]:
    """Runs of at least min_frames above-threshold frames, with margins. Overlapping events are merged."""
    margin = int(cfg.margin_sec * 1_000_000)
    runs: list[tuple[int, int]] = []
    i, n = 0, len(scores)
    while i < n:
        if scores[i] > cfg.threshold:
            j = i
            while j + 1 < n and scores[j + 1] > cfg.threshold:
                j += 1
            if j - i + 1 >= cfg.min_frames:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    events: list[Event] = []
    for a, b in runs:
        s, e = timestamps_us[a] - margin, timestamps_us[b] + margin
        peak = max(scores[a:b + 1])
        if events and s <= events[-1].end_us:
            prev = events.pop()
            events.append(Event(prev.start_us, e, max(prev.score, peak)))
        else:
            events.append(Event(s, e, peak))
    return events
