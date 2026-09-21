"""Detection of larger musical transitions such as drops and choruses."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import statistics


@dataclass(frozen=True, slots=True)
class MusicalMoment:
    kind: str
    intensity: float
    text: str
    duration: float


class MusicalMomentDetector:
    """Compare current energy with a rolling baseline to find real transitions."""

    _LINES = ("VOT ETO DROP!", "DAVAI DAVAI!", "ZHARA, BLYAT!")

    def __init__(self) -> None:
        self._history: deque[tuple[float, float, float]] = deque(maxlen=500)
        self._active_until = 0.0
        self._cooldown_until = 0.0
        self._cursor = 0
        self._current: MusicalMoment | None = None

    def update(self, features, role_scores: dict[str, float], now: float) -> MusicalMoment | None:
        energy = float(features.level) if features.active else 0.0
        rhythmic = max(float(features.onset), float(features.stem_drums))
        self._history.append((now, energy, rhythmic))
        if self._current is not None and now < self._active_until:
            return self._current
        self._current = None
        if not features.active or now < self._cooldown_until:
            return None

        baseline_values = [
            level for timestamp, level, _rhythm in self._history
            if 1.0 <= now - timestamp <= 7.0
        ]
        if len(baseline_values) < 25:
            return None
        baseline = statistics.median(baseline_values)
        lift = energy - baseline
        active_roles = sum(score >= 0.18 for score in role_scores.values())
        convincing = (
            energy >= 0.70
            and lift >= 0.18
            and rhythmic >= 0.42
            and active_roles >= 3
        )
        if not convincing:
            return None

        intensity = min(1.0, 0.58 + lift + rhythmic * 0.25)
        text = self._LINES[self._cursor % len(self._LINES)]
        self._cursor += 1
        self._current = MusicalMoment("drop", intensity, text, 1.8)
        self._active_until = now + self._current.duration
        self._cooldown_until = now + 8.0
        return self._current

    def is_active(self, now: float) -> bool:
        return self._current is not None and now < self._active_until
