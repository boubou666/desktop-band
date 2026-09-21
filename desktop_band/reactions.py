"""Deterministic musical-moment reactions for the band overlay."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShoutEvent:
    text: str
    role: str
    duration: float = 0.95


class ShoutDetector:
    """Turn meaningful musical transitions into occasional speech bubbles."""

    _IMPACT_LINES = ("BLYAT!", "CYKA!", "EBASH!")
    _BASS_LINES = ("O-PA!", "DAVAI BAS!", "ZHGI!")
    _VOCAL_LINES = ("NU DAVAI!", "GOLOS!", "KRASAVA!")
    _CLIMAX_LINES = ("ZHARA!", "DAVAI!", "MOLODETS!")

    def __init__(self, cooldown: float = 3.6) -> None:
        self.cooldown = cooldown
        self._last_event_at = -100.0
        self._previous_active = False
        self._seen_audio = False
        self._silence_started: float | None = None
        self._previous_bass = 0.0
        self._previous_vocals = 0.0
        self._previous_level = 0.0
        self._line_cursor = 0

    def update(self, features, role_scores: dict[str, float], now: float) -> ShoutEvent | None:
        active = bool(features.active)
        bass = (
            features.stem_bass
            if getattr(features, "separated", False)
            else features.harmonic_low
        )
        vocals = (
            features.stem_vocals
            if getattr(features, "separated", False)
            else features.vocal
        )
        drums = (
            features.stem_drums
            if getattr(features, "separated", False)
            else features.percussive
        )

        if not active:
            if self._previous_active:
                self._silence_started = now
            self._previous_active = False
            self._remember(bass, vocals, features.level)
            return None

        ready = now - self._last_event_at >= self.cooldown
        event = None

        if not self._previous_active:
            silence_duration = (
                now - self._silence_started
                if self._silence_started is not None
                else 0.0
            )
            if self._seen_audio and silence_duration >= 1.0 and ready:
                event = ShoutEvent("POEHALI!", self._available_role("vibing", role_scores))
            self._seen_audio = True

        if event is None and ready:
            impact = features.onset * (0.55 + 0.45 * drums)
            active_roles = sum(score >= 0.12 for score in role_scores.values())
            if impact >= 0.62 and drums >= 0.30:
                role = self._stronger_role(("drummer", "percussion"), role_scores)
                event = ShoutEvent(self._next(self._IMPACT_LINES), role)
            elif bass >= 0.72 and self._previous_bass < 0.42:
                event = ShoutEvent(
                    self._next(self._BASS_LINES),
                    self._available_role("bassist", role_scores),
                )
            elif vocals >= 0.72 and self._previous_vocals < 0.38:
                event = ShoutEvent(
                    self._next(self._VOCAL_LINES),
                    self._available_role("singer", role_scores),
                )
            elif (
                features.level >= 0.90
                and self._previous_level < 0.72
                and active_roles >= 3
            ):
                event = ShoutEvent(
                    self._next(self._CLIMAX_LINES),
                    self._stronger_role(tuple(role_scores), role_scores),
                )

        if event is not None:
            self._last_event_at = now
        self._previous_active = True
        self._remember(bass, vocals, features.level)
        return event

    def _remember(self, bass: float, vocals: float, level: float) -> None:
        self._previous_bass = bass
        self._previous_vocals = vocals
        self._previous_level = level

    def _next(self, lines: tuple[str, ...]) -> str:
        text = lines[self._line_cursor % len(lines)]
        self._line_cursor += 1
        return text

    @staticmethod
    def _available_role(preferred: str, role_scores: dict[str, float]) -> str:
        if preferred in role_scores:
            return preferred
        return ShoutDetector._stronger_role(tuple(role_scores), role_scores)

    @staticmethod
    def _stronger_role(
        candidates: tuple[str, ...], role_scores: dict[str, float]
    ) -> str:
        available = [role for role in candidates if role in role_scores]
        if not available:
            return next(iter(role_scores), "singer")
        return max(available, key=lambda role: role_scores[role])
