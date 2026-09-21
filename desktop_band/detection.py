"""Confidence-based musical-role routing for the V2 animation system."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .analysis import _clamp
from .band import role_activity


@dataclass(frozen=True, slots=True)
class DetectionProfile:
    name: str
    play_threshold: float = 0.18
    groove_threshold: float = 0.075
    competition_margin: float = 0.11
    vocal_bias: float = 1.0
    bass_bias: float = 1.0
    drum_bias: float = 1.0
    other_bias: float = 1.0


PROFILES = {
    "balanced": DetectionProfile("balanced"),
    "taiko": DetectionProfile(
        "taiko", play_threshold=0.20, competition_margin=0.09,
        vocal_bias=0.72, bass_bias=0.76, drum_bias=1.18, other_bias=0.70,
    ),
    "hardbass": DetectionProfile(
        "hardbass", play_threshold=0.17, bass_bias=1.16, drum_bias=1.10,
        other_bias=0.88,
    ),
    "soft": DetectionProfile(
        "soft", play_threshold=0.14, groove_threshold=0.06,
        vocal_bias=1.08, drum_bias=0.86, other_bias=1.08,
    ),
}


@dataclass(frozen=True, slots=True)
class RoleDecision:
    scores: dict[str, float]
    raw_scores: dict[str, float]
    states: dict[str, str]
    dominant_role: str | None
    dominant_confidence: float


def _blend(current: float, target: float, dt: float) -> float:
    time_constant = 0.16 if target > current else 0.42
    amount = 1.0 - math.exp(-max(0.0, dt) / time_constant)
    return current + (target - current) * amount


def _compete(
    scores: dict[str, float], first: str, second: str, margin: float
) -> None:
    """Suppress ambiguous leakage while retaining convincing simultaneous parts."""

    a, b = scores.get(first, 0.0), scores.get(second, 0.0)
    if max(a, b) < 0.10:
        return
    difference = abs(a - b)
    if a >= 0.58 and b >= 0.58 and difference < margin * 0.7:
        scores[first] *= 0.82
        scores[second] *= 0.82
        return
    if difference < margin:
        scores[first] *= 0.62
        scores[second] *= 0.62
        return
    loser = second if a > b else first
    scores[loser] *= 0.16


def role_confidences(features, profile: DetectionProfile) -> dict[str, float]:
    """Build instantaneous role confidences from stems and acoustic evidence."""

    roles = (
        "singer", "drummer", "bassist", "keyboard",
        "guitarist", "percussion", "vibing",
    )
    if not features.active:
        return {role: 0.0 for role in roles}
    if not getattr(features, "separated", False):
        scores = {role: _clamp(role_activity(role, features)) for role in roles}
        _compete(scores, "guitarist", "keyboard", profile.competition_margin)
        _compete(scores, "drummer", "percussion", profile.competition_margin)
        return scores

    other = features.stem_other
    other_low = features.stem_other_low
    other_mid = features.stem_other_mid
    other_high = features.stem_other_high
    other_onset = getattr(features, "stem_other_onset", features.onset)
    other_harmonic = getattr(features, "stem_other_harmonic", features.harmonic)
    centroid = getattr(features, "stem_other_centroid", 0.0)
    flatness = getattr(features, "stem_other_flatness", 0.0)
    stability = getattr(features, "stem_other_pitch_stability", 0.0)
    density = getattr(features, "stem_other_note_density", 0.0)

    drum_low = features.percussive_low
    drum_mid = features.percussive_mid
    drum_high = features.percussive_high
    drum_total = max(1e-6, drum_low + drum_mid + drum_high)
    low_share = drum_low / drum_total
    bright_share = (drum_mid * 0.35 + drum_high) / drum_total

    guitar_shape = (
        0.20 + 0.48 * other_mid + 0.16 * density
        + 0.10 * flatness + 0.10 * other_onset
        - 0.12 * max(0.0, other_high - other_mid)
    )
    keyboard_shape = (
        0.18 + 0.22 * other_low + 0.26 * other_high
        + 0.24 * stability + 0.16 * other_harmonic
        - 0.10 * density + 0.08 * centroid
    )
    scores = {
        "singer": features.stem_vocals * profile.vocal_bias,
        "drummer": features.stem_drums * profile.drum_bias * (0.62 + 0.50 * low_share),
        "bassist": features.stem_bass * profile.bass_bias,
        "keyboard": other * profile.other_bias * keyboard_shape,
        "guitarist": other * profile.other_bias * guitar_shape,
        "percussion": features.stem_drums * profile.drum_bias * (
            0.18 + 0.68 * bright_share + 0.16 * features.onset
        ),
        "vibing": features.level,
    }
    scores = {role: _clamp(value) for role, value in scores.items()}
    _compete(scores, "guitarist", "keyboard", profile.competition_margin)
    _compete(scores, "drummer", "percussion", profile.competition_margin)

    # A mostly percussive mixture is not credible evidence of voice or melody,
    # even when a separator leaks a small amount into those stems.
    percussion_dominance = features.stem_drums / max(
        1e-6,
        features.stem_drums + features.stem_bass
        + features.stem_vocals + features.stem_other,
    )
    if percussion_dominance >= 0.72:
        rejection = _clamp((percussion_dominance - 0.72) / 0.20)
        for role in ("singer", "bassist", "keyboard", "guitarist"):
            scores[role] *= 1.0 - rejection * 0.88
    return scores


class RoleRouter:
    """Accumulate evidence and emit stable idle/groove/playing decisions."""

    def __init__(self, profile: str = "balanced") -> None:
        self.profile = PROFILES.get(profile, PROFILES["balanced"])
        self._evidence: dict[str, float] = {}

    def set_profile(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError("profil de détection inconnu : {}".format(profile))
        self.profile = PROFILES[profile]
        self._evidence.clear()

    def reset(self) -> None:
        self._evidence.clear()

    def update(self, features, roles, dt: float) -> RoleDecision:
        raw = role_confidences(features, self.profile)
        for role in roles:
            self._evidence[role] = _blend(
                self._evidence.get(role, 0.0), raw.get(role, 0.0), dt
            )
        scores = {role: _clamp(self._evidence.get(role, 0.0)) for role in roles}
        states = {}
        for role, score in scores.items():
            if not features.active or score < self.profile.groove_threshold:
                states[role] = "idle"
            elif score < self.profile.play_threshold:
                states[role] = "groove"
            else:
                states[role] = "playing"

        candidates = [
            (score, role) for role, score in scores.items()
            if role != "vibing" and states[role] == "playing"
        ]
        candidates.sort(reverse=True)
        dominant = None
        confidence = 0.0
        if candidates:
            confidence, dominant = candidates[0]
            runner_up = candidates[1][0] if len(candidates) > 1 else 0.0
            if confidence < 0.34 or confidence - runner_up < 0.09:
                dominant = None
        return RoleDecision(scores, raw, states, dominant, confidence)
