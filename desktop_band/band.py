"""Band composition rules, independent from audio and UI code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Musician:
    role: str
    label: str
    color: str


_MUSICIANS = {
    "singer": Musician("singer", "CHANT", "#ef476f"),
    "drummer": Musician("drummer", "BATTERIE", "#ffd166"),
    "bassist": Musician("bassist", "BASSE", "#06d6a0"),
    "keyboard": Musician("keyboard", "CLAVIER", "#118ab2"),
    "guitarist": Musician("guitarist", "GUITARE", "#9b5de5"),
    "percussion": Musician("percussion", "PERCUS", "#f9844a"),
    "vibing": Musician("vibing", "VIBE", "#a3e635"),
}

ROLE_ORDER = (
    "vibing", "guitarist", "bassist", "singer",
    "drummer", "percussion", "keyboard",
)

_LINEUPS = {
    1: ("singer",),
    2: ("singer", "drummer"),
    3: ("bassist", "singer", "drummer"),
    4: ("bassist", "singer", "drummer", "keyboard"),
    5: ("guitarist", "bassist", "singer", "drummer", "keyboard"),
    6: ("percussion", "guitarist", "bassist", "singer", "drummer", "keyboard"),
    7: (
        "vibing",
        "guitarist",
        "bassist",
        "singer",
        "drummer",
        "percussion",
        "keyboard",
    ),
}

# Scores below these values are too ambiguous to justify visible playing.
# Percussive roles are slightly more sensitive because their events are short.
_PLAY_THRESHOLDS = {
    "singer": 0.12,
    "drummer": 0.09,
    "bassist": 0.11,
    "keyboard": 0.11,
    "guitarist": 0.11,
    "percussion": 0.09,
    "vibing": 0.05,
}


def create_lineup(count: int) -> tuple[Musician, ...]:
    """Return a balanced, deterministic lineup of one to six musicians."""

    if count not in _LINEUPS:
        raise ValueError("le groupe doit contenir entre 1 et 7 musiciens")
    return tuple(_MUSICIANS[role] for role in _LINEUPS[count])


def create_custom_lineup(roles) -> tuple[Musician, ...]:
    """Build a stable left-to-right lineup from an arbitrary role selection."""

    selected = {str(role) for role in roles}
    unknown = selected.difference(_MUSICIANS)
    if unknown:
        raise ValueError("rôle inconnu : {}".format(sorted(unknown)[0]))
    if not selected:
        raise ValueError("le groupe doit contenir au moins un gopnik")
    return tuple(_MUSICIANS[role] for role in ROLE_ORDER if role in selected)


def role_activity(role: str, features) -> float:
    """Map analyzed audio roles to a visual instrument."""

    if not features.active:
        return 0.0
    if role == "vibing":
        return features.level
    if getattr(features, "separated", False):
        if role == "singer":
            return features.stem_vocals
        if role == "drummer":
            return features.stem_drums
        if role == "bassist":
            return features.stem_bass
        if role == "keyboard":
            color = 0.30 + 0.45 * features.stem_other_high + 0.25 * features.stem_other_low
            return features.stem_other * min(1.0, color)
        if role == "guitarist":
            color = 0.25 + 0.75 * features.stem_other_mid
            return features.stem_other * min(1.0, color)
        if role == "percussion":
            return features.stem_drums * (
                0.35 + 0.65 * max(features.percussive_high, features.percussive_mid)
            )
        return features.level
    percussive = features.percussive
    percussive_memory = features.percussive_memory
    harmonic = features.harmonic
    total_separated = percussive + harmonic
    harmonic_share = harmonic / max(1e-9, total_separated)
    # When the separated signal is overwhelmingly percussive, reject weak
    # melodic leakage. When both components are present, allow both to play.
    melodic_gate = harmonic_share * harmonic_share if percussive > harmonic else 1.0
    if percussive_memory > harmonic and harmonic < 0.35:
        tail_share = harmonic / max(1e-9, percussive_memory + harmonic)
        melodic_gate *= tail_share * tail_share
    percussion_only = (
        percussive_memory >= 0.22
        and percussive_memory > harmonic * 1.8
        and harmonic < 0.4
    )
    if role == "singer":
        if percussion_only:
            return 0.0
        return features.vocal * melodic_gate
    if role == "drummer":
        drum_body = (
            0.55 * features.percussive_low
            + 0.2 * features.percussive_mid
            + 0.25 * features.percussive_high
        )
        memory_body = percussive_memory * (
            0.55 * features.low + 0.2 * features.mid + 0.25 * features.high
        )
        return max(features.onset * 0.9, drum_body, memory_body)
    if role == "bassist":
        if percussion_only:
            return 0.0
        return melodic_gate * (
            0.9 * features.harmonic_low + 0.1 * harmonic
        )
    if role == "keyboard":
        if percussion_only:
            return 0.0
        return melodic_gate * (
            0.68 * features.harmonic_mid
            + 0.27 * features.harmonic_high
            + 0.05 * harmonic
        )
    if role == "guitarist":
        if percussion_only:
            return 0.0
        return melodic_gate * (
            0.85 * features.harmonic_mid + 0.15 * harmonic
        )
    if role == "percussion":
        percussion_body = (
            0.65 * features.percussive_high + 0.35 * features.percussive_mid
        )
        return max(features.onset * 0.85, percussion_body)
    return features.level


def gated_activity(role: str, score: float, was_playing: bool) -> tuple[float, bool]:
    """Apply a role threshold with hysteresis and normalize visible motion."""

    score = max(0.0, min(1.0, float(score)))
    start_threshold = _PLAY_THRESHOLDS.get(role, 0.11)
    stop_threshold = start_threshold * 0.65
    threshold = stop_threshold if was_playing else start_threshold
    playing = score >= threshold
    if not playing:
        return 0.0, False

    # Once a role is credible, give it a clearly readable minimum motion.
    normalized = (score - threshold) / max(1e-9, 1.0 - threshold)
    return min(1.0, 0.22 + 0.78 * normalized), True
