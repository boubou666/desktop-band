"""Low-frequency stage direction: interactions, rare events and variants."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True, slots=True)
class StageEvent:
    kind: str
    roles: tuple[str, ...]
    text: str
    until: float


class StageDirector:
    _RARE_EVENTS = (
        ("squat_wave", "PRISYADKA PROTOCOL!"),
        ("vodka_toast", "ZA ZDOROVYE!"),
        ("bat_tap", "SERIOUS BUSINESS."),
        ("red_alert", "KRASNAYA TREVOGA!"),
        ("stick_toss", "NYET, GRAVITY."),
        ("vodka_round", "HYDRATION PROTOCOL."),
    )
    _INTERACTIONS = (
        ("fist_bump", "RESPECT."),
        ("nod", "DA."),
        ("challenge", "NU, DAVAI."),
    )
    _ACCESSORIES = ("none", "gold", "red_star", "bandana", "extra_smoke")

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)
        self._next_interaction = 0.0
        self._next_rare = 0.0
        self._event: StageEvent | None = None
        self._variants: dict[str, str] = {}

    def variant_for(self, role: str) -> str:
        if role not in self._variants:
            self._variants[role] = self._random.choice(self._ACCESSORIES)
        return self._variants[role]

    def update(self, features, decision, now: float) -> StageEvent | None:
        if self._event is not None and now < self._event.until:
            return self._event
        self._event = None
        if not features.active:
            return None

        playing = [
            role for role, state in decision.states.items()
            if state == "playing" and role != "vibing"
        ]
        if self._next_rare == 0.0:
            self._next_rare = now + self._random.uniform(35.0, 65.0)
        if self._next_interaction == 0.0:
            self._next_interaction = now + self._random.uniform(9.0, 16.0)

        if now >= self._next_rare and features.level >= 0.55:
            kind, text = self._random.choice(self._RARE_EVENTS)
            roles = tuple(decision.states)
            self._event = StageEvent(kind, roles, text, now + 2.4)
            self._next_rare = now + self._random.uniform(42.0, 85.0)
            self._next_interaction = max(self._next_interaction, now + 6.0)
            return self._event

        if now >= self._next_interaction and len(playing) >= 2:
            lead = decision.dominant_role if decision.dominant_role in playing else playing[0]
            others = [role for role in playing if role != lead]
            partner = self._random.choice(others)
            if lead != "singer" and "singer" in playing:
                kind, text = "point_solo", "DAVAI, SOLO."
                roles = ("singer", lead)
            elif "drummer" in playing and getattr(features, "onset", 0.0) >= 0.62:
                kind, text = "stick_toss", "STICK GOES UP."
                roles = ("drummer",)
            elif "vibing" in decision.states and len(playing) >= 3:
                kind, text = "vodka_round", "ONE FOR COMRADES."
                roles = tuple(playing)
            else:
                kind, text = self._random.choice(self._INTERACTIONS)
                roles = (lead, partner)
            self._event = StageEvent(kind, roles, text, now + 1.8)
            self._next_interaction = now + self._random.uniform(12.0, 24.0)
        return self._event
