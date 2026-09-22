"""Persistent named band formations shared by the Lab and tray."""

from __future__ import annotations

import json
from pathlib import Path

from .band import ROLE_ORDER


PRESET_FORMATIONS = {
    "Hardbass": {
        "roles": ["vibing", "bassist", "singer", "drummer", "keyboard"],
        "decor": "panelki",
        "lighting": True,
    },
    "Rhythm section": {
        "roles": ["bassist", "drummer", "percussion"],
        "decor": "garage",
        "lighting": True,
    },
    "Instrumental": {
        "roles": ["guitarist", "bassist", "drummer", "percussion", "keyboard"],
        "decor": "none",
        "lighting": True,
    },
    "Comité complet": {
        "roles": list(ROLE_ORDER),
        "decor": "panelki",
        "lighting": True,
    },
}


class FormationStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list(self) -> dict[str, dict[str, object]]:
        formations = {name: dict(value) for name, value in PRESET_FORMATIONS.items()}
        if self.path.is_file():
            try:
                document = json.loads(self.path.read_text(encoding="utf-8"))
                for name, value in document.get("formations", {}).items():
                    formations[str(name)] = self._validate(value)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        return formations

    def save(self, name: str, value: dict[str, object]) -> dict[str, object]:
        clean_name = " ".join(str(name).strip().split())[:60]
        if not clean_name:
            raise ValueError("nom de formation vide")
        validated = self._validate(value)
        custom = {}
        if self.path.is_file():
            try:
                custom = json.loads(self.path.read_text(encoding="utf-8")).get(
                    "formations", {}
                )
            except (OSError, json.JSONDecodeError):
                custom = {}
        custom[clean_name] = validated
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"formations": custom}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"name": clean_name, **validated}

    @staticmethod
    def _validate(value: dict[str, object]) -> dict[str, object]:
        roles = [role for role in value.get("roles", []) if role in ROLE_ORDER]
        roles = list(dict.fromkeys(roles))
        if not roles:
            raise ValueError("une formation doit contenir au moins un gopnik")
        decor = str(value.get("decor", "none"))
        if decor not in {"none", "garage", "panelki"}:
            raise ValueError("décor inconnu")
        placements = {}
        raw_placements = value.get("placements", {})
        if isinstance(raw_placements, dict):
            for role in roles:
                candidate = raw_placements.get(role, {})
                if not isinstance(candidate, dict):
                    candidate = {}
                placements[role] = {
                    "x": max(0.04, min(0.96, float(candidate.get("x", 0.5)))),
                    "scale": max(0.8, min(1.2, float(candidate.get("scale", 1.0)))),
                    "depth": max(-3, min(3, int(candidate.get("depth", 0)))),
                }
        return {
            "roles": roles,
            "decor": decor,
            "lighting": bool(value.get("lighting", True)),
            "placements": placements,
        }
