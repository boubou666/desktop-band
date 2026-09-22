"""Sprite-pack discovery and manifest loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


ROLES = {
    "singer", "drummer", "bassist", "keyboard",
    "guitarist", "percussion", "vibing",
}


@dataclass(frozen=True, slots=True)
class SpriteSheet:
    path: Path
    rows: int
    role_rows: dict[str, int]


def _packs_dir() -> Path:
    return Path(__file__).with_name("assets") / "packs"


def available_sprite_packs() -> tuple[str, ...]:
    packs = _packs_dir()
    names = []
    if packs.is_dir():
        names = sorted(
            child.name for child in packs.iterdir()
            if child.is_dir() and (child / "manifest.json").is_file()
        )
    if "gopnik" in names:
        names.remove("gopnik")
        names.insert(0, "gopnik")
    return tuple(names)


def load_sprite_pack(name_or_path: str = "gopnik") -> tuple[SpriteSheet, ...]:
    candidate = Path(name_or_path).expanduser()
    if not candidate.is_dir():
        candidate = _packs_dir() / name_or_path
    manifest_path = candidate / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("pack de sprites introuvable : {}".format(name_or_path))
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    sheets = []
    for item in data.get("sheets", ()):
        rows = int(item.get("rows", 1))
        role_rows = {str(role): int(row) for role, row in item["roles"].items()}
        if (
            rows < 1
            or not set(role_rows).issubset(ROLES)
            or any(row < 0 or row >= rows for row in role_rows.values())
        ):
            raise ValueError("manifest de sprites invalide : {}".format(manifest_path))
        path = (candidate / item["file"]).resolve()
        if not path.is_file() or candidate.resolve() not in path.parents:
            raise ValueError("image de pack invalide : {}".format(item["file"]))
        sheets.append(SpriteSheet(path, rows, role_rows))
    if not sheets:
        raise ValueError("pack sans feuille de sprites : {}".format(manifest_path))
    return tuple(sheets)
