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


_GOPNIK_SHEETS = (
    ("gopnik_bassist.png", 1, {"bassist": 0}),
    ("gopnik_bassist_extra.png", 1, {"bassist": 0}),
    ("gopnik_singer_base.png", 1, {"singer": 0}),
    ("gopnik_singer_mouths.png", 1, {"singer": 0}),
    ("gopnik_drummer.png", 1, {"drummer": 0}),
    ("gopnik_drummer_extra.png", 1, {"drummer": 0}),
    ("gopnik_keyboard.png", 1, {"keyboard": 0}),
    ("gopnik_keyboard_extra.png", 1, {"keyboard": 0}),
    ("gopnik_guitarist.png", 1, {"guitarist": 0}),
    ("gopnik_guitarist_extra.png", 1, {"guitarist": 0}),
    ("gopnik_percussion.png", 1, {"percussion": 0}),
    ("gopnik_percussion_extra.png", 1, {"percussion": 0}),
    ("gopnik_vibe_dance_left.png", 1, {"vibing": 0}),
    ("gopnik_vibe_dance_right.png", 1, {"vibing": 0}),
)


def _asset_dir() -> Path:
    return Path(__file__).with_name("assets")


def available_sprite_packs() -> tuple[str, ...]:
    names = ["gopnik"]
    packs = _asset_dir() / "packs"
    if packs.is_dir():
        names.extend(
            child.name for child in packs.iterdir()
            if child.is_dir() and (child / "manifest.json").is_file()
        )
    return tuple(dict.fromkeys(names))


def load_sprite_pack(name_or_path: str = "gopnik") -> tuple[SpriteSheet, ...]:
    if name_or_path == "gopnik":
        root = _asset_dir()
        return tuple(
            SpriteSheet(root / filename, rows, role_rows)
            for filename, rows, role_rows in _GOPNIK_SHEETS
        )

    candidate = Path(name_or_path).expanduser()
    if not candidate.is_dir():
        candidate = _asset_dir() / "packs" / name_or_path
    manifest_path = candidate / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("pack de sprites introuvable : {}".format(name_or_path))
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    sheets = []
    for item in data.get("sheets", ()):
        rows = int(item.get("rows", 1))
        role_rows = {str(role): int(row) for role, row in item["roles"].items()}
        if rows < 1 or not set(role_rows).issubset(ROLES):
            raise ValueError("manifest de sprites invalide : {}".format(manifest_path))
        path = (candidate / item["file"]).resolve()
        if not path.is_file() or candidate.resolve() not in path.parents:
            raise ValueError("image de pack invalide : {}".format(item["file"]))
        sheets.append(SpriteSheet(path, rows, role_rows))
    if not sheets:
        raise ValueError("pack sans feuille de sprites : {}".format(manifest_path))
    return tuple(sheets)


# Compatibility alias used by existing tests and third-party pack tools.
SPRITE_SHEETS = _GOPNIK_SHEETS
