"""Download/import a small balanced BabySlakh training subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from desktop_band.slakh import (
    download_babyslakh,
    extract_babyslakh,
    import_slakh,
)
from desktop_band.training_tools import build_training_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prépare des stems Slakh isolés pour Gopnik Band."
    )
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument("--download-baby", action="store_true")
    parser.add_argument("--download-dir", type=Path, default=Path("training/datasets"))
    parser.add_argument("--training-dir", type=Path, default=Path("training"))
    parser.add_argument("--minutes-per-role", type=float, default=30.0)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--minimum-rms", type=float, default=0.003)
    parser.add_argument("--max-tracks", type=int)
    args = parser.parse_args()
    dataset = args.dataset
    if args.download_baby:
        print("Téléchargement de BabySlakh (882,8 Mo)…", flush=True)
        archive = download_babyslakh(args.download_dir)
        print("Extraction de BabySlakh…", flush=True)
        dataset = extract_babyslakh(archive, args.download_dir / "babyslakh")
    if dataset is None:
        parser.error("indique un dossier Slakh ou utilise --download-baby")
    print("Sélection des passages actifs et équilibrage…", flush=True)
    report = import_slakh(
        dataset,
        args.training_dir,
        minutes_per_role=args.minutes_per_role,
        clip_seconds=args.clip_seconds,
        minimum_rms=args.minimum_rms,
        max_tracks=args.max_tracks,
    )
    build_training_manifest(args.training_dir)
    print(json.dumps({
        "clips": len(report["clips"]),
        "totals_seconds": report["totals_seconds"],
        "failures": report["failures"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
