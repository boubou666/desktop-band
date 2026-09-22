"""Import a small, balanced collection of isolated Slakh stems.

The importer deliberately writes short PCM WAV excerpts instead of copying a
whole Slakh release.  Each source song receives a stable validation group so
that excerpts from the same composition cannot leak across train/validation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
import wave

import numpy as np


BABYSLAKH_URL = (
    "https://zenodo.org/records/4603870/files/babyslakh_16k.tar.gz?download=1"
)
BABYSLAKH_MD5 = "311096dc2bde7d61c97e930edbfc7f78"

_ROLE_CLASSES = {
    "bass": "bassist",
    "guitar": "guitarist",
    "piano": "keyboard",
    "organ": "keyboard",
    "synth lead": "keyboard",
    "synth pad": "keyboard",
    "chromatic percussion": "percussion",
    "percussive": "percussion",
}


@dataclass(frozen=True, slots=True)
class Window:
    role: str
    track: str
    stem: str
    path: Path
    start_frame: int
    frame_count: int
    sample_rate: int
    rms: float


def role_for_stem(metadata: dict[str, object]) -> str | None:
    """Map the Slakh taxonomy to the six Gopnik Band roles."""

    if bool(metadata.get("is_drum")):
        return "drummer"
    instrument_class = str(metadata.get("inst_class", "")).strip().lower()
    return _ROLE_CLASSES.get(instrument_class)


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - exercised by CLI users
        raise RuntimeError(
            'PyYAML est requis : python -m pip install -e ".[training]"'
        ) from error
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"métadonnées Slakh invalides : {path}")
    return document


def _pcm16_mono(raw: bytes, channels: int) -> np.ndarray:
    samples = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.int32).mean(axis=1)
        samples = np.clip(samples, -32768, 32767).astype("<i2")
    return samples


def _scan_windows(
    path: Path,
    *,
    role: str,
    track: str,
    stem: str,
    seconds: float,
    minimum_rms: float,
) -> list[Window]:
    windows = []
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        rate = source.getframerate()
        width = source.getsampwidth()
        if width != 2 or channels not in (1, 2):
            raise ValueError(f"WAV PCM 16 bits mono/stéréo attendu : {path}")
        window_frames = max(1, round(seconds * rate))
        start = 0
        while start < source.getnframes():
            raw = source.readframes(window_frames)
            samples = _pcm16_mono(raw, channels)
            if samples.size < window_frames // 2:
                break
            values = samples.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(values * values)))
            if rms >= minimum_rms:
                windows.append(Window(
                    role, track, stem, path, start, int(samples.size), rate, rms
                ))
            start += window_frames
    return windows


def _select_balanced(
    candidates: dict[str, dict[str, list[Window]]],
    quota_seconds: float,
) -> dict[tuple[str, str], list[Window]]:
    selected: dict[tuple[str, str], list[Window]] = defaultdict(list)
    for role, by_track in candidates.items():
        queues = {
            track: sorted(windows, key=lambda item: item.rms, reverse=True)
            for track, windows in by_track.items()
        }
        total = 0.0
        # Round-robin selection prevents one long song from filling a role.
        while total < quota_seconds:
            progressed = False
            for track in sorted(queues):
                if not queues[track] or total >= quota_seconds:
                    continue
                window = queues[track].pop(0)
                selected[(track, role)].append(window)
                total += window.frame_count / window.sample_rate
                progressed = True
            if not progressed:
                break
    return selected


def _write_windows(path: Path, windows: list[Window]) -> float:
    if not windows:
        return 0.0
    rate = windows[0].sample_rate
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(rate)
        for item in windows:
            if item.sample_rate != rate:
                continue
            with wave.open(str(item.path), "rb") as source:
                source.setpos(item.start_frame)
                samples = _pcm16_mono(
                    source.readframes(item.frame_count), source.getnchannels()
                )
            destination.writeframes(samples.astype("<i2", copy=False).tobytes())
            frames_written += int(samples.size)
    return frames_written / rate


def find_slakh_root(path: Path) -> Path:
    """Accept the archive extraction directory or the dataset directory itself."""

    path = path.resolve()
    if any(path.glob("**/Track*/metadata.yaml")):
        return path
    raise FileNotFoundError(f"aucun TrackXXXXX/metadata.yaml trouvé dans {path}")


def import_slakh(
    dataset_dir: Path,
    training_dir: Path,
    *,
    minutes_per_role: float = 30.0,
    clip_seconds: float = 10.0,
    minimum_rms: float = 0.003,
    max_tracks: int | None = None,
) -> dict[str, object]:
    """Create a balanced local training manifest from isolated Slakh stems."""

    dataset_dir = find_slakh_root(dataset_dir)
    training_dir = training_dir.resolve()
    track_dirs = sorted({path.parent for path in dataset_dir.glob("**/metadata.yaml")})
    if max_tracks is not None:
        track_dirs = track_dirs[:max(1, int(max_tracks))]
    candidates: dict[str, dict[str, list[Window]]] = defaultdict(
        lambda: defaultdict(list)
    )
    failures = []
    for track_dir in track_dirs:
        try:
            metadata = _load_yaml(track_dir / "metadata.yaml")
            stems = metadata.get("stems", {})
            if not isinstance(stems, dict):
                continue
            audio_dir = track_dir / str(metadata.get("audio_dir", "stems"))
            for stem_name, stem_metadata in stems.items():
                if not isinstance(stem_metadata, dict):
                    continue
                role = role_for_stem(stem_metadata)
                if role is None:
                    continue
                options = [
                    audio_dir / f"{stem_name}.wav",
                    audio_dir / f"{stem_name}.WAV",
                ]
                audio_path = next((item for item in options if item.is_file()), None)
                if audio_path is None:
                    continue
                candidates[role][track_dir.name].extend(_scan_windows(
                    audio_path,
                    role=role,
                    track=track_dir.name,
                    stem=str(stem_name),
                    seconds=clip_seconds,
                    minimum_rms=minimum_rms,
                ))
        except (OSError, ValueError, wave.Error, RuntimeError) as error:
            failures.append(f"{track_dir.name}: {error}")

    selected = _select_balanced(
        candidates, max(1.0 / 60.0, minutes_per_role) * 60.0
    )
    output_dir = training_dir / "slakh" / "audio"
    clips = []
    totals = defaultdict(float)
    for (track, role), windows in sorted(selected.items()):
        output = output_dir / f"slakh-{track}-{role}.wav"
        duration = _write_windows(output, windows)
        if duration <= 0:
            continue
        totals[role] += duration
        clips.append({
            "audio": output.relative_to(training_dir).as_posix(),
            "exclusive_label": role,
            "validation_group": f"slakh:{track}",
            "source_title": track,
            "source_dataset": "BabySlakh/Slakh2100",
            "source_stems": sorted({item.stem for item in windows}),
            "duration_seconds": round(duration, 3),
        })
    manifest = {
        "schema_version": 1,
        "dataset": "BabySlakh/Slakh2100",
        "license": "CC BY 4.0",
        "source": "https://zenodo.org/records/4603870",
        "settings": {
            "minutes_per_role": minutes_per_role,
            "clip_seconds": clip_seconds,
            "minimum_rms": minimum_rms,
            "tracks_scanned": len(track_dirs),
        },
        "totals_seconds": {
            role: round(value, 3) for role, value in sorted(totals.items())
        },
        "clips": clips,
        "failures": failures,
    }
    manifest_path = training_dir / "slakh_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 - dataset integrity, not security
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_babyslakh(destination: Path) -> Path:
    """Download and verify the official 883 MB BabySlakh archive."""

    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "babyslakh_16k.tar.gz"
    if archive.is_file() and _md5(archive) == BABYSLAKH_MD5:
        return archive
    temporary = archive.with_suffix(archive.suffix + ".part")
    request = urllib.request.Request(
        BABYSLAKH_URL, headers={"User-Agent": "gopnik-band/0.1"}
    )
    with urllib.request.urlopen(request) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target, length=1024 * 1024)
    if _md5(temporary) != BABYSLAKH_MD5:
        raise ValueError("checksum BabySlakh invalide")
    temporary.replace(archive)
    return archive


def extract_babyslakh(archive: Path, destination: Path) -> Path:
    """Extract an archive after rejecting links and traversal paths."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if member.issym() or member.islnk():
                raise ValueError("archive BabySlakh avec lien refusée")
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError("chemin dangereux dans l'archive BabySlakh")
            source.extract(member, root)
    return find_slakh_root(root)
