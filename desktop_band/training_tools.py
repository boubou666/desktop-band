"""Dataset synthesis, model registry and A/B evaluation for Gopnik Lab."""

from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import time
import wave

import numpy as np

from .classifier import CLASSIFIER_LABELS, FEATURE_NAMES


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        rate = source.getframerate()
        width = source.getsampwidth()
        raw = source.readframes(source.getnframes())
    if width != 2 or channels not in (1, 2):
        raise ValueError(f"WAV PCM 16 bits attendu : {path.name}")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    samples = samples.reshape(-1, channels)
    if channels == 1:
        samples = np.repeat(samples, 2, axis=1)
    return samples, rate


def _write_wav(path: Path, samples: np.ndarray, rate: int = 44_100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(2)
        destination.setsampwidth(2)
        destination.setframerate(rate)
        destination.writeframes(pcm.tobytes())


def load_training_manifest(
    training_dir: Path, *, include_synthetic: bool = True
) -> dict[str, object]:
    """Merge human annotations with generated and imported local datasets."""

    manifest: dict[str, object] = {"clips": []}
    names = ["annotations.json", "slakh_manifest.json"]
    if include_synthetic:
        names.append("synthetic_manifest.json")
    for name in names:
        source = training_dir / name
        if not source.is_file():
            continue
        document = json.loads(source.read_text(encoding="utf-8"))
        manifest["clips"] = [
            *manifest.get("clips", []), *document.get("clips", [])
        ]
    return manifest


def build_training_manifest(training_dir: Path) -> Path:
    """Persist the combined human, imported and generated manifest."""

    manifest = load_training_manifest(training_dir)
    target = training_dir / "training_manifest.json"
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def generate_synthetic_mixes(
    training_dir: Path,
    *,
    count: int = 12,
    seconds: float = 12.0,
    seed: int = 1917,
) -> dict[str, object]:
    """Mix exclusive stems with deterministic gain/EQ/speed/noise variants."""

    manifest = load_training_manifest(training_dir, include_synthetic=False)
    sources = []
    for clip in manifest.get("clips", []):
        role = clip.get("exclusive_label")
        path = training_dir / str(clip.get("audio", ""))
        if role in CLASSIFIER_LABELS and path.is_file():
            samples, rate = _read_wav(path)
            if rate != 44_100:
                continue
            sources.append((role, samples))
    roles = sorted({role for role, _samples in sources})
    if len(roles) < 2:
        raise ValueError(
            "il faut au moins deux rôles marqués mono-instrument pour fabriquer des mixes"
        )

    randomizer = random.Random(seed)
    output_dir = training_dir / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    seconds = max(1.0, min(60.0, float(seconds)))
    target_frames = round(seconds * 44_100)
    clips = []
    for mix_index in range(max(1, min(100, int(count)))):
        role_count = randomizer.randint(2, min(4, len(roles)))
        chosen_roles = randomizer.sample(roles, role_count)
        mixed = np.zeros((target_frames, 2), dtype=np.float32)
        for role in chosen_roles:
            candidates = [samples for candidate, samples in sources if candidate == role]
            source = randomizer.choice(candidates)
            speed = randomizer.uniform(0.96, 1.04)
            needed = max(2, round(target_frames * speed))
            if source.shape[0] >= needed:
                start = randomizer.randint(0, source.shape[0] - needed)
                excerpt = source[start:start + needed]
            else:
                repeats = int(np.ceil(needed / max(1, source.shape[0])))
                excerpt = np.tile(source, (repeats, 1))[:needed]
            positions = np.linspace(0, excerpt.shape[0] - 1, target_frames)
            left = np.interp(positions, np.arange(excerpt.shape[0]), excerpt[:, 0])
            right = np.interp(positions, np.arange(excerpt.shape[0]), excerpt[:, 1])
            transformed = np.column_stack((left, right)).astype(np.float32)
            # Gentle two-band EQ and compression keep the labels intact while
            # varying timbre, loudness and dynamics.
            kernel = np.ones(31, dtype=np.float32) / 31.0
            low = np.column_stack((
                np.convolve(transformed[:, 0], kernel, mode="same"),
                np.convolve(transformed[:, 1], kernel, mode="same"),
            )).astype(np.float32)
            high = transformed - low
            transformed = (
                low * randomizer.uniform(0.82, 1.18)
                + high * randomizer.uniform(0.82, 1.18)
            )
            transformed = np.tanh(
                transformed * randomizer.uniform(0.65, 1.15)
            )
            mixed += transformed * randomizer.uniform(0.42, 0.88)
        noise = np.random.default_rng(seed + mix_index).normal(
            0.0, randomizer.uniform(0.0, 0.004), mixed.shape
        ).astype(np.float32)
        mixed += noise
        peak = max(0.05, float(np.max(np.abs(mixed))))
        mixed *= min(1.0, 0.92 / peak)
        filename = f"synthetic-{mix_index + 1:03d}.wav"
        _write_wav(output_dir / filename, mixed)
        clips.append({
            "audio": f"synthetic/{filename}",
            "global_labels": chosen_roles,
            "segments": [{
                "start": 0.0,
                "end": round(seconds, 3),
                "labels": chosen_roles,
            }],
            "synthetic": True,
        })
    output = training_dir / "synthetic_manifest.json"
    output.write_text(
        json.dumps({"clips": clips}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    build_training_manifest(training_dir)
    return {
        "ok": True,
        "mixes": len(clips),
        "seconds_each": seconds,
        "roles": roles,
        "manifest": str(output),
    }


def _safe_model_candidates(workspace: Path) -> list[tuple[str, Path]]:
    active = workspace / "desktop_band" / "models" / "instrument_classifier.onnx"
    candidates = [("active", active)] if active.is_file() else []
    training = workspace / "training"
    candidate = training / "candidate_classifier.onnx"
    if candidate.is_file():
        candidates.append(("candidate", candidate))
    for path in sorted((training / "models").glob("*.onnx"), reverse=True):
        candidates.append((path.stem, path))
    return candidates


def list_models(workspace: Path) -> list[dict[str, object]]:
    models = []
    for name, path in _safe_model_candidates(workspace.resolve()):
        metadata_path = path.with_suffix(".json")
        metadata = None
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        models.append({
            "name": name,
            "active": name == "active",
            "size": path.stat().st_size,
            "modified": int(path.stat().st_mtime),
            "metadata": metadata,
        })
    return models


def activate_model(workspace: Path, name: str) -> dict[str, object]:
    workspace = workspace.resolve()
    selected = next(
        ((candidate_name, path) for candidate_name, path in _safe_model_candidates(workspace)
         if candidate_name == name),
        None,
    )
    if selected is None or name == "active":
        raise ValueError("modèle inconnu ou déjà actif")
    _candidate_name, source = selected
    active = workspace / "desktop_band" / "models" / "instrument_classifier.onnx"
    archive = workspace / "training" / "models"
    archive.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    backup = archive / f"instrument_classifier-{stamp}.onnx"
    shutil.copy2(active, backup)
    active_meta = active.with_suffix(".json")
    if active_meta.is_file():
        shutil.copy2(active_meta, backup.with_suffix(".json"))
    shutil.copy2(source, active)
    source_meta = source.with_suffix(".json")
    if source_meta.is_file():
        shutil.copy2(source_meta, active_meta)
    elif active_meta.is_file():
        active_meta.unlink()
    return {"ok": True, "activated": name, "backup": backup.stem}


def compare_models(workspace: Path) -> dict[str, object]:
    """Evaluate every local model on the current labeled feature set."""

    from tools.train_classifier import load_dataset, macro_f1, per_role_f1, predict_onnx

    features = workspace / "training" / "features.jsonl"
    if not features.is_file():
        raise ValueError("entraîne une première fois pour créer les caractéristiques")
    x, y, _sources = load_dataset(features)
    comparisons = []
    for name, path in _safe_model_candidates(workspace):
        predictions = predict_onnx(path, x)
        comparisons.append({
            "name": name,
            "active": name == "active",
            "macro_f1": round(macro_f1(predictions, y), 6),
            "per_role_f1": {
                role: round(score, 6)
                for role, score in per_role_f1(predictions, y).items()
            },
        })
    comparisons.sort(key=lambda item: item["macro_f1"], reverse=True)
    return {"models": comparisons, "samples": int(x.shape[0])}
