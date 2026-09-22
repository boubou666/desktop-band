"""Extract labeled Stemgen features from annotated PCM WAV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import wave

import numpy as np

from desktop_band.classifier import CLASSIFIER_LABELS, FEATURE_NAMES
from desktop_band.stemgen import STEMGEN_SAMPLE_RATE, StemgenAnalyzer


def _cache_key(audio_path: Path, analyzer: StemgenAnalyzer) -> str:
    stat = audio_path.stat()
    model = getattr(analyzer, "model_path", None)
    model_stat = Path(model).stat() if model and Path(model).is_file() else None
    identity = json.dumps({
        "audio": str(audio_path.resolve()),
        "size": stat.st_size,
        "mtime": stat.st_mtime_ns,
        "model_size": model_stat.st_size if model_stat else 0,
        "model_mtime": model_stat.st_mtime_ns if model_stat else 0,
        "features": FEATURE_NAMES,
        "schema": 2,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def _write_progress(path: Path | None, **values) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _acquire_training_lock(path: Path):
    """Hold a portable advisory lock for the lifetime of feature extraction."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as error:
        handle.close()
        raise RuntimeError("une extraction de caractéristiques est déjà active") from error
    return handle


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        rate = source.getframerate()
        width = source.getsampwidth()
        raw = source.readframes(source.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("seuls les WAV PCM 16/32 bits sont acceptés")
    return data.reshape(-1, channels), rate


def resample(samples: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == STEMGEN_SAMPLE_RATE:
        return samples
    duration = samples.shape[0] / source_rate
    old = np.linspace(0.0, duration, samples.shape[0], endpoint=False)
    new = np.linspace(
        0.0, duration, round(duration * STEMGEN_SAMPLE_RATE), endpoint=False
    )
    return np.stack(
        [np.interp(new, old, samples[:, channel]) for channel in range(samples.shape[1])],
        axis=1,
    ).astype(np.float32)


def labels_at(segments, timestamp: float) -> list[str]:
    labels = set()
    for segment in segments:
        if float(segment["start"]) <= timestamp < float(segment["end"]):
            labels.update(segment["labels"])
    return sorted(labels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("training/features.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("training/cache"))
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()
    lock_handle = _acquire_training_lock(args.output.with_suffix(".lock"))
    manifest = json.loads(args.annotations.read_text(encoding="utf-8"))
    rows = []
    clips = manifest["clips"]
    started = time.monotonic()
    cache_hits = 0
    for clip_index, clip in enumerate(clips):
        audio_path = (args.annotations.parent / clip["audio"]).resolve()
        elapsed = time.monotonic() - started
        _write_progress(
            args.progress_file,
            phase="extract",
            current_track=audio_path.name,
            tracks_done=clip_index,
            tracks_total=len(clips),
            progress=clip_index / max(1, len(clips)),
            elapsed_seconds=round(elapsed, 1),
            eta_seconds=None,
            cache_hits=cache_hits,
        )
        analyzer = StemgenAnalyzer(args.model_path)
        block_size = 1024
        key = _cache_key(audio_path, analyzer)
        cache_path = args.cache_dir / (key + ".npz")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.is_file():
            cached = np.load(cache_path)
            times = cached["times"]
            matrix = cached["features"]
            active = cached["active"].astype(bool)
            cache_hits += 1
        else:
            samples, rate = read_wav(audio_path)
            samples = resample(samples, rate)
            times_list = []
            feature_rows = []
            active_list = []
            for start in range(0, samples.shape[0] - block_size + 1, block_size):
                timestamp = (start + block_size / 2) / STEMGEN_SAMPLE_RATE
                features = analyzer.process(
                    samples[start:start + block_size],
                    STEMGEN_SAMPLE_RATE,
                    now=timestamp,
                )
                times_list.append(timestamp)
                feature_rows.append([
                    float(getattr(features, name, 0.0)) for name in FEATURE_NAMES
                ])
                active_list.append(bool(features.active))
            times = np.asarray(times_list, dtype=np.float32)
            matrix = np.asarray(feature_rows, dtype=np.float32)
            active = np.asarray(active_list, dtype=np.uint8)
            np.savez_compressed(
                cache_path, times=times, features=matrix, active=active
            )
        segments = clip.get("segments", [])
        global_labels = clip.get("global_labels", [])
        exclusive_label = clip.get("exclusive_label")
        for block_index, (timestamp, values, is_active) in enumerate(
            zip(times, matrix, active)
        ):
            timestamp = float(timestamp)
            labels = labels_at(segments, timestamp)
            label_source = "timeline"
            if not segments:
                labels = sorted(set(global_labels))
                label_source = "global"
            if exclusive_label:
                labels = [exclusive_label] if is_active else []
                label_source = "exclusive"
            # Unlabeled timeline windows are useful negative examples, but a
            # whole song of silence must not drown the positive corrections.
            if (segments or exclusive_label) and not labels and block_index % 8:
                continue
            rows.append({
                "source": str(clip.get("validation_group") or audio_path.name),
                "time": round(timestamp, 4),
                "features": {
                    name: float(value) for name, value in zip(FEATURE_NAMES, values)
                },
                "labels": labels,
                "label_source": label_source,
            })
        done = clip_index + 1
        elapsed = time.monotonic() - started
        eta = elapsed / done * max(0, len(clips) - done)
        _write_progress(
            args.progress_file,
            phase="extract",
            current_track=audio_path.name,
            tracks_done=done,
            tracks_total=len(clips),
            progress=done / max(1, len(clips)),
            elapsed_seconds=round(elapsed, 1),
            eta_seconds=round(eta, 1),
            cache_hits=cache_hits,
        )
    review_dir = args.annotations.parent / "review_queue"
    for review_path in sorted(review_dir.glob("*.json")):
        try:
            document = json.loads(review_path.read_text(encoding="utf-8"))
            labels = [
                label for label in document.get("correct_labels", [])
                if label in CLASSIFIER_LABELS
            ]
            if not labels:
                continue
            for index, frame in enumerate(document.get("frames", [])):
                if index % 3:
                    continue
                values = frame.get("features", {})
                rows.append({
                    "source": "live-review:" + review_path.name,
                    "time": round(float(frame.get("timestamp", 0.0)), 4),
                    "features": {
                        name: float(values.get(name, 0.0)) for name in FEATURE_NAMES
                    },
                    "labels": labels,
                    "label_source": "live-review",
                })
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print("samples={} cache_hits={} output={}".format(
        len(rows), cache_hits, args.output
    ))
    lock_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
