"""Extract labeled Stemgen features from annotated PCM WAV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import wave

import numpy as np

from desktop_band.classifier import FEATURE_NAMES
from desktop_band.stemgen import STEMGEN_SAMPLE_RATE, StemgenAnalyzer


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
    args = parser.parse_args()
    manifest = json.loads(args.annotations.read_text(encoding="utf-8"))
    rows = []
    for clip in manifest["clips"]:
        audio_path = (args.annotations.parent / clip["audio"]).resolve()
        samples, rate = read_wav(audio_path)
        samples = resample(samples, rate)
        analyzer = StemgenAnalyzer(args.model_path)
        block_size = 1024
        segments = clip.get("segments", [])
        global_labels = clip.get("global_labels", [])
        exclusive_label = clip.get("exclusive_label")
        for block_index, start in enumerate(
            range(0, samples.shape[0] - block_size + 1, block_size)
        ):
            timestamp = (start + block_size / 2) / STEMGEN_SAMPLE_RATE
            labels = labels_at(segments, timestamp)
            label_source = "timeline"
            if not segments:
                labels = sorted(set(global_labels))
                label_source = "global"
            features = analyzer.process(
                samples[start:start + block_size],
                STEMGEN_SAMPLE_RATE,
                now=timestamp,
            )
            if exclusive_label:
                labels = [exclusive_label] if features.active else []
                label_source = "exclusive"
            # Unlabeled timeline windows are useful negative examples, but a
            # whole song of silence must not drown the positive corrections.
            if (segments or exclusive_label) and not labels and block_index % 8:
                continue
            rows.append({
                "source": str(audio_path.name),
                "time": round(timestamp, 4),
                "features": {
                    name: float(getattr(features, name, 0.0))
                    for name in FEATURE_NAMES
                },
                "labels": labels,
                "label_source": label_source,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print("samples={} output={}".format(len(rows), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
