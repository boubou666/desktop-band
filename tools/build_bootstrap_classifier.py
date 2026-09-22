"""Build the shipped classifier from deterministic acoustic prototypes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from desktop_band.classifier import CLASSIFIER_LABELS, FEATURE_NAMES
from train_classifier import export_onnx, train


PROTOTYPES = {
    "singer": {
        "level": 0.82, "low": 0.24, "mid": 0.78, "high": 0.38,
        "vocal": 0.88, "harmonic": 0.82, "percussive": 0.12,
        "stem_vocals": 0.94, "stem_bass": 0.06,
        "stem_drums": 0.06, "stem_other": 0.12,
    },
    "drummer": {
        "level": 0.88, "low": 0.82, "mid": 0.34, "high": 0.22,
        "percussive": 0.92, "harmonic": 0.10,
        "stem_drums": 0.90, "stem_other": 0.08,
        "percussive_low": 0.86, "percussive_mid": 0.38,
        "percussive_high": 0.22,
    },
    "percussion": {
        "level": 0.82, "low": 0.16, "mid": 0.74, "high": 0.90,
        "percussive": 0.94, "harmonic": 0.08,
        "stem_drums": 0.82, "stem_other": 0.10,
        "percussive_low": 0.18, "percussive_mid": 0.78,
        "percussive_high": 0.90,
    },
    "bassist": {
        "level": 0.80, "low": 0.94, "mid": 0.24, "high": 0.06,
        "vocal": 0.06, "harmonic": 0.88, "percussive": 0.10,
        "stem_bass": 0.94, "stem_vocals": 0.05,
        "stem_drums": 0.08, "stem_other": 0.12,
    },
    "guitarist": {
        "level": 0.78, "low": 0.20, "mid": 0.90, "high": 0.34,
        "vocal": 0.08, "harmonic": 0.80, "percussive": 0.18,
        "stem_drums": 0.08, "stem_other": 0.90,
        "stem_other_low": 0.18, "stem_other_mid": 0.92,
        "stem_other_high": 0.28, "stem_other_onset": 0.58,
        "stem_other_harmonic": 0.58, "stem_other_centroid": 0.46,
        "stem_other_flatness": 0.34, "stem_other_pitch_stability": 0.34,
        "stem_other_note_density": 0.78,
    },
    "keyboard": {
        "level": 0.76, "low": 0.52, "mid": 0.52, "high": 0.72,
        "vocal": 0.06, "harmonic": 0.92, "percussive": 0.08,
        "stem_drums": 0.06, "stem_other": 0.92,
        "stem_other_low": 0.54, "stem_other_mid": 0.52,
        "stem_other_high": 0.74, "stem_other_onset": 0.18,
        "stem_other_harmonic": 0.91, "stem_other_centroid": 0.52,
        "stem_other_flatness": 0.08, "stem_other_pitch_stability": 0.90,
        "stem_other_note_density": 0.24,
    },
}


def build_samples(seed: int = 1917) -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(seed)
    samples = []
    targets = []
    for _ in range(420):
        count = 2 if random.random() < 0.22 else 1
        labels = random.choice(CLASSIFIER_LABELS, size=count, replace=False)
        values = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        for label in labels:
            prototype = PROTOTYPES[str(label)]
            values = np.maximum(
                values,
                np.asarray([prototype.get(name, 0.03) for name in FEATURE_NAMES]),
            )
        values = np.clip(
            values * random.uniform(0.72, 1.08)
            + random.normal(0.0, 0.055, values.shape),
            0.0,
            1.0,
        )
        samples.append(values)
        targets.append([float(label in labels) for label in CLASSIFIER_LABELS])
    return np.asarray(samples, np.float32), np.asarray(targets, np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("desktop_band/models/instrument_classifier.onnx"),
    )
    args = parser.parse_args()
    x, y = build_samples()
    weights, bias = train(x, y, epochs=2200, rate=0.16)
    export_onnx(weights, bias, args.output)
    print("bootstrap samples={} output={}".format(x.shape[0], args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
