"""Optional tiny ONNX classifier used after StemgenRT separation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CLASSIFIER_LABELS = (
    "singer",
    "drummer",
    "bassist",
    "keyboard",
    "guitarist",
    "percussion",
)
FEATURE_NAMES = (
    "level",
    "low",
    "mid",
    "high",
    "vocal",
    "percussive",
    "harmonic",
    "stem_bass",
    "stem_vocals",
    "stem_drums",
    "stem_other",
    "percussive_low",
    "percussive_mid",
    "percussive_high",
    "stem_other_low",
    "stem_other_mid",
    "stem_other_high",
    "stem_other_onset",
    "stem_other_harmonic",
    "stem_other_centroid",
    "stem_other_flatness",
    "stem_other_pitch_stability",
    "stem_other_note_density",
)


def feature_vector(features) -> np.ndarray:
    return np.asarray(
        [[float(getattr(features, name, 0.0)) for name in FEATURE_NAMES]],
        dtype=np.float32,
    )


class InstrumentClassifier:
    def __init__(self, model_path: str | Path | None = None, session_factory=None) -> None:
        path = (
            Path(model_path)
            if model_path is not None
            else Path(__file__).with_name("models") / "instrument_classifier.onnx"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        if session_factory is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        else:
            self._session = session_factory(str(path))
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self.thresholds = {label: 0.5 for label in CLASSIFIER_LABELS}
        metadata_path = path.with_suffix(".json")
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                configured = metadata.get("thresholds", {})
                for label in CLASSIFIER_LABELS:
                    value = float(configured.get(label, 0.5))
                    self.thresholds[label] = max(0.05, min(0.95, value))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

    def predict(self, features) -> dict[str, float]:
        output = self._session.run(
            [self._output_name], {self._input_name: feature_vector(features)}
        )[0]
        values = np.asarray(output, dtype=np.float32).reshape(-1)
        if values.size != len(CLASSIFIER_LABELS):
            raise RuntimeError("sortie du classifieur ONNX incompatible")
        scores = {}
        for label, raw in zip(CLASSIFIER_LABELS, values):
            value = max(0.0, min(1.0, float(raw)))
            threshold = self.thresholds[label]
            # Normalize each learned operating point back to 0.5 so the
            # existing router can remain model-agnostic.
            if value < threshold:
                value = 0.5 * value / threshold
            else:
                value = 0.5 + 0.5 * (value - threshold) / (1.0 - threshold)
            scores[label] = max(0.0, min(1.0, value))
        return scores


def load_default_classifier() -> InstrumentClassifier | None:
    try:
        return InstrumentClassifier()
    except Exception:
        return None
