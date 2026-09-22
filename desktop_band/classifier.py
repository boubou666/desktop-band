"""Optional tiny ONNX classifier used after StemgenRT separation."""

from __future__ import annotations

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

    def predict(self, features) -> dict[str, float]:
        output = self._session.run(
            [self._output_name], {self._input_name: feature_vector(features)}
        )[0]
        values = np.asarray(output, dtype=np.float32).reshape(-1)
        if values.size != len(CLASSIFIER_LABELS):
            raise RuntimeError("sortie du classifieur ONNX incompatible")
        return {
            label: max(0.0, min(1.0, float(value)))
            for label, value in zip(CLASSIFIER_LABELS, values)
        }


def load_default_classifier() -> InstrumentClassifier | None:
    try:
        return InstrumentClassifier()
    except Exception:
        return None
