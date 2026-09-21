"""Streaming StemgenRT inference adapted to animation controls.

StemgenRT's ONNX graph consumes stereo hops of 128 samples at 44.1 kHz and
returns drums, bass, vocals and other.  We keep its recurrent tensors between
calls and reduce the separated audio to stable 0..1 activity values; the
audio itself never leaves this module.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .analysis import AudioAnalyzer, AudioFeatures, _clamp


STEMGEN_SAMPLE_RATE = 44_100
STEMGEN_HOP_SIZE = 128
MODEL_FILENAME = "stemgen_rt.onnx"
MODEL_URL = (
    "https://media.githubusercontent.com/media/sweetspotsoundsystem/"
    "stemgen-rt/main/model/model.onnx"
)


def find_model_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate a locally downloaded StemgenRT model."""

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    configured = os.environ.get("DESKTOP_BAND_STEMGEN_MODEL")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        (
            Path.cwd() / "models" / MODEL_FILENAME,
            Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME,
            Path(__file__).resolve().parent / "models" / MODEL_FILENAME,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "modèle StemgenRT introuvable ; place-le dans "
        f"models/{MODEL_FILENAME} ou utilise --model-path"
    )


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return float(value >= edge1)
    position = _clamp((value - edge0) / (edge1 - edge0))
    return position * position * (3.0 - 2.0 * position)


def _stem_activity_target(
    source: str, rms: float, loudest: float, mix_level: float
) -> tuple[float, float]:
    """Return a leakage-gated activity score and its relative dB level."""

    relative_db = 20.0 * math.log10(max(float(rms), 1e-9) / max(loudest, 1e-9))
    if source == "bass":
        # Bass is commonly much quieter in the separated output than vocals,
        # drums or Other. A dedicated gate retains legitimate bass around
        # -20 dB without relaxing the leak protection for every other role.
        credibility = _smoothstep(-24.0, -9.0, relative_db)
        audible = _smoothstep(0.00035, 0.004, float(rms))
    else:
        credibility = _smoothstep(-16.0, -6.0, relative_db)
        audible = _smoothstep(0.0007, 0.008, float(rms))
    return mix_level * credibility * audible, relative_db


def _rescue_percussive_other(
    levels: dict[str, float], base: AudioFeatures
) -> dict[str, float]:
    """Reroute low percussive material that the model calls ``other``.

    StemgenRT's four-source vocabulary sometimes treats large resonant drums
    (notably taiko) as ``other`` and emits short vocal leaks on their attacks.
    Only intervene when other is overwhelmingly alone; this keeps the model's
    normal decisions intact on complete musical mixes.
    """

    adjusted = dict(levels)
    total = sum(adjusted.values())
    if total <= 1e-9:
        return adjusted
    other_dominance = adjusted["other"] / total
    isolated_other = _smoothstep(0.62, 0.86, other_dominance)
    if isolated_other <= 0.0:
        return adjusted

    transient = max(base.percussive, base.percussive_memory)
    percussive_shape = _smoothstep(0.18, 0.50, transient)
    low_body = _smoothstep(0.14, 0.40, base.low)
    bright_penalty = 1.0 - 0.70 * _smoothstep(0.20, 0.45, base.high)
    rescue = _clamp(isolated_other * percussive_shape * low_body * bright_penalty)
    if rescue <= 0.0:
        return adjusted

    redirected = adjusted["other"] * rescue
    adjusted["drums"] = max(adjusted["drums"], redirected)
    adjusted["other"] *= max(0.0, 1.0 - rescue * 1.15)
    # Taiko attacks can create a brief false vocal estimate. Once the same
    # block has a convincing drum rescue, that vocal evidence is not credible.
    adjusted["vocals"] *= max(0.0, 1.0 - rescue * 1.7)
    return adjusted


def _percussive_component_strength(features: AudioFeatures) -> float:
    transient = max(features.percussive, features.percussive_memory)
    attack = _smoothstep(0.20, 0.58, transient)
    low_body = _smoothstep(0.12, 0.42, features.low)
    bright_penalty = 1.0 - 0.65 * _smoothstep(0.22, 0.52, features.high)
    return _clamp(attack * low_body * bright_penalty)


def _refine_model_stems(
    levels: dict[str, float],
    vocal_features: AudioFeatures,
    other_features: AudioFeatures,
) -> dict[str, float]:
    """Reject drum-shaped leakage inside the model's vocal/other stems."""

    adjusted = dict(levels)
    vocal_percussion = _percussive_component_strength(vocal_features)
    tonal_share = vocal_features.harmonic / max(
        1e-9, vocal_features.harmonic + vocal_features.percussive_memory
    )
    # Do not hard-gate on tonality: real singing over a dense mix can look
    # surprisingly percussive. Continuity is handled by the temporal gate
    # below; tonality only supplies a modest confidence adjustment.
    vocal_confidence = 0.65 + 0.35 * _smoothstep(0.35, 0.68, tonal_share)
    vocal_before = adjusted["vocals"]
    adjusted["vocals"] *= vocal_confidence

    other_percussion = _percussive_component_strength(other_features)
    rescued = max(
        vocal_before * vocal_percussion,
        adjusted["other"] * other_percussion,
    )
    adjusted["drums"] = max(adjusted["drums"], rescued)
    adjusted["other"] *= max(0.0, 1.0 - other_percussion * 1.15)
    return adjusted


def _stabilize_vocal_activity(presence: float, score: float) -> tuple[float, float]:
    """Require a short sustained vocal run before animating the singer."""

    if score >= 0.12:
        presence += (1.0 - presence) * 0.16
    else:
        presence *= 0.65
    gate = _smoothstep(0.58, 0.78, presence)
    return presence, score * gate


class StemgenAnalyzer:
    """Run the causal StemgenRT graph and expose semantic stem activity."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        session_factory: Callable | None = None,
    ) -> None:
        self.model_path = find_model_path(model_path)
        if session_factory is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        else:
            self._session = session_factory(str(self.model_path))

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        self._input_names = [item.name for item in inputs]
        self._output_names = [item.name for item in outputs]
        if not inputs or inputs[0].name != "audio_chunk":
            raise RuntimeError("modèle StemgenRT incompatible : audio_chunk absent")
        if "separated_chunk" not in self._output_names:
            raise RuntimeError("modèle StemgenRT incompatible : sortie séparée absente")

        self._states: dict[str, np.ndarray] = {}
        for item in inputs[1:]:
            if any(not isinstance(size, int) for size in item.shape):
                raise RuntimeError(f"forme d'état StemgenRT invalide : {item.name}")
            self._states[item.name] = np.zeros(tuple(item.shape), dtype=np.float32)

        metadata = self._session.get_modelmeta().custom_metadata_map
        source_order = metadata.get("hs_tasnet.source_order", "drums,bass,vocals,other")
        if source_order.startswith("["):
            source_order = ",".join(json.loads(source_order))
        self._source_order = tuple(part.strip() for part in source_order.split(","))
        required = {"drums", "bass", "vocals", "other"}
        if set(self._source_order) != required:
            raise RuntimeError(f"ordre de sources StemgenRT inconnu : {source_order}")

        self._base = AudioAnalyzer()
        self._component_analyzers = {
            "vocals": AudioAnalyzer(),
            "other": AudioAnalyzer(),
        }
        self._pending = np.empty((0, 2), dtype=np.float32)
        self._discard_first = True
        self._stem_levels = np.zeros(4, dtype=np.float32)
        self._vocal_presence = 0.0
        self.last_diagnostics: dict[str, object] = {}

    @property
    def provider(self) -> str:
        providers = self._session.get_providers()
        return providers[0] if providers else "ONNX"

    def process(
        self,
        block: np.ndarray,
        sample_rate: int,
        *,
        now: float | None = None,
    ) -> AudioFeatures:
        if sample_rate != STEMGEN_SAMPLE_RATE:
            raise ValueError(
                f"StemgenRT exige {STEMGEN_SAMPLE_RATE} Hz, reçu {sample_rate} Hz"
            )

        samples = np.asarray(block, dtype=np.float32)
        if samples.ndim == 1:
            stereo = np.repeat(samples[:, None], 2, axis=1)
        elif samples.ndim == 2:
            if samples.shape[1] == 1:
                stereo = np.repeat(samples, 2, axis=1)
            else:
                stereo = samples[:, :2]
        else:
            stereo = samples.reshape(-1, 1).repeat(2, axis=1)
        stereo = np.ascontiguousarray(np.nan_to_num(stereo, copy=False))
        base = self._base.process(stereo, sample_rate, now=now)

        self._pending = np.concatenate((self._pending, stereo), axis=0)
        separated_hops: list[np.ndarray] = []
        while self._pending.shape[0] >= STEMGEN_HOP_SIZE:
            hop = self._pending[:STEMGEN_HOP_SIZE]
            self._pending = self._pending[STEMGEN_HOP_SIZE:]
            separated = self._infer_hop(hop)
            if self._discard_first:
                self._discard_first = False
            else:
                separated_hops.append(separated)

        if not separated_hops or not base.active:
            if not base.active:
                self._stem_levels *= 0.55
            return replace(base, separated=True)

        stems = np.concatenate(separated_hops, axis=2)
        rms_by_source = np.sqrt(np.mean(stems * stems, axis=(1, 2)))
        loudest = max(float(np.max(rms_by_source)), 1e-9)
        targets = np.zeros(4, dtype=np.float32)
        relative_db_by_source = {}
        for index, rms in enumerate(rms_by_source):
            source = self._source_order[index]
            target, relative_db = _stem_activity_target(
                source, float(rms), loudest, base.level
            )
            relative_db_by_source[source] = relative_db
            targets[index] = target
        self._stem_levels = np.maximum(targets, self._stem_levels * 0.78)

        levels = {
            name: _clamp(self._stem_levels[index])
            for index, name in enumerate(self._source_order)
        }
        component_features = {}
        for name, analyzer in self._component_analyzers.items():
            component = stems[self._source_order.index(name)].mean(axis=0)
            # The model's reset preroll discards one 128-sample hop. Keep the
            # lightweight component analyzers on a stable FFT size by padding
            # that first block back to the capture block length.
            if component.size < stereo.shape[0]:
                component = np.pad(component, (stereo.shape[0] - component.size, 0))
            elif component.size > stereo.shape[0]:
                component = component[-stereo.shape[0]:]
            component_features[name] = analyzer.process(
                component, STEMGEN_SAMPLE_RATE, now=now
            )
        raw_levels = dict(levels)
        levels = _refine_model_stems(
            levels,
            component_features["vocals"],
            component_features["other"],
        )
        levels = _rescue_percussive_other(levels, base)
        self._vocal_presence, levels["vocals"] = _stabilize_vocal_activity(
            self._vocal_presence, levels["vocals"]
        )
        self.last_diagnostics = {
            "raw_levels": raw_levels,
            "routed_levels": dict(levels),
            "rms": {
                name: float(rms_by_source[index])
                for index, name in enumerate(self._source_order)
            },
            "relative_db": relative_db_by_source,
            "vocals": component_features["vocals"],
            "other": component_features["other"],
        }
        other = stems[self._source_order.index("other")].mean(axis=0)
        low, mid, high = self._spectral_profile(other)

        return replace(
            base,
            separated=True,
            vocal=levels["vocals"],
            percussive=levels["drums"],
            percussive_memory=levels["drums"],
            harmonic=max(levels["bass"], levels["vocals"], levels["other"]),
            stem_drums=levels["drums"],
            stem_bass=levels["bass"],
            stem_vocals=levels["vocals"],
            stem_other=levels["other"],
            stem_other_low=low,
            stem_other_mid=mid,
            stem_other_high=high,
        )

    def _infer_hop(self, hop: np.ndarray) -> np.ndarray:
        feed = {
            "audio_chunk": np.ascontiguousarray(hop.T[None, :, :], dtype=np.float32),
            **self._states,
        }
        values = self._session.run(self._output_names, feed)
        result = dict(zip(self._output_names, values))
        for name in tuple(self._states):
            next_name = f"next_{name}"
            if next_name not in result:
                raise RuntimeError(f"état StemgenRT manquant : {next_name}")
            self._states[name] = result[next_name]
        return np.asarray(result["separated_chunk"][0], dtype=np.float32)

    @staticmethod
    def _spectral_profile(samples: np.ndarray) -> tuple[float, float, float]:
        if samples.size < 16:
            return 0.0, 0.0, 0.0
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size))) ** 2
        frequencies = np.fft.rfftfreq(samples.size, 1.0 / STEMGEN_SAMPLE_RATE)
        audible = (frequencies >= 40) & (frequencies <= 12_000)
        total = float(spectrum[audible].sum())
        if total <= 1e-12:
            return 0.0, 0.0, 0.0

        def share(start: float, end: float) -> float:
            mask = (frequencies >= start) & (frequencies < end)
            return _clamp(math.sqrt(float(spectrum[mask].sum()) / total))

        return share(40, 250), share(250, 2_500), share(2_500, 12_000)
