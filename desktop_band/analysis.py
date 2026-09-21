"""Small real-time audio feature extractor used by the animation layer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioFeatures:
    """Normalized values consumed by the renderer.

    Values other than ``bpm`` are clamped to the 0..1 range. ``vocal`` is a
    deliberately modest hint derived from tonal mid-range energy; it is not an
    instrument classifier.
    """

    active: bool = False
    level: float = 0.0
    low: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    onset: float = 0.0
    vocal: float = 0.0
    percussive: float = 0.0
    percussive_memory: float = 0.0
    harmonic: float = 0.0
    percussive_low: float = 0.0
    percussive_mid: float = 0.0
    percussive_high: float = 0.0
    harmonic_low: float = 0.0
    harmonic_mid: float = 0.0
    harmonic_high: float = 0.0
    separated: bool = False
    stem_drums: float = 0.0
    stem_bass: float = 0.0
    stem_vocals: float = 0.0
    stem_other: float = 0.0
    stem_other_low: float = 0.0
    stem_other_mid: float = 0.0
    stem_other_high: float = 0.0
    spectral_centroid: float = 0.0
    spectral_flatness: float = 0.0
    pitch_stability: float = 0.0
    note_density: float = 0.0
    stem_other_onset: float = 0.0
    stem_other_harmonic: float = 0.0
    stem_other_centroid: float = 0.0
    stem_other_flatness: float = 0.0
    stem_other_pitch_stability: float = 0.0
    stem_other_note_density: float = 0.0
    bpm: float = 100.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _estimate_bpm_from_envelope(
    timestamps: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float] | None:
    """Estimate tempo from a continuous rhythmic envelope autocorrelation."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if timestamps.size < 80 or values.size != timestamps.size:
        return None
    intervals = np.diff(timestamps)
    intervals = intervals[(intervals > 0.005) & (intervals < 0.1)]
    if intervals.size < 40:
        return None
    dt = float(np.median(intervals))
    if timestamps[-1] - timestamps[0] < 3.8:
        return None

    centered = values - float(np.mean(values))
    energy = float(np.dot(centered, centered))
    if energy < 1e-5:
        return None

    minimum_lag = max(2, round(60.0 / 180.0 / dt))
    maximum_lag = min(centered.size // 2, round(60.0 / 50.0 / dt))
    scores = {}
    for lag in range(minimum_lag, maximum_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        denominator = math.sqrt(
            max(1e-12, float(np.dot(left, left) * np.dot(right, right)))
        )
        scores[lag] = float(np.dot(left, right)) / denominator

    peak_score = max(scores.values())
    if peak_score < 0.10:
        return None

    # Exact periodic pulses correlate at the beat, half-time and third-time.
    # Prefer the shortest credible lag only when it is essentially as strong
    # as the global peak. Softer music usually has much weaker fast harmonics,
    # so its genuine 55–80 BPM fundamental remains selected.
    credible_threshold = max(0.10, peak_score * 0.82)
    best_lag = min(
        lag for lag, score in scores.items() if score >= credible_threshold
    )
    best_score = scores[best_lag]

    bpm = 60.0 / (best_lag * dt)
    return max(50.0, min(180.0, bpm)), best_score


class AudioAnalyzer:
    """Convert successive PCM blocks into stable animation controls."""

    def __init__(self) -> None:
        self._peak = 0.025
        self._previous_spectrum: np.ndarray | None = None
        self._flux_average = 0.04
        self._bpm = 100.0
        self._tempo_locked = False
        self._tempo_envelope: deque[tuple[float, float]] = deque(maxlen=620)
        self._tempo_estimates: deque[float] = deque(maxlen=5)
        self._last_tempo_update_at = -10.0
        self._previous_rms = 0.0
        self._previous_low = 0.0
        self._percussive_state = 0.0
        self._spectral_history: deque[np.ndarray] = deque(maxlen=17)
        self._pitch_history: deque[float] = deque(maxlen=10)
        self._note_density = 0.0

    def process(
        self,
        block: np.ndarray,
        sample_rate: int,
        *,
        now: float | None = None,
    ) -> AudioFeatures:
        """Analyze one frames-by-channels or mono floating-point block."""

        timestamp = time.monotonic() if now is None else float(now)
        samples = np.asarray(block, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        elif samples.ndim != 1:
            samples = samples.reshape(-1)

        if samples.size < 16 or sample_rate <= 0:
            return AudioFeatures(bpm=self._bpm)

        samples = np.nan_to_num(samples, copy=False)
        rms = float(np.sqrt(np.mean(samples * samples)))
        self._peak = max(rms, self._peak * 0.997)
        level = _clamp(rms / max(0.008, self._peak))
        # WASAPI can expose a tiny residual signal while applications are
        # silent. Treat anything below roughly -52 dBFS as digital silence so
        # the band does not twitch without audible sound.
        active = rms >= 0.0025

        windowed = samples * np.hanning(samples.size)
        spectrum = np.abs(np.fft.rfft(windowed))
        power = spectrum * spectrum
        frequencies = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
        total_power = float(power[(frequencies >= 20) & (frequencies <= 16_000)].sum())

        def band(start: float, end: float) -> tuple[float, float]:
            mask = (frequencies >= start) & (frequencies < end)
            ratio = float(power[mask].sum()) / max(total_power, 1e-12)
            # Square-root compression makes quiet but meaningful bands visible.
            return ratio, _clamp(math.sqrt(max(0.0, ratio)) * 1.65 * level)

        low_ratio, low = band(20, 180)
        mid_ratio, mid = band(180, 2_200)
        high_ratio, high = band(2_200, 12_000)

        onset = 0.0
        if self._previous_spectrum is not None and self._previous_spectrum.shape == spectrum.shape:
            positive_flux = np.maximum(spectrum - self._previous_spectrum, 0.0)
            flux = float(positive_flux.sum() / max(spectrum.sum(), 1e-12))
            baseline = max(0.008, self._flux_average)
            onset = _clamp((flux - baseline * 1.15) / (baseline * 2.5)) * level
            self._flux_average = 0.96 * self._flux_average + 0.04 * flux
        self._previous_spectrum = spectrum

        peak_sample = float(np.max(np.abs(samples)))
        crest = peak_sample / max(rms, 1e-9)
        crest_score = _clamp((crest - 1.7) / 5.0)
        attack = _clamp(
            (rms - self._previous_rms) / max(0.0025, self._previous_rms) * 0.35
        )
        self._previous_rms = rms
        percussive_now = _clamp(max(onset * 1.15, attack * 0.7 + crest_score * 0.45))
        low_attack = max(0.0, low - self._previous_low)
        self._previous_low = low
        tempo_pulse = max(onset, low_attack * 2.0, low * 0.35)
        self._track_tempo(timestamp, tempo_pulse, active)
        # Keep the percussive classification alive during the decay of a drum
        # hit. Otherwise its resonant tail would look like a bass or keyboard.
        # Large drums such as taiko can resonate for well over a second. Keep
        # the classification long enough for that tail instead of immediately
        # re-labelling it as bass or another tonal instrument.
        self._percussive_state = max(percussive_now, self._percussive_state * 0.98)

        harmonic_power, percussive_power = self._separate_power(
            spectrum, power, self._percussive_state
        )

        def separated_band(component, start: float, end: float) -> float:
            mask = (frequencies >= start) & (frequencies < end)
            ratio = float(component[mask].sum()) / max(total_power, 1e-12)
            return _clamp(math.sqrt(max(0.0, ratio)) * 1.65 * level)

        percussive_low = separated_band(percussive_power, 20, 180)
        percussive_mid = separated_band(percussive_power, 180, 2_200)
        percussive_high = separated_band(percussive_power, 2_200, 12_000)
        harmonic_low = separated_band(harmonic_power, 20, 180)
        harmonic_mid = separated_band(harmonic_power, 180, 2_200)
        harmonic_high = separated_band(harmonic_power, 2_200, 12_000)

        non_zero = spectrum[1:]
        if non_zero.size:
            arithmetic = float(np.mean(non_zero)) + 1e-12
            geometric = float(np.exp(np.mean(np.log(non_zero + 1e-12))))
            spectral_flatness = _clamp(geometric / arithmetic)
            tonality = 1.0 - spectral_flatness
        else:
            spectral_flatness = 0.0
            tonality = 0.0
        audible_mask = (frequencies >= 20) & (frequencies <= 16_000)
        audible_magnitude = spectrum[audible_mask]
        audible_frequencies = frequencies[audible_mask]
        magnitude_sum = float(audible_magnitude.sum())
        centroid_hz = (
            float(np.dot(audible_frequencies, audible_magnitude)) / magnitude_sum
            if magnitude_sum > 1e-12
            else 0.0
        )
        spectral_centroid = _clamp(centroid_hz / 6_000.0)

        pitch_mask = (frequencies >= 80) & (frequencies <= 4_000)
        pitch_slice = spectrum[pitch_mask]
        if active and pitch_slice.size and float(pitch_slice.max()) > 1e-9:
            dominant_hz = float(frequencies[pitch_mask][int(np.argmax(pitch_slice))])
            self._pitch_history.append(dominant_hz)
        if len(self._pitch_history) >= 4:
            pitch_values = np.asarray(self._pitch_history, dtype=np.float64)
            coefficient = float(np.std(pitch_values) / max(1.0, np.mean(pitch_values)))
            pitch_stability = _clamp(1.0 - coefficient / 0.48)
        else:
            pitch_stability = 0.0
        density_target = _clamp(onset * 1.35 + attack * 0.55)
        self._note_density = 0.88 * self._note_density + 0.12 * density_target
        percussive_ratio = float(percussive_power[audible_mask].sum()) / max(
            total_power, 1e-12
        )
        harmonic_ratio = float(harmonic_power[audible_mask].sum()) / max(
            total_power, 1e-12
        )
        percussive = _clamp(max(level * percussive_ratio * 1.25, onset))
        percussive_memory = _clamp(
            self._percussive_state * math.sqrt(max(0.0, level))
        )
        harmonic = _clamp(level * harmonic_ratio * max(0.35, tonality) * 1.3)
        vocal = _clamp(harmonic_mid * tonality * 1.15)

        if not active:
            self._percussive_state *= 0.7
            self._pitch_history.clear()
            self._note_density *= 0.5
            return AudioFeatures(bpm=self._bpm)
        return AudioFeatures(
            active=True,
            level=level,
            low=low,
            mid=mid,
            high=high,
            onset=onset,
            vocal=vocal,
            percussive=percussive,
            percussive_memory=percussive_memory,
            harmonic=harmonic,
            percussive_low=percussive_low,
            percussive_mid=percussive_mid,
            percussive_high=percussive_high,
            harmonic_low=harmonic_low,
            harmonic_mid=harmonic_mid,
            harmonic_high=harmonic_high,
            spectral_centroid=spectral_centroid,
            spectral_flatness=spectral_flatness,
            pitch_stability=pitch_stability,
            note_density=self._note_density,
            bpm=self._bpm,
        )

    def _separate_power(
        self, spectrum: np.ndarray, power: np.ndarray, percussive_hint: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate HPSS with time/frequency median filters.

        A stable frequency across frames is harmonic. A broad shape within one
        frame is percussive. The first frames use the transient detector while
        the rolling spectrogram is warming up.
        """

        self._spectral_history.append(spectrum.copy())
        if len(self._spectral_history) < 5:
            percussive_mask = np.full_like(power, _clamp(percussive_hint))
            return power * (1.0 - percussive_mask), power * percussive_mask

        history = np.stack(self._spectral_history, axis=0)
        harmonic_median = np.median(history, axis=0)

        kernel = 17
        radius = kernel // 2
        padded = np.pad(spectrum, (radius, radius), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
        percussive_median = np.median(windows, axis=-1)

        harmonic_score = harmonic_median * harmonic_median
        percussive_score = percussive_median * percussive_median
        denominator = harmonic_score + percussive_score + 1e-12
        harmonic_mask = harmonic_score / denominator
        percussive_mask = percussive_score / denominator
        return power * harmonic_mask, power * percussive_mask

    def _track_tempo(self, timestamp: float, pulse: float, active: bool) -> None:
        self._tempo_envelope.append((timestamp, pulse if active else 0.0))
        if not active or timestamp - self._last_tempo_update_at < 0.45:
            return
        self._last_tempo_update_at = timestamp
        recent = [
            sample
            for sample in self._tempo_envelope
            if timestamp - sample[0] <= 10.0
        ]
        if len(recent) < 80:
            return
        estimate = _estimate_bpm_from_envelope(
            np.asarray([sample[0] for sample in recent]),
            np.asarray([sample[1] for sample in recent]),
        )
        if estimate is None:
            return
        bpm, confidence = estimate
        self._tempo_estimates.append(bpm)
        stable_bpm = float(np.median(np.asarray(self._tempo_estimates)))
        if not self._tempo_locked:
            self._bpm = stable_bpm
            self._tempo_locked = True
            return
        blend = 0.32 if confidence >= 0.2 else 0.18
        self._bpm = (1.0 - blend) * self._bpm + blend * stable_bpm
