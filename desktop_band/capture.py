"""Desktop loopback capture and a deterministic visual demo source."""

from __future__ import annotations

import math
import threading
import time

from .analysis import AudioAnalyzer, AudioFeatures


class FeatureSource:
    """Thread-safe enough feature source: object assignment is atomic in CPython."""

    def __init__(self) -> None:
        self.features = AudioFeatures()
        self.status = "initialisation"
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    def _run(self) -> None:  # pragma: no cover - implemented by sources
        raise NotImplementedError


class SystemAudioSource(FeatureSource):
    """Capture the default Windows output through SoundCard/WASAPI loopback."""

    def __init__(
        self,
        sample_rate: int = 48_000,
        block_size: int = 1_024,
        *,
        analysis: str = "stemgen",
        model_path: str | None = None,
    ) -> None:
        super().__init__()
        self.analysis = analysis
        self.sample_rate = 44_100 if analysis == "stemgen" else sample_rate
        self.block_size = block_size
        self.model_path = model_path

    def _run(self) -> None:
        try:
            import soundcard as sc
        except Exception as exc:
            self.error = str(exc)
            self.status = "capture indisponible"
            self.features = AudioFeatures()
            return

        analysis_warning = None
        if self.analysis == "stemgen":
            try:
                from .stemgen import StemgenAnalyzer

                analyzer = StemgenAnalyzer(self.model_path)
                analysis_name = "StemgenRT"
            except Exception as exc:
                analyzer = AudioAnalyzer()
                analysis_name = "analyse légère"
                analysis_warning = "StemgenRT indisponible : {}".format(exc)
        else:
            analyzer = AudioAnalyzer()
            analysis_name = "analyse légère"

        while not self._stop.is_set():
            try:
                speaker = sc.default_speaker()
                if speaker is None:
                    raise RuntimeError("aucune sortie audio par défaut")
                loopback = _loopback_for_speaker(sc, speaker)
                if loopback is None:
                    raise RuntimeError("aucune source loopback WASAPI")

                selected_key = _device_key(speaker)
                self.status = "{} : {}".format(analysis_name, speaker.name)
                self.error = analysis_warning
                with loopback.recorder(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size * 2,
                ) as recorder:
                    next_device_check = time.monotonic() + 0.75
                    while not self._stop.is_set():
                        block = recorder.record(numframes=self.block_size)
                        self.features = analyzer.process(block, self.sample_rate)
                        now = time.monotonic()
                        if now < next_device_check:
                            continue
                        next_device_check = now + 0.75
                        current_speaker = sc.default_speaker()
                        if (
                            current_speaker is None
                            or _device_key(current_speaker) != selected_key
                        ):
                            self.status = "changement de sortie audio"
                            break
            except Exception as exc:
                self.error = str(exc)
                self.status = "reconnexion audio"
                self.features = AudioFeatures(bpm=self.features.bpm)
                if self._stop.wait(0.5):
                    break


def _device_key(device) -> str:
    """Return the most stable identity SoundCard exposes for a device."""

    identifier = getattr(device, "id", None)
    if identifier is not None:
        return str(identifier)
    return str(getattr(device, "name", ""))


def _loopback_for_speaker(sc, speaker):
    loopback = sc.get_microphone(speaker.name, include_loopback=True)
    if loopback is not None and getattr(loopback, "isloopback", False):
        return loopback
    candidates = [
        device
        for device in sc.all_microphones(include_loopback=True)
        if getattr(device, "isloopback", False)
    ]
    speaker_name = str(getattr(speaker, "name", "")).casefold()
    return next(
        (
            device
            for device in candidates
            if speaker_name and speaker_name in str(device.name).casefold()
        ),
        candidates[0] if candidates else None,
    )


class DemoAudioSource(FeatureSource):
    """Generate repeatable musical controls without touching an audio device."""

    def _run(self) -> None:
        self.status = "mode démo"
        started = time.monotonic()
        while not self._stop.wait(1.0 / 40.0):
            elapsed = time.monotonic() - started
            beat_phase = (elapsed * 2.0) % 1.0
            onset = max(0.0, 1.0 - beat_phase * 8.0)
            measure = int(elapsed * 2.0) % 8
            low = 0.88 if measure in (0, 4) else 0.38
            vocal = 0.45 + 0.30 * math.sin(elapsed * 3.1)
            self.features = AudioFeatures(
                active=True,
                level=0.68 + 0.16 * math.sin(elapsed * 1.7),
                low=low,
                mid=0.58 + 0.24 * math.sin(elapsed * 2.3),
                high=0.38 + onset * 0.45,
                onset=onset,
                vocal=max(0.0, vocal),
                percussive=max(0.15, onset),
                percussive_memory=max(0.18, onset),
                harmonic=0.72,
                percussive_low=low * onset,
                percussive_mid=0.32 * onset,
                percussive_high=(0.38 + onset * 0.45) * onset,
                harmonic_low=low * 0.55,
                harmonic_mid=(0.58 + 0.24 * math.sin(elapsed * 2.3)) * 0.8,
                harmonic_high=0.28,
                bpm=120.0,
            )
