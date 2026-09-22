"""Dataset reporting and guarded background training for Gopnik Lab."""

from __future__ import annotations

import json
import io
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import wave
import zipfile

from .classifier import CLASSIFIER_LABELS, FEATURE_NAMES
from .training_tools import build_training_manifest, load_training_manifest


def _audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as source:
            return source.getnframes() / max(1, source.getframerate())
    except (OSError, wave.Error):
        return 0.0


def dataset_summary(training_dir: Path) -> dict[str, object]:
    manifest = load_training_manifest(training_dir)
    totals = {role: {"segments": 0, "seconds": 0.0, "tracks": 0}
              for role in CLASSIFIER_LABELS}
    tracks = []
    source_keys = set()
    for clip in manifest.get("clips", []):
        audio = str(clip.get("audio", ""))
        source_keys.add(str(clip.get("validation_group") or audio))
        audio_path = training_dir / audio
        audio_duration = _audio_duration(audio_path)
        per_role = {role: {"segments": 0, "seconds": 0.0}
                    for role in CLASSIFIER_LABELS}
        segments = clip.get("segments", [])
        exclusive_label = clip.get("exclusive_label")
        effective_segments = [] if exclusive_label else segments
        for segment in effective_segments:
            duration = max(0.0, float(segment["end"]) - float(segment["start"]))
            for role in set(segment.get("labels", [])):
                if role not in per_role:
                    continue
                per_role[role]["segments"] += 1
                per_role[role]["seconds"] += duration
        weak_labels = not segments and not exclusive_label
        if exclusive_label:
            if exclusive_label in per_role:
                per_role[exclusive_label]["segments"] = 1
                per_role[exclusive_label]["seconds"] = audio_duration
        elif weak_labels:
            for role in set(clip.get("global_labels", [])):
                if role in per_role:
                    per_role[role]["segments"] = 1
                    per_role[role]["seconds"] = audio_duration
        present = []
        for role, values in per_role.items():
            if values["segments"]:
                present.append(role)
                totals[role]["segments"] += values["segments"]
                totals[role]["seconds"] += values["seconds"]
                totals[role]["tracks"] += 1
            values["seconds"] = round(values["seconds"], 2)
        tracks.append({
            "audio": audio,
            "title": str(clip.get("source_title") or Path(audio).stem),
            "source_dataset": clip.get("source_dataset"),
            "duration": round(audio_duration, 2),
            "segments": sum(value["segments"] for value in per_role.values()),
            "labels": present,
            "by_role": per_role,
            "weak_labels": weak_labels,
            "exclusive_label": exclusive_label,
        })
    for values in totals.values():
        values["seconds"] = round(values["seconds"], 2)
    missing = [role for role, values in totals.items() if values["segments"] == 0]
    underrepresented = [
        role for role, values in totals.items() if values["tracks"] < 2
    ]
    reasons = []
    if len(tracks) < 3:
        reasons.append("au moins 3 morceaux corrigés")
    if underrepresented:
        reasons.append(
            "2 morceaux minimum par instrument : "
            + ", ".join(underrepresented)
        )
    return {
        "tracks": tracks,
        "track_count": len(tracks),
        "source_count": len(source_keys),
        "totals": totals,
        "missing_labels": missing,
        "underrepresented_labels": underrepresented,
        "ready": not reasons,
        "readiness": "prêt pour validation" if not reasons else " ; ".join(reasons),
    }


def export_model_bundle(workspace: Path) -> tuple[str, bytes]:
    workspace = workspace.resolve()
    model = workspace / "desktop_band" / "models" / "instrument_classifier.onnx"
    if not model.is_file():
        raise FileNotFoundError("modèle ONNX actif introuvable")
    report_path = workspace / "training" / "training_report.json"
    report = None
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = {
        "format": "gopnik-band-instrument-classifier",
        "schema_version": 1,
        "labels": list(CLASSIFIER_LABELS),
        "features": list(FEATURE_NAMES),
        "training_report": report,
    }
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(model, "instrument_classifier.onnx")
        model_metadata = model.with_suffix(".json")
        if model_metadata.is_file():
            archive.write(model_metadata, "instrument_classifier.json")
        archive.writestr(
            "model.json",
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        )
        archive.writestr(
            "README.txt",
            "Gopnik Band instrument classifier\n"
            "Copy instrument_classifier.onnx into desktop_band/models/.\n"
            "The exact input feature and output label order is in model.json.\n",
        )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"gopnik-band-classifier-{stamp}.zip", payload.getvalue()


class TrainingManager:
    """Run extraction/training once and promote only validated improvements."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self._lock = threading.Lock()
        self._state: dict[str, object] = {
            "status": "idle",
            "step": "",
            "message": "Aucun entraînement lancé.",
            "progress": 0,
            "report": None,
            "elapsed_seconds": 0,
            "eta_seconds": None,
            "current_track": None,
            "cache_hits": 0,
        }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._state)

    def _update(self, **values) -> None:
        with self._lock:
            self._state.update(values)

    def start(self) -> dict[str, object]:
        summary = dataset_summary(self.workspace / "training")
        if not summary["ready"]:
            raise ValueError("dataset pas encore prêt : " + str(summary["readiness"]))
        with self._lock:
            if self._state["status"] == "running":
                raise ValueError("un entraînement est déjà en cours")
            self._state = {
                "status": "running",
                "step": "extract",
                "message": "Extraction des caractéristiques StemgenRT…",
                "progress": 10,
                "report": None,
                "started_at": time.time(),
                "elapsed_seconds": 0,
                "eta_seconds": None,
                "current_track": None,
                "cache_hits": 0,
            }
        threading.Thread(target=self._run, daemon=True).start()
        return self.snapshot()

    def _run_command(
        self,
        arguments: list[str],
        *,
        progress_file: Path | None = None,
        progress_start: float = 0.0,
        progress_span: float = 100.0,
    ) -> str:
        process = subprocess.Popen(
            [sys.executable, *arguments],
            cwd=self.workspace,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        last_progress_mtime = None
        while process.poll() is None:
            if progress_file is not None and progress_file.is_file():
                try:
                    mtime = progress_file.stat().st_mtime_ns
                    if mtime != last_progress_mtime:
                        state = json.loads(progress_file.read_text(encoding="utf-8"))
                        fraction = max(0.0, min(1.0, float(state.get("progress", 0))))
                        self._update(
                            progress=round(progress_start + fraction * progress_span, 1),
                            elapsed_seconds=state.get("elapsed_seconds", 0),
                            eta_seconds=state.get("eta_seconds"),
                            current_track=state.get("current_track"),
                            cache_hits=state.get("cache_hits", self._state.get("cache_hits", 0)),
                        )
                        last_progress_mtime = mtime
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            time.sleep(0.2)
        stdout, stderr = process.communicate()
        if process.returncode:
            detail = (stderr or stdout).strip()
            raise RuntimeError(detail[-1200:] or "commande d'entraînement échouée")
        return stdout.strip()

    def _run(self) -> None:
        training = self.workspace / "training"
        features = training / "features.jsonl"
        candidate = training / "candidate_classifier.onnx"
        report_path = training / "training_report.json"
        progress_path = training / "training_progress.json"
        baseline = self.workspace / "desktop_band" / "models" / "instrument_classifier.onnx"
        try:
            manifest = build_training_manifest(training)
            progress_path.unlink(missing_ok=True)
            self._run_command([
                "tools/collect_features.py",
                str(manifest),
                "--output", str(features),
                "--cache-dir", str(training / "cache"),
                "--progress-file", str(progress_path),
            ], progress_file=progress_path, progress_start=5, progress_span=62)
            self._update(
                step="train", progress=70,
                message="Entraînement et validation morceau par morceau…",
            )
            progress_path.unlink(missing_ok=True)
            self._run_command([
                "tools/train_classifier.py", str(features),
                "--output", str(candidate),
                "--baseline", str(baseline),
                "--report", str(report_path),
                "--progress-file", str(progress_path),
            ], progress_file=progress_path, progress_start=70, progress_span=27)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            promoted = bool(report.get("improved"))
            if promoted:
                archive = training / "models"
                archive.mkdir(parents=True, exist_ok=True)
                backup = archive / f"instrument_classifier-{int(time.time())}.onnx"
                shutil.copy2(baseline, backup)
                baseline_metadata = baseline.with_suffix(".json")
                if baseline_metadata.is_file():
                    shutil.copy2(baseline_metadata, backup.with_suffix(".json"))
                shutil.copy2(candidate, baseline)
                candidate_metadata = candidate.with_suffix(".json")
                if candidate_metadata.is_file():
                    shutil.copy2(candidate_metadata, baseline_metadata)
                message = "Nouveau modèle validé et activé au prochain redémarrage."
            else:
                message = "Candidat conservé, mais non activé : il ne bat pas le modèle actuel."
            report["promoted"] = promoted
            self._update(
                status="succeeded", step="done", progress=100,
                message=message, report=report,
                elapsed_seconds=round(
                    time.time() - float(self._state.get("started_at", time.time())), 1
                ),
                eta_seconds=0,
            )
        except Exception as error:
            self._update(
                status="failed", step="error", progress=100,
                message=str(error), report=None,
            )
