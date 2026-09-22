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


def _audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as source:
            return source.getnframes() / max(1, source.getframerate())
    except (OSError, wave.Error):
        return 0.0


def dataset_summary(training_dir: Path) -> dict[str, object]:
    manifest_path = training_dir / "annotations.json"
    manifest = {"clips": []}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    totals = {role: {"segments": 0, "seconds": 0.0, "tracks": 0}
              for role in CLASSIFIER_LABELS}
    tracks = []
    for clip in manifest.get("clips", []):
        audio = str(clip.get("audio", ""))
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
            "title": Path(audio).stem,
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
            }
        threading.Thread(target=self._run, daemon=True).start()
        return self.snapshot()

    def _run_command(self, arguments: list[str]) -> str:
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=self.workspace,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(detail[-1200:] or "commande d'entraînement échouée")
        return result.stdout.strip()

    def _run(self) -> None:
        training = self.workspace / "training"
        features = training / "features.jsonl"
        candidate = training / "candidate_classifier.onnx"
        report_path = training / "training_report.json"
        baseline = self.workspace / "desktop_band" / "models" / "instrument_classifier.onnx"
        try:
            self._run_command([
                "tools/collect_features.py",
                "training/annotations.json",
                "--output", str(features),
            ])
            self._update(
                step="train", progress=70,
                message="Entraînement et validation morceau par morceau…",
            )
            self._run_command([
                "tools/train_classifier.py", str(features),
                "--output", str(candidate),
                "--baseline", str(baseline),
                "--report", str(report_path),
            ])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            promoted = bool(report.get("improved"))
            if promoted:
                archive = training / "models"
                archive.mkdir(parents=True, exist_ok=True)
                backup = archive / f"instrument_classifier-{int(time.time())}.onnx"
                shutil.copy2(baseline, backup)
                shutil.copy2(candidate, baseline)
                message = "Nouveau modèle validé et activé au prochain redémarrage."
            else:
                message = "Candidat conservé, mais non activé : il ne bat pas le modèle actuel."
            report["promoted"] = promoted
            self._update(
                status="succeeded", step="done", progress=100,
                message=message, report=report,
            )
        except Exception as error:
            self._update(
                status="failed", step="error", progress=100,
                message=str(error), report=None,
            )
