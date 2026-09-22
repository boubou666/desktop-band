"""Privacy-preserving feature recording and live mistake bookmarks."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import time


class FeatureRecorder:
    """Keep a short feature-only rolling buffer; never retain raw audio."""

    def __init__(self, training_dir: Path, seconds: float = 12.0) -> None:
        self.training_dir = training_dir
        self.seconds = seconds
        self._frames: deque[dict[str, object]] = deque()
        self._recording: list[dict[str, object]] | None = None
        self._recording_started = 0.0

    @property
    def recording(self) -> bool:
        return self._recording is not None

    def capture(self, features, decision=None) -> None:
        now = time.time()
        frame = {
            "timestamp": now,
            "features": asdict(features),
            "scores": dict(getattr(decision, "scores", {}) or {}),
            "states": dict(getattr(decision, "states", {}) or {}),
            "dominant_role": getattr(decision, "dominant_role", None),
        }
        self._frames.append(frame)
        while self._frames and now - float(self._frames[0]["timestamp"]) > self.seconds:
            self._frames.popleft()
        if self._recording is not None:
            relative = dict(frame)
            relative["time"] = round(now - self._recording_started, 4)
            self._recording.append(relative)

    def mark_issue(self, note: str = "mauvaise détection") -> Path:
        if not self._frames:
            raise ValueError("aucune caractéristique audio récente")
        output_dir = self.training_dir / "review_queue"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = output_dir / f"review-{stamp}.json"
        document = {
            "schema_version": 1,
            "note": note,
            "audio_included": False,
            "created_at": int(time.time()),
            "frames": list(self._frames),
        }
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def toggle_session(self) -> Path | None:
        if self._recording is None:
            self._recording = []
            self._recording_started = time.time()
            return None
        output_dir = self.training_dir / "replays"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = output_dir / f"session-{stamp}.jsonl"
        frames = self._recording
        self._recording = None
        path.write_text(
            "".join(json.dumps(frame, ensure_ascii=False) + "\n" for frame in frames),
            encoding="utf-8",
        )
        return path


def list_review_bookmarks(training_dir: Path) -> list[dict[str, object]]:
    output = []
    for path in sorted((training_dir / "review_queue").glob("*.json"), reverse=True):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            frames = document.get("frames", [])
            output.append({
                "name": path.name,
                "note": document.get("note", "mauvaise détection"),
                "frames": len(frames),
                "seconds": round(
                    max(0.0, float(frames[-1]["timestamp"]) - float(frames[0]["timestamp"]))
                    if len(frames) > 1 else 0.0,
                    1,
                ),
                "labels": document.get("correct_labels", []),
            })
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return output


def label_review_bookmark(
    training_dir: Path, name: str, labels: list[str]
) -> dict[str, object]:
    from .classifier import CLASSIFIER_LABELS

    path = training_dir / "review_queue" / Path(name).name
    if not path.is_file():
        raise ValueError("signalement live inconnu")
    selected = [label for label in CLASSIFIER_LABELS if label in set(labels)]
    if not selected:
        raise ValueError("choisis au moins un instrument correct")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["correct_labels"] = selected
    document["labeled_at"] = int(time.time())
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"ok": True, "name": path.name, "labels": selected}
