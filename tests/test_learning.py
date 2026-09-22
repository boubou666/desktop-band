import json
from pathlib import Path
import shutil
import unittest
import uuid
import wave
import zipfile
import io

import numpy as np

from desktop_band.classifier import CLASSIFIER_LABELS
from desktop_band.learning import dataset_summary, export_model_bundle


class LearningTests(unittest.TestCase):
    def make_workspace(self) -> Path:
        path = Path.cwd() / "tests" / ("_tmp_learning_" + uuid.uuid4().hex)
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    @staticmethod
    def write_audio(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(2)
            destination.setsampwidth(2)
            destination.setframerate(44_100)
            destination.writeframes(np.zeros((44_100, 2), dtype="<i2").tobytes())

    def test_dataset_lists_tracks_and_requires_repeated_role_coverage(self):
        workspace = self.make_workspace()
        training = workspace / "training"
        clips = []
        for index in range(3):
            audio = f"audio/track-{index}.wav"
            self.write_audio(training / audio)
            clips.append({
                "audio": audio,
                "global_labels": list(CLASSIFIER_LABELS),
                "segments": [{
                    "start": 0.0,
                    "end": 1.0,
                    "labels": list(CLASSIFIER_LABELS),
                }],
            })
        (training / "annotations.json").write_text(
            json.dumps({"clips": clips}), encoding="utf-8"
        )
        summary = dataset_summary(training)
        self.assertTrue(summary["ready"])
        self.assertEqual(summary["track_count"], 3)
        self.assertEqual(summary["totals"]["singer"]["tracks"], 3)

    def test_export_contains_model_schema_and_no_audio(self):
        workspace = self.make_workspace()
        model_dir = workspace / "desktop_band" / "models"
        model_dir.mkdir(parents=True)
        shutil.copy2(
            Path("desktop_band/models/instrument_classifier.onnx"),
            model_dir / "instrument_classifier.onnx",
        )
        filename, payload = export_model_bundle(workspace)
        self.assertTrue(filename.endswith(".zip"))
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIn("instrument_classifier.onnx", archive.namelist())
            metadata = json.loads(archive.read("model.json"))
            self.assertEqual(metadata["labels"], list(CLASSIFIER_LABELS))
            self.assertFalse(any(name.endswith(".wav") for name in archive.namelist()))

    def test_exclusive_track_ignores_other_timeline_labels(self):
        workspace = self.make_workspace()
        training = workspace / "training"
        self.write_audio(training / "audio" / "bass-solo.wav")
        (training / "annotations.json").write_text(json.dumps({"clips": [{
            "audio": "audio/bass-solo.wav",
            "global_labels": ["bassist"],
            "exclusive_label": "bassist",
            "segments": [{
                "start": 0.0,
                "end": 1.0,
                "labels": ["singer", "drummer"],
            }],
        }]}), encoding="utf-8")

        summary = dataset_summary(training)
        track = summary["tracks"][0]
        self.assertEqual(track["exclusive_label"], "bassist")
        self.assertEqual(track["labels"], ["bassist"])
        self.assertEqual(summary["totals"]["bassist"]["seconds"], 1.0)
        self.assertEqual(summary["totals"]["singer"]["segments"], 0)
        self.assertEqual(summary["totals"]["drummer"]["segments"], 0)


if __name__ == "__main__":
    unittest.main()
