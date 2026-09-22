import json
from pathlib import Path
import shutil
import unittest
import uuid
import wave

import numpy as np

from desktop_band.analysis import AudioFeatures
from desktop_band.annotator import AnalysisFrame, review_windows_from_frames
from desktop_band.detection import RoleDecision
from desktop_band.diagnostics import (
    FeatureRecorder,
    label_review_bookmark,
    list_review_bookmarks,
)
from desktop_band.formations import FormationStore
from desktop_band.slakh import import_slakh, role_for_stem
from desktop_band.training_tools import (
    activate_model,
    build_training_manifest,
    generate_synthetic_mixes,
    list_models,
)
from tools.train_classifier import (
    augment_dataset,
    choose_validation_sources,
    confusion_by_role,
    optimal_thresholds,
    per_role_f1,
)


class LabWorkflowTests(unittest.TestCase):
    def make_workspace(self) -> Path:
        path = Path.cwd() / "tests" / ("_tmp_lab_" + uuid.uuid4().hex)
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    @staticmethod
    def write_tone(path: Path, frequency: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        timeline = np.arange(44_100, dtype=np.float32) / 44_100
        tone = np.sin(timeline * np.pi * 2 * frequency) * 0.2
        stereo = np.column_stack((tone, tone))
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(2)
            destination.setsampwidth(2)
            destination.setframerate(44_100)
            destination.writeframes((stereo * 32767).astype("<i2").tobytes())

    def test_uncertain_competing_roles_are_grouped_for_review(self):
        frames = [
            AnalysisFrame(0.0, 0.1, {"guitarist": 0.42, "keyboard": 0.38}),
            AnalysisFrame(0.1, 0.2, {"guitarist": 0.40, "keyboard": 0.37}),
            AnalysisFrame(0.2, 0.3, {"guitarist": 0.44, "keyboard": 0.39}),
        ]
        reviews = review_windows_from_frames(frames)
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["labels"], ["guitarist", "keyboard"])

    def test_live_bookmark_and_replay_never_contain_audio(self):
        training = self.make_workspace() / "training"
        recorder = FeatureRecorder(training)
        decision = RoleDecision(
            {"bassist": 0.8}, {"bassist": 0.8}, {"bassist": "playing"},
            "bassist", 0.8,
        )
        recorder.capture(AudioFeatures(active=True, stem_bass=0.8), decision)
        bookmark = recorder.mark_issue()
        document = json.loads(bookmark.read_text(encoding="utf-8"))
        self.assertFalse(document["audio_included"])
        self.assertNotIn("audio", document["frames"][0])
        self.assertEqual(len(list_review_bookmarks(training)), 1)
        label_review_bookmark(training, bookmark.name, ["bassist"])
        self.assertEqual(list_review_bookmarks(training)[0]["labels"], ["bassist"])

        self.assertIsNone(recorder.toggle_session())
        recorder.capture(AudioFeatures(active=True, bpm=128), decision)
        replay = recorder.toggle_session()
        self.assertTrue(replay.is_file())

    def test_custom_formations_are_validated_and_persisted(self):
        store = FormationStore(self.make_workspace() / "formations.json")
        saved = store.save("Trio sérieux", {
            "roles": ["vibing", "bassist", "drummer", "unknown"],
            "decor": "garage",
            "lighting": False,
            "placements": {
                "bassist": {"x": 0.2, "scale": 1.2, "depth": 2},
            },
        })
        self.assertEqual(saved["roles"], ["vibing", "bassist", "drummer"])
        self.assertFalse(store.list()["Trio sérieux"]["lighting"])
        self.assertEqual(saved["placements"]["bassist"]["depth"], 2)

    def test_synthetic_mixer_combines_exclusive_tracks(self):
        training = self.make_workspace() / "training"
        self.write_tone(training / "audio" / "bass.wav", 110)
        self.write_tone(training / "audio" / "voice.wav", 440)
        (training / "annotations.json").write_text(json.dumps({"clips": [
            {"audio": "audio/bass.wav", "exclusive_label": "bassist"},
            {"audio": "audio/voice.wav", "exclusive_label": "singer"},
        ]}), encoding="utf-8")
        result = generate_synthetic_mixes(training, count=1, seconds=0.1)
        self.assertEqual(result["mixes"], 1)
        generated = json.loads(
            (training / "synthetic_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(generated["clips"][0]["global_labels"]), {"bassist", "singer"}
        )
        self.assertTrue((training / generated["clips"][0]["audio"]).is_file())

    def test_slakh_import_maps_stems_and_preserves_validation_group(self):
        workspace = self.make_workspace()
        dataset = workspace / "BabySlakh" / "train" / "Track00001"
        stems = dataset / "stems"
        for name, frequency in (("S00", 110), ("S01", 220), ("S02", 330)):
            self.write_tone(stems / f"{name}.wav", frequency)
        (dataset / "metadata.yaml").write_text(
            "audio_dir: stems\n"
            "stems:\n"
            "  S00:\n"
            "    audio_rendered: true\n"
            "    inst_class: Bass\n"
            "    is_drum: false\n"
            "  S01:\n"
            "    audio_rendered: true\n"
            "    inst_class: Guitar\n"
            "    is_drum: false\n"
            "  S02:\n"
            "    audio_rendered: true\n"
            "    inst_class: Drums\n"
            "    is_drum: true\n",
            encoding="utf-8",
        )
        training = workspace / "training"
        report = import_slakh(
            workspace / "BabySlakh", training,
            minutes_per_role=0.01, clip_seconds=0.25,
        )
        self.assertEqual(
            {clip["exclusive_label"] for clip in report["clips"]},
            {"bassist", "guitarist", "drummer"},
        )
        self.assertEqual(
            {clip["validation_group"] for clip in report["clips"]},
            {"slakh:Track00001"},
        )
        combined = json.loads(
            build_training_manifest(training).read_text(encoding="utf-8")
        )
        self.assertEqual(len(combined["clips"]), 3)

    def test_slakh_taxonomy_keeps_drums_and_percussion_separate(self):
        self.assertEqual(role_for_stem({"is_drum": True}), "drummer")
        self.assertEqual(
            role_for_stem({"inst_class": "Chromatic Percussion"}),
            "percussion",
        )
        self.assertEqual(role_for_stem({"inst_class": "Piano"}), "keyboard")
        self.assertIsNone(role_for_stem({"inst_class": "Strings"}))

    def test_training_metrics_and_augmentation_are_multi_label(self):
        probabilities = np.asarray([
            [0.9, 0.2, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.8, 0.1, 0.1, 0.1, 0.7],
            [0.1, 0.1, 0.9, 0.1, 0.1, 0.1],
        ], dtype=np.float32)
        targets = np.asarray([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)
        thresholds = optimal_thresholds(probabilities, targets)
        scores = per_role_f1(probabilities, targets, thresholds)
        confusion = confusion_by_role(probabilities, targets, thresholds)
        self.assertEqual(scores["singer"], 1.0)
        self.assertEqual(confusion["percussion"]["tp"], 1)
        x = np.ones((3, 23), dtype=np.float32) * 0.5
        sources = np.asarray(["a", "b", "c"])
        augmented, labels, names = augment_dataset(x, targets, sources, 1, 7)
        self.assertEqual(augmented.shape, (6, 23))
        self.assertEqual(labels.shape, (6, 6))
        self.assertTrue(names[-1].endswith("#aug1"))

    def test_validation_reserves_groups_covering_every_role(self):
        sources = np.asarray([
            "voice-a", "voice-a", "voice-b", "voice-b",
            "rhythm-a", "rhythm-a", "rhythm-b", "rhythm-b",
            "band-a", "band-a", "band-b", "band-b",
        ])
        targets = np.asarray([
            [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0],
        ], dtype=np.float32)
        selected = choose_validation_sources(sources, targets, fraction=0.5)
        validation = np.isin(sources, selected)
        training = ~validation
        self.assertTrue(np.all(targets[validation].sum(axis=0) > 0))
        self.assertTrue(np.all(targets[training].sum(axis=0) > 0))

    def test_archived_model_can_replace_active_model_with_backup(self):
        workspace = self.make_workspace()
        active = workspace / "desktop_band" / "models" / "instrument_classifier.onnx"
        archive = workspace / "training" / "models" / "old.onnx"
        active.parent.mkdir(parents=True)
        archive.parent.mkdir(parents=True)
        active.write_bytes(b"active-model")
        archive.write_bytes(b"archived-model")
        names = {item["name"] for item in list_models(workspace)}
        self.assertEqual(names, {"active", "old"})
        result = activate_model(workspace, "old")
        self.assertEqual(active.read_bytes(), b"archived-model")
        self.assertTrue(result["backup"].startswith("instrument_classifier-"))


if __name__ == "__main__":
    unittest.main()
