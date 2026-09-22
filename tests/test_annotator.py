import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid
import wave

import numpy as np

from desktop_band.annotator import (
    AnalysisFrame,
    AnnotationStore,
    analyze_wav,
    download_youtube_audio,
    is_youtube_url,
    _ui_html,
    read_pcm_wav,
    safe_audio_name,
    segments_from_frames,
)


class AnnotatorTests(unittest.TestCase):
    def make_workspace(self) -> Path:
        path = Path.cwd() / "tests" / ("_tmp_annotator_" + uuid.uuid4().hex)
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def test_audio_name_rejects_directories_and_unsafe_characters(self):
        self.assertEqual(safe_audio_name("../../Taïko: test.mp3"), "Ta_ko_ test.wav")

    def test_only_youtube_hosts_are_accepted(self):
        self.assertTrue(is_youtube_url("https://youtu.be/C7HL5wYqAbU"))
        self.assertTrue(is_youtube_url("https://music.youtube.com/watch?v=test"))
        self.assertFalse(is_youtube_url("https://youtube.com.example.org/watch?v=test"))
        self.assertFalse(is_youtube_url("file:///private/audio.mp3"))
        with self.assertRaises(ValueError):
            download_youtube_audio(
                "https://example.org/audio", self.make_workspace()
            )

    def test_ui_exposes_the_listening_first_annotation_flow(self):
        html = _ui_html().decode("utf-8")
        self.assertIn("Mode punch-in", html)
        self.assertIn("Fin ici + ajouter", html)
        self.assertIn("Espace lecture/pause", html)
        self.assertIn("Dataset du commissaire", html)
        self.assertIn("Exporter le modèle", html)
        self.assertIn("Morceau mono-instrument", html)

    def test_frames_become_editable_role_segments(self):
        frames = [
            AnalysisFrame(0.0, 0.2, {"drummer": 0.8}),
            AnalysisFrame(0.2, 0.4, {"drummer": 0.7}),
            AnalysisFrame(0.6, 0.8, {"drummer": 0.9}),
            AnalysisFrame(1.5, 1.7, {"singer": 0.9}),
        ]
        segments = segments_from_frames(frames)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["labels"], ["drummer"])
        self.assertEqual((segments[0]["start"], segments[0]["end"]), (0.0, 0.8))

    def test_store_replaces_annotation_for_the_same_audio(self):
        store = AnnotationStore(self.make_workspace())
        audio = store.audio_dir / "track.wav"
        with wave.open(str(audio), "wb") as destination:
            destination.setnchannels(2)
            destination.setsampwidth(2)
            destination.setframerate(44_100)
            destination.writeframes(np.zeros((100, 2), dtype="<i2").tobytes())
        document = {
            "audio": "track.wav",
            "duration": 10.0,
            "global_labels": ["drummer"],
            "segments": [{"start": 1.0, "end": 3.0, "labels": ["drummer"]}],
        }
        store.save_annotation(document)
        document["segments"][0]["end"] = 4.0
        store.save_annotation(document)
        manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["clips"]), 1)
        self.assertEqual(manifest["clips"][0]["segments"][0]["end"], 4.0)

    def test_store_persists_an_exclusive_track_label(self):
        store = AnnotationStore(self.make_workspace())
        audio = store.audio_dir / "solo.wav"
        with wave.open(str(audio), "wb") as destination:
            destination.setnchannels(2)
            destination.setsampwidth(2)
            destination.setframerate(44_100)
            destination.writeframes(np.zeros((100, 2), dtype="<i2").tobytes())
        store.save_annotation({
            "audio": "solo.wav",
            "duration": 10.0,
            "global_labels": ["singer", "drummer"],
            "exclusive_label": "bassist",
            "segments": [{"start": 1.0, "end": 3.0, "labels": ["singer"]}],
        })
        manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        clip = manifest["clips"][0]
        self.assertEqual(clip["exclusive_label"], "bassist")
        self.assertEqual(clip["global_labels"], ["bassist"])

    def test_store_keeps_restorable_annotation_history(self):
        store = AnnotationStore(self.make_workspace())
        audio = store.audio_dir / "history.wav"
        with wave.open(str(audio), "wb") as destination:
            destination.setnchannels(2)
            destination.setsampwidth(2)
            destination.setframerate(44_100)
            destination.writeframes(np.zeros((100, 2), dtype="<i2").tobytes())
        document = {
            "audio": "history.wav", "duration": 5.0,
            "global_labels": ["singer"],
            "segments": [{"start": 0.0, "end": 1.0, "labels": ["singer"]}],
        }
        store.save_annotation(document)
        document["segments"][0]["end"] = 2.0
        store.save_annotation(document)
        revision = store.list_history()[0]
        store.restore_history(revision["name"])
        manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["clips"][0]["segments"][0]["end"], 1.0)

    def test_browser_pcm_wav_can_be_read(self):
        path = self.make_workspace() / "audio.wav"
        expected = np.asarray([[0, 1000], [-1000, 0]], dtype="<i2")
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(2)
            destination.setsampwidth(2)
            destination.setframerate(44_100)
            destination.writeframes(expected.tobytes())
        samples, rate = read_pcm_wav(path)
        self.assertEqual(rate, 44_100)
        self.assertEqual(samples.shape, (2, 2))

    def test_analysis_falls_back_when_stemgen_is_unavailable(self):
        path = self.make_workspace() / "tone.wav"
        time = np.arange(44_100, dtype=np.float32) / 44_100
        tone = (np.sin(2 * np.pi * 110 * time) * 6000).astype("<i2")
        stereo = np.column_stack((tone, tone))
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(2)
            destination.setsampwidth(2)
            destination.setframerate(44_100)
            destination.writeframes(stereo.tobytes())
        with patch(
            "desktop_band.annotator.StemgenAnalyzer",
            side_effect=FileNotFoundError,
        ):
            result = analyze_wav(path)
        self.assertEqual(result["duration"], 1.0)
        self.assertIn("analyse légère", result["engine"])
        self.assertIn("segments", result)


if __name__ == "__main__":
    unittest.main()
