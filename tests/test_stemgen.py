import unittest

from desktop_band.analysis import AudioFeatures
from desktop_band.stemgen import (
    _refine_model_stems,
    _rescue_percussive_other,
    _stem_activity_target,
    _stabilize_vocal_activity,
)


class StemgenRoutingTests(unittest.TestCase):
    def test_quiet_bass_has_a_more_sensitive_gate_than_other_stems(self):
        loudest = 0.05
        quiet = loudest * (10 ** (-18.0 / 20.0))
        bass, bass_db = _stem_activity_target("bass", quiet, loudest, 0.8)
        vocals, vocal_db = _stem_activity_target("vocals", quiet, loudest, 0.8)
        self.assertAlmostEqual(bass_db, -18.0, places=3)
        self.assertAlmostEqual(vocal_db, -18.0, places=3)
        self.assertGreater(bass, 0.2)
        self.assertEqual(vocals, 0.0)

    def test_isolated_low_percussion_in_other_is_rerouted_to_drums(self):
        base = AudioFeatures(
            active=True,
            low=0.7,
            mid=0.6,
            high=0.08,
            percussive=0.6,
            percussive_memory=0.75,
        )
        result = _rescue_percussive_other(
            {"drums": 0.0, "bass": 0.0, "vocals": 0.2, "other": 0.8},
            base,
        )
        self.assertGreater(result["drums"], result["other"])
        self.assertLess(result["vocals"], 0.05)

    def test_bright_other_in_full_mix_is_left_alone(self):
        base = AudioFeatures(
            active=True,
            low=0.2,
            high=0.8,
            percussive=0.5,
            percussive_memory=0.5,
        )
        original = {"drums": 0.6, "bass": 0.4, "vocals": 0.5, "other": 0.7}
        self.assertEqual(_rescue_percussive_other(original, base), original)

    def test_percussive_vocal_leak_is_moved_to_drums(self):
        vocal_hit = AudioFeatures(
            active=True,
            low=0.8,
            high=0.08,
            percussive=0.7,
            percussive_memory=0.8,
            harmonic=0.1,
        )
        quiet_other = AudioFeatures()
        result = _refine_model_stems(
            {"drums": 0.0, "bass": 0.0, "vocals": 0.8, "other": 0.1},
            vocal_hit,
            quiet_other,
        )
        self.assertGreater(result["drums"], 0.5)
        # Spectral shape alone does not mute it: the temporal gate below is
        # responsible for rejecting a short false-vocal burst.
        presence = 0.0
        score = result["vocals"]
        for _ in range(5):
            presence, gated = _stabilize_vocal_activity(presence, score)
        self.assertLess(gated, 0.12)

    def test_sustained_harmonic_vocal_is_preserved(self):
        voice = AudioFeatures(
            active=True,
            low=0.2,
            mid=0.8,
            percussive=0.05,
            percussive_memory=0.08,
            harmonic=0.9,
        )
        result = _refine_model_stems(
            {"drums": 0.1, "bass": 0.1, "vocals": 0.8, "other": 0.2},
            voice,
            AudioFeatures(),
        )
        self.assertGreater(result["vocals"], 0.75)
        self.assertEqual(result["drums"], 0.1)

    def test_short_vocal_burst_does_not_pass_temporal_gate(self):
        presence = 0.0
        outputs = []
        for _ in range(5):
            presence, score = _stabilize_vocal_activity(presence, 0.8)
            outputs.append(score)
        self.assertLess(max(outputs), 0.12)

    def test_sustained_vocal_opens_temporal_gate(self):
        presence = 0.0
        score = 0.0
        for _ in range(12):
            presence, score = _stabilize_vocal_activity(presence, 0.8)
        self.assertGreater(score, 0.6)


if __name__ == "__main__":
    unittest.main()
