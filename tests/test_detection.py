import unittest

from desktop_band.analysis import AudioFeatures
from desktop_band.detection import PROFILES, RoleRouter, role_confidences


class CalibratedDetectionTests(unittest.TestCase):
    def test_taiko_profile_routes_resonant_hits_to_drums(self):
        features = AudioFeatures(
            active=True, separated=True, level=0.9, onset=0.8,
            stem_drums=0.9, stem_bass=0.2, stem_vocals=0.25, stem_other=0.35,
            percussive_low=0.9, percussive_mid=0.2, percussive_high=0.05,
        )
        scores = role_confidences(features, PROFILES["taiko"])
        self.assertGreater(scores["drummer"], 0.9)
        self.assertLess(scores["singer"], PROFILES["taiko"].play_threshold)
        self.assertLess(scores["bassist"], PROFILES["taiko"].play_threshold)
        self.assertLess(scores["percussion"], PROFILES["taiko"].play_threshold)

    def test_hardbass_profile_retains_bass_and_kick(self):
        features = AudioFeatures(
            active=True, separated=True, level=0.9, onset=0.7,
            stem_drums=0.8, stem_bass=0.75, stem_vocals=0.05, stem_other=0.25,
            percussive_low=0.75, percussive_mid=0.3, percussive_high=0.25,
        )
        scores = role_confidences(features, PROFILES["hardbass"])
        self.assertGreater(scores["bassist"], 0.8)
        self.assertGreater(scores["drummer"], 0.7)
        self.assertLess(scores["keyboard"], 0.1)

    def test_mid_dense_other_favors_guitar(self):
        features = AudioFeatures(
            active=True, separated=True, level=0.7, stem_other=0.8,
            stem_other_mid=0.95, stem_other_high=0.08, stem_other_low=0.15,
            stem_other_note_density=0.8, stem_other_onset=0.4,
            stem_other_flatness=0.3,
        )
        scores = role_confidences(features, PROFILES["balanced"])
        self.assertGreater(scores["guitarist"], scores["keyboard"] * 10)

    def test_stable_broad_other_favors_keyboard(self):
        features = AudioFeatures(
            active=True, separated=True, level=0.7, stem_other=0.8,
            stem_other_mid=0.45, stem_other_high=0.75, stem_other_low=0.55,
            stem_other_pitch_stability=0.9, stem_other_harmonic=0.8,
        )
        scores = role_confidences(features, PROFILES["balanced"])
        self.assertGreater(scores["keyboard"], scores["guitarist"] * 10)

    def test_router_accumulates_evidence_before_full_playing(self):
        router = RoleRouter()
        features = AudioFeatures(active=True, separated=True, stem_vocals=0.9, level=0.8)
        first = router.update(features, ("singer",), 0.02)
        self.assertNotEqual(first.states["singer"], "playing")
        decision = first
        for _ in range(20):
            decision = router.update(features, ("singer",), 0.03)
        self.assertEqual(decision.states["singer"], "playing")
        self.assertEqual(decision.dominant_role, "singer")


if __name__ == "__main__":
    unittest.main()
