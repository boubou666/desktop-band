import unittest

from desktop_band.analysis import AudioFeatures
from desktop_band.reactions import ShoutDetector


class ShoutDetectorTests(unittest.TestCase):
    def test_strong_drum_impact_triggers_blyat(self):
        detector = ShoutDetector()
        features = AudioFeatures(
            active=True,
            level=0.8,
            onset=0.95,
            separated=True,
            stem_drums=0.9,
        )
        event = detector.update(features, {"drummer": 0.9, "singer": 0.0}, 1.0)
        self.assertIsNotNone(event)
        self.assertEqual(event.text, "BLYAT!")
        self.assertEqual(event.role, "drummer")

    def test_cooldown_blocks_repeated_impacts(self):
        detector = ShoutDetector(cooldown=3.6)
        features = AudioFeatures(
            active=True,
            level=0.8,
            onset=1.0,
            separated=True,
            stem_drums=1.0,
        )
        scores = {"drummer": 1.0}
        self.assertIsNotNone(detector.update(features, scores, 1.0))
        self.assertIsNone(detector.update(features, scores, 2.0))
        self.assertIsNotNone(detector.update(features, scores, 5.0))

    def test_restart_after_real_silence_triggers_poehali(self):
        detector = ShoutDetector()
        quiet_music = AudioFeatures(active=True, level=0.4)
        silence = AudioFeatures(active=False)
        scores = {"vibing": 0.4}
        self.assertIsNone(detector.update(quiet_music, scores, 0.0))
        self.assertIsNone(detector.update(silence, scores, 1.0))
        event = detector.update(quiet_music, scores, 2.2)
        self.assertIsNotNone(event)
        self.assertEqual(event.text, "POEHALI!")


if __name__ == "__main__":
    unittest.main()
