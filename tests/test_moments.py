import unittest

from desktop_band.analysis import AudioFeatures
from desktop_band.moments import MusicalMomentDetector


class MusicalMomentTests(unittest.TestCase):
    def test_energy_jump_with_multiple_roles_triggers_drop(self):
        detector = MusicalMomentDetector()
        quiet = AudioFeatures(active=True, level=0.35, stem_drums=0.15)
        scores = {"drummer": 0.1, "bassist": 0.1, "guitarist": 0.1}
        for index in range(60):
            self.assertIsNone(detector.update(quiet, scores, index * 0.05))
        drop = AudioFeatures(active=True, level=0.95, onset=0.9, stem_drums=0.85)
        event = detector.update(
            drop,
            {"drummer": 0.9, "bassist": 0.8, "guitarist": 0.7},
            3.05,
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.kind, "drop")
        self.assertTrue(detector.is_active(3.5))

    def test_loud_but_steady_section_is_not_a_drop(self):
        detector = MusicalMomentDetector()
        loud = AudioFeatures(active=True, level=0.86, onset=0.5, stem_drums=0.6)
        scores = {"drummer": 0.8, "bassist": 0.7, "guitarist": 0.6}
        event = None
        for index in range(80):
            event = detector.update(loud, scores, index * 0.05)
        self.assertIsNone(event)


if __name__ == "__main__":
    unittest.main()
