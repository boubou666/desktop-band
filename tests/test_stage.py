import unittest

from desktop_band.analysis import AudioFeatures
from desktop_band.detection import RoleDecision
from desktop_band.stage import StageDirector


class StageDirectorTests(unittest.TestCase):
    def test_variants_are_stable_for_a_seed(self):
        first = StageDirector(1917)
        second = StageDirector(1917)
        self.assertEqual(first.variant_for("bassist"), second.variant_for("bassist"))

    def test_rare_event_requires_active_audio(self):
        director = StageDirector(1)
        director._next_rare = 1.0
        decision = RoleDecision(
            {"drummer": 0.9}, {"drummer": 0.9}, {"drummer": "playing"},
            "drummer", 0.9,
        )
        self.assertIsNone(director.update(AudioFeatures(), decision, 2.0))
        event = director.update(
            AudioFeatures(active=True, level=0.8), decision, 2.0
        )
        self.assertIsNotNone(event)


if __name__ == "__main__":
    unittest.main()
