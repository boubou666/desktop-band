import unittest

from desktop_band.analysis import AudioFeatures
from desktop_band.classifier import InstrumentClassifier, feature_vector


class InstrumentClassifierTests(unittest.TestCase):
    def test_feature_vector_has_stable_shape(self):
        self.assertEqual(feature_vector(AudioFeatures()).shape, (1, 23))

    def test_shipped_model_recognizes_six_specialized_roles(self):
        classifier = InstrumentClassifier()
        cases = {
            "singer": AudioFeatures(
                level=0.8, mid=0.8, vocal=0.9, harmonic=0.8,
                stem_vocals=0.95,
            ),
            "drummer": AudioFeatures(
                stem_drums=0.9, percussive_low=0.9,
                percussive_mid=0.3, percussive_high=0.2,
            ),
            "percussion": AudioFeatures(
                stem_drums=0.85, percussive_low=0.15,
                percussive_mid=0.8, percussive_high=0.9,
            ),
            "bassist": AudioFeatures(
                level=0.8, low=0.95, harmonic=0.9, stem_bass=0.95,
            ),
            "guitarist": AudioFeatures(
                stem_other=0.9, stem_other_mid=0.9,
                stem_other_note_density=0.8, stem_other_onset=0.6,
                stem_other_flatness=0.35,
            ),
            "keyboard": AudioFeatures(
                stem_other=0.9, stem_other_low=0.5, stem_other_mid=0.5,
                stem_other_high=0.75, stem_other_harmonic=0.9,
                stem_other_pitch_stability=0.9,
            ),
        }
        for expected, features in cases.items():
            with self.subTest(expected=expected):
                scores = classifier.predict(features)
                # This is a multi-label classifier: closely related roles may
                # legitimately co-activate here. The runtime router applies
                # stem gates and pairwise competition afterwards.
                self.assertGreater(scores[expected], 0.5)


if __name__ == "__main__":
    unittest.main()
