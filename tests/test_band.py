import unittest

from desktop_band.analysis import AudioFeatures
from desktop_band.band import (
    create_custom_lineup,
    create_lineup,
    gated_activity,
    role_activity,
)


class BandTests(unittest.TestCase):
    def test_lineups_have_requested_size(self):
        for count in range(1, 8):
            with self.subTest(count=count):
                self.assertEqual(len(create_lineup(count)), count)

    def test_invalid_lineup_is_rejected(self):
        with self.assertRaises(ValueError):
            create_lineup(0)

    def test_custom_lineup_keeps_stage_order(self):
        lineup = create_custom_lineup(("keyboard", "vibing", "singer"))
        self.assertEqual(
            tuple(musician.role for musician in lineup),
            ("vibing", "singer", "keyboard"),
        )

    def test_custom_lineup_cannot_be_empty(self):
        with self.assertRaises(ValueError):
            create_custom_lineup(())

    def test_bass_responds_more_to_low_energy(self):
        low = AudioFeatures(
            active=True,
            level=0.5,
            low=1.0,
            mid=0.1,
            harmonic=0.8,
            harmonic_low=1.0,
        )
        mid = AudioFeatures(
            active=True,
            level=0.5,
            low=0.1,
            mid=1.0,
            harmonic=0.8,
            harmonic_low=0.1,
            harmonic_mid=1.0,
        )
        self.assertGreater(role_activity("bassist", low), role_activity("bassist", mid))

    def test_taiko_like_event_favors_drums(self):
        taiko = AudioFeatures(
            active=True,
            level=0.9,
            low=0.95,
            mid=0.7,
            high=0.45,
            onset=1.0,
            percussive=1.0,
            harmonic=0.03,
        )
        drums = role_activity("drummer", taiko)
        self.assertGreater(drums, role_activity("bassist", taiko) * 10)
        self.assertGreater(drums, role_activity("keyboard", taiko) * 10)
        self.assertEqual(role_activity("singer", taiko), 0.0)

    def test_dominant_taiko_tail_hard_mutes_melodic_leakage(self):
        tail = AudioFeatures(
            active=True,
            level=0.7,
            low=0.6,
            mid=0.5,
            percussive=0.3,
            percussive_memory=0.72,
            harmonic=0.28,
            harmonic_low=0.35,
            harmonic_mid=0.5,
            vocal=0.45,
        )
        self.assertEqual(role_activity("singer", tail), 0.0)
        self.assertEqual(role_activity("bassist", tail), 0.0)
        self.assertEqual(role_activity("keyboard", tail), 0.0)

    def test_strong_harmonic_component_survives_drum_memory(self):
        mixed_music = AudioFeatures(
            active=True,
            level=0.8,
            percussive=0.55,
            percussive_memory=0.7,
            harmonic=0.62,
            harmonic_mid=0.75,
        )
        self.assertGreater(role_activity("keyboard", mixed_music), 0.4)

    def test_weak_role_score_does_not_trigger_animation(self):
        activity, playing = gated_activity("keyboard", 0.06, False)
        self.assertEqual(activity, 0.0)
        self.assertFalse(playing)

    def test_hysteresis_prevents_threshold_flicker(self):
        activity, playing = gated_activity("keyboard", 0.12, False)
        self.assertGreater(activity, 0.0)
        self.assertTrue(playing)
        activity, playing = gated_activity("keyboard", 0.08, True)
        self.assertGreater(activity, 0.0)
        self.assertTrue(playing)
        activity, playing = gated_activity("keyboard", 0.05, True)
        self.assertEqual(activity, 0.0)
        self.assertFalse(playing)

    def test_stemgen_semantic_stems_drive_only_their_roles(self):
        drums = AudioFeatures(
            active=True,
            separated=True,
            stem_drums=0.9,
            stem_bass=0.0,
            stem_vocals=0.0,
            stem_other=0.0,
        )
        self.assertEqual(role_activity("drummer", drums), 0.9)
        self.assertEqual(role_activity("singer", drums), 0.0)
        self.assertEqual(role_activity("bassist", drums), 0.0)
        self.assertEqual(role_activity("keyboard", drums), 0.0)
        self.assertEqual(role_activity("guitarist", drums), 0.0)

    def test_stemgen_other_uses_timbre_for_guitar_and_keyboard(self):
        mid_heavy = AudioFeatures(
            active=True,
            separated=True,
            stem_other=0.8,
            stem_other_mid=0.9,
            stem_other_high=0.1,
        )
        self.assertGreater(
            role_activity("guitarist", mid_heavy),
            role_activity("keyboard", mid_heavy),
        )

    def test_vibing_character_reacts_to_any_active_sound(self):
        features = AudioFeatures(active=True, level=0.65)
        self.assertEqual(role_activity("vibing", features), 0.65)


if __name__ == "__main__":
    unittest.main()
