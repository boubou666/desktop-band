import unittest

from desktop_band.renderer import (
    _role_frame_rate,
    _smooth_activity,
)
from desktop_band.sprites import load_sprite_pack


class VisualEnvelopeTests(unittest.TestCase):
    def test_roles_receive_distinct_tempo_synced_frame_rates(self):
        rates = {
            role: _role_frame_rate(role, 120.0, 0.7, 8)
            for role in (
                "keyboard",
                "bassist",
                "vibing",
                "guitarist",
                "drummer",
                "percussion",
            )
        }
        self.assertLess(rates["keyboard"], rates["bassist"])
        self.assertLess(rates["bassist"], rates["guitarist"])
        self.assertLess(rates["guitarist"], rates["drummer"])
        self.assertLess(rates["drummer"], rates["percussion"])

    def test_singer_speed_uses_vocal_activity_instead_of_bpm(self):
        quiet = _role_frame_rate("singer", 70.0, 0.2, 8)
        loud = _role_frame_rate("singer", 170.0, 0.9, 8)
        self.assertAlmostEqual(quiet, 3.5)
        self.assertAlmostEqual(loud, 5.25)

    def test_instrument_speed_follows_its_activity(self):
        quiet = _role_frame_rate("bassist", 145.0, 0.15, 8)
        loud = _role_frame_rate("bassist", 145.0, 0.95, 8)
        self.assertGreater(loud, quiet * 1.2)

    def test_percussive_onset_temporarily_boosts_motion(self):
        regular = _role_frame_rate("drummer", 120.0, 0.5, 8, onset=0.0)
        accented = _role_frame_rate("drummer", 120.0, 0.5, 8, onset=1.0)
        self.assertGreater(accented, regular)
        self.assertLessEqual(accented, 10.0)

    def test_slow_reported_bpm_still_has_readable_animation(self):
        half_time = _role_frame_rate("bassist", 72.5, 0.7, 8)
        self.assertGreater(half_time, 3.0)

    def test_genuine_fast_tempo_gets_an_extra_energy_boost(self):
        slow_double_time = _role_frame_rate("bassist", 75.0, 0.7, 8)
        genuine_fast = _role_frame_rate("bassist", 160.0, 0.7, 8)
        self.assertGreater(genuine_fast, slow_double_time * 1.4)

    def test_fast_percussion_respects_its_role_cap(self):
        rate = _role_frame_rate("percussion", 180.0, 1.0, 8, onset=1.0)
        self.assertEqual(rate, 14.0)

    def test_every_role_has_at_least_eight_sprite_frames(self):
        frame_counts = {}
        for sheet in load_sprite_pack("gopnik"):
            for role in sheet.role_rows:
                frame_counts[role] = frame_counts.get(role, 0) + 4
        self.assertEqual(
            set(frame_counts),
            {
                "bassist",
                "singer",
                "drummer",
                "keyboard",
                "guitarist",
                "percussion",
                "vibing",
            },
        )
        self.assertTrue(all(count >= 8 for count in frame_counts.values()))

    def test_singer_style_attack_does_not_jump_to_target(self):
        value = _smooth_activity(0.0, 1.0, 1.0 / 30.0, 0.24, 0.34)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 0.2)

    def test_release_is_gradual(self):
        value = _smooth_activity(1.0, 0.0, 1.0 / 30.0, 0.24, 0.34)
        self.assertGreater(value, 0.85)
        self.assertLess(value, 1.0)

    def test_envelope_is_nearly_frame_rate_independent(self):
        one_step = _smooth_activity(0.0, 1.0, 0.1, 0.24, 0.34)
        three_steps = 0.0
        for _ in range(3):
            three_steps = _smooth_activity(
                three_steps, 1.0, 0.1 / 3.0, 0.24, 0.34
            )
        self.assertAlmostEqual(one_step, three_steps, places=8)


if __name__ == "__main__":
    unittest.main()
