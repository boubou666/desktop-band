import unittest

import numpy as np

from desktop_band.analysis import AudioAnalyzer, _estimate_bpm_from_envelope


class AudioAnalyzerTests(unittest.TestCase):
    def test_tempo_autocorrelation_finds_fast_regular_kicks(self):
        timestamps = np.arange(0.0, 10.0, 0.02)
        period = 60.0 / 150.0
        phase = np.mod(timestamps, period)
        envelope = np.exp(-((phase / 0.045) ** 2))
        estimate = _estimate_bpm_from_envelope(timestamps, envelope)
        self.assertIsNotNone(estimate)
        bpm, confidence = estimate
        self.assertAlmostEqual(bpm, 150.0, delta=3.0)
        self.assertGreater(confidence, 0.5)

    def test_tempo_autocorrelation_rejects_flat_signal(self):
        timestamps = np.arange(0.0, 10.0, 0.02)
        estimate = _estimate_bpm_from_envelope(
            timestamps, np.full(timestamps.shape, 0.2)
        )
        self.assertIsNone(estimate)

    def test_slow_tempo_is_not_doubled_in_reported_bpm(self):
        timestamps = np.arange(0.0, 12.0, 0.02)
        period = 60.0 / 72.5
        phase = np.mod(timestamps, period)
        envelope = np.exp(-((phase / 0.05) ** 2))
        estimate = _estimate_bpm_from_envelope(timestamps, envelope)
        self.assertIsNotNone(estimate)
        bpm, _confidence = estimate
        self.assertAlmostEqual(bpm, 72.5, delta=3.0)

    def test_silence_is_inactive(self):
        result = AudioAnalyzer().process(np.zeros(1024), 48_000, now=0.0)
        self.assertFalse(result.active)
        self.assertEqual(result.level, 0.0)

    def test_tiny_loopback_residue_is_inactive(self):
        residue = np.full(1024, 0.001, dtype=np.float32)
        result = AudioAnalyzer().process(residue, 48_000, now=0.0)
        self.assertFalse(result.active)
        self.assertEqual(result.level, 0.0)

    def test_low_tone_prefers_low_band(self):
        rate = 48_000
        samples = np.arange(4096) / rate
        tone = (0.2 * np.sin(2 * np.pi * 90 * samples)).astype(np.float32)
        result = AudioAnalyzer().process(tone, rate, now=0.0)
        self.assertTrue(result.active)
        self.assertGreater(result.low, result.mid)
        self.assertGreater(result.low, result.high)

    def test_high_tone_prefers_high_band(self):
        rate = 48_000
        samples = np.arange(4096) / rate
        tone = (0.2 * np.sin(2 * np.pi * 5000 * samples)).astype(np.float32)
        result = AudioAnalyzer().process(tone, rate, now=0.0)
        self.assertGreater(result.high, result.low)
        self.assertGreater(result.high, result.mid)

    def test_drum_like_attack_is_classified_as_percussive(self):
        rate = 48_000
        analyzer = AudioAnalyzer()
        analyzer.process(np.zeros(1024, dtype=np.float32), rate, now=0.0)
        samples = np.arange(1024) / rate
        envelope = np.exp(-np.arange(1024) / 120.0)
        hit = (0.8 * envelope * np.sin(2 * np.pi * 90 * samples)).astype(np.float32)
        result = analyzer.process(hit, rate, now=0.02)
        self.assertGreater(result.percussive, 0.7)
        self.assertGreater(result.percussive, result.harmonic)

    def test_long_drum_tail_keeps_percussive_identity(self):
        rate = 48_000
        analyzer = AudioAnalyzer()
        analyzer.process(np.zeros(1024, dtype=np.float32), rate, now=0.0)
        samples = np.arange(1024) / rate
        hit = (0.8 * np.sin(2 * np.pi * 90 * samples)).astype(np.float32)
        analyzer.process(hit, rate, now=0.02)

        result = None
        for index in range(40):
            amplitude = 0.25 * (0.97 ** index)
            tail = (amplitude * np.sin(2 * np.pi * 90 * samples)).astype(np.float32)
            result = analyzer.process(tail, rate, now=0.04 + index * 1024 / rate)

        self.assertIsNotNone(result)
        self.assertGreater(result.percussive_memory, result.harmonic)

    def test_sustained_tone_moves_to_harmonic_component(self):
        rate = 48_000
        analyzer = AudioAnalyzer()
        samples = np.arange(1024) / rate
        tone = (0.2 * np.sin(2 * np.pi * 440 * samples)).astype(np.float32)
        result = None
        for index in range(24):
            result = analyzer.process(tone, rate, now=index * 1024 / rate)
        self.assertIsNotNone(result)
        self.assertGreater(result.harmonic_mid, result.percussive_mid)
        self.assertGreater(result.harmonic, result.percussive)


if __name__ == "__main__":
    unittest.main()
