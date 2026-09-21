import unittest

from desktop_band.capture import _device_key, _loopback_for_speaker


class _Device:
    def __init__(self, name, identifier=None, isloopback=False):
        self.name = name
        self.id = identifier
        self.isloopback = isloopback


class _SoundCard:
    def __init__(self, direct=None, candidates=()):
        self.direct = direct
        self.candidates = tuple(candidates)

    def get_microphone(self, _name, include_loopback=False):
        return self.direct

    def all_microphones(self, include_loopback=False):
        return self.candidates


class DynamicOutputTests(unittest.TestCase):
    def test_device_identity_prefers_stable_id(self):
        self.assertEqual(_device_key(_Device("Headset", "wasapi-42")), "wasapi-42")

    def test_matching_loopback_is_preferred_from_fallbacks(self):
        hdmi = _Device("HDMI Monitor loopback", isloopback=True)
        headset = _Device("USB Headset loopback", isloopback=True)
        sc = _SoundCard(candidates=(hdmi, headset))
        selected = _loopback_for_speaker(sc, _Device("USB Headset"))
        self.assertIs(selected, headset)


if __name__ == "__main__":
    unittest.main()
