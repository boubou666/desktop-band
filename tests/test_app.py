import unittest

from desktop_band.app import _window_size


class WindowSizeTests(unittest.TestCase):
    def test_seven_member_compact_layout_is_much_smaller(self):
        compact_width, compact_height = _window_size(7, "compact")
        wide_width, wide_height = _window_size(7, "wide")
        self.assertEqual((compact_width, compact_height), (824, 235))
        self.assertLess(compact_width * compact_height, wide_width * wide_height / 2)

    def test_small_lineup_keeps_a_usable_minimum_width(self):
        self.assertEqual(_window_size(1, "compact"), (280, 235))


if __name__ == "__main__":
    unittest.main()
