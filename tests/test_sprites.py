from pathlib import Path
import unittest

from desktop_band.sprites import load_sprite_pack


class SpritePackTests(unittest.TestCase):
    def test_external_manifest_is_loaded(self):
        root = Path(__file__).parent / "fixtures" / "sprite_pack"
        sheets = load_sprite_pack(str(root))
        self.assertEqual(len(sheets), 1)
        self.assertEqual(sheets[0].role_rows, {"bassist": 0})

    def test_manifest_rejects_unknown_role(self):
        root = Path(__file__).parent / "fixtures" / "invalid_sprite_pack"
        with self.assertRaises(ValueError):
            load_sprite_pack(str(root))


if __name__ == "__main__":
    unittest.main()
