"""Unit tests for the centralized model.units registry."""

from __future__ import annotations

import unittest

from model.units import (
    CANONICAL_UNITS,
    UNIT_ALIASES,
    is_recognized_unit,
    normalize_unit,
)


class UnitsRegistryTests(unittest.TestCase):
    def test_every_alias_maps_to_a_canonical_unit(self) -> None:
        for alias, canonical in UNIT_ALIASES.items():
            self.assertIn(
                canonical,
                CANONICAL_UNITS,
                msg=f"alias '{alias}' maps to unknown canonical '{canonical}'",
            )

    def test_normalize_resolves_aliases_case_insensitively(self) -> None:
        self.assertEqual(normalize_unit("Celsius"), "C")
        self.assertEqual(normalize_unit("GRAMS"), "g")
        self.assertEqual(normalize_unit("ml"), "mL")
        self.assertEqual(normalize_unit("sec"), "s")

    def test_normalize_preserves_canonical_symbols(self) -> None:
        self.assertEqual(normalize_unit("mL"), "mL")
        self.assertEqual(normalize_unit("C"), "C")

    def test_normalize_leaves_unknown_units_unchanged(self) -> None:
        self.assertEqual(normalize_unit("furlongs"), "furlongs")

    def test_normalized_aliases_are_recognized(self) -> None:
        for alias in UNIT_ALIASES:
            self.assertTrue(is_recognized_unit(normalize_unit(alias)))

    def test_is_recognized_unit(self) -> None:
        self.assertTrue(is_recognized_unit("g"))
        self.assertTrue(is_recognized_unit(""))
        self.assertFalse(is_recognized_unit("furlongs"))


if __name__ == "__main__":
    unittest.main()
