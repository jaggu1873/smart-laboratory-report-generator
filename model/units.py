"""Central registry of supported laboratory measurement units.

Single source of truth for the unit vocabulary used across the pipeline:

    * :data:`CANONICAL_UNITS` — the unit symbols the pipeline recognizes.
    * :data:`UNIT_ALIASES` — accepted synonyms that normalize to a canonical
      symbol.

Both :mod:`model.preprocessing` (which normalizes raw units) and
:mod:`model.validator` (which validates them) consume this registry, so the
normalization and validation stages can never drift out of sync.
"""

from __future__ import annotations

#: Canonical unit symbols recognized by the pipeline. The empty string denotes
#: a dimensionless / unit-less measurement (e.g. a plain count) and is
#: intentionally allowed.
CANONICAL_UNITS: frozenset[str] = frozenset(
    {"g", "kg", "mg", "mL", "L", "C", "K", "s", "min", "h", "mol", "M", "%", ""}
)

#: Accepted aliases/synonyms mapping a lower-cased input to its canonical
#: symbol. Canonical symbols are matched case-sensitively (so units where case
#: is significant, such as ``mL``, are preserved) and do not need an entry
#: here.
UNIT_ALIASES: dict[str, str] = {
    "celsius": "C",
    "degc": "C",
    "°c": "C",
    "grams": "g",
    "gram": "g",
    "milliliters": "mL",
    "milliliter": "mL",
    "ml": "mL",
    "seconds": "s",
    "second": "s",
    "sec": "s",
}


def normalize_unit(text: str) -> str:
    """Map a cleaned unit string to its canonical symbol.

    ``text`` is expected to be whitespace-trimmed already. Known aliases are
    resolved case-insensitively; any other value is returned unchanged so the
    validator can flag genuinely unrecognized units.
    """
    return UNIT_ALIASES.get(text.lower(), text)


def is_recognized_unit(unit: str) -> bool:
    """Return ``True`` if ``unit`` is a canonical unit symbol."""
    return unit in CANONICAL_UNITS
