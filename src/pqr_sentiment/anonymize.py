"""Generic text anonymization engine.

The public dataset in this repository was produced by running the original
confidential records through this engine. Two layers of rules are applied:

1. Term rules: an externally supplied list of literal names or regex
   patterns (companies, contractors, people) mapped to placeholders. The
   real term list is private and never committed; see
   ``data/sensitive_terms.example.json`` for the format.
2. Builtin rules: structural patterns for phone numbers, national identity
   documents, street addresses, vehicle plates and equipment serials.

Placeholders are uppercase Spanish tokens in square brackets, for example
``[TELEFONO]``, so scrubbed text remains readable and the transformer
models treat them as neutral out of vocabulary tokens. Scrubbing is
idempotent: running the engine over already scrubbed text changes nothing.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = ["TermRule", "load_term_rules", "scrub_text", "scrub_series", "BUILTIN_RULES"]


@dataclass(frozen=True)
class TermRule:
    """A sensitive term to replace: a literal name or a regex pattern."""

    pattern: str
    replacement: str
    regex: bool = False


# Literal terms are matched case insensitively and accent insensitively so a
# single entry covers spelling variants like "Andres" and "Andrés".
_ACCENT_CLASSES = {"a": "[aá]", "e": "[eé]", "i": "[ií]", "o": "[oó]", "u": "[uúü]", "n": "[nñ]"}


def _literal_pattern(text: str) -> str:
    parts: list[str] = []
    for char in text:
        base = unicodedata.normalize("NFD", char.lower())[0]
        parts.append(_ACCENT_CLASSES.get(base, re.escape(char)))
    return r"\b" + "".join(parts) + r"\b"


def _compile_term(rule: TermRule) -> re.Pattern[str]:
    if rule.regex:
        return re.compile(rule.pattern, re.IGNORECASE)
    return re.compile(_literal_pattern(rule.pattern), re.IGNORECASE)


_ADDRESS_KEYWORDS = r"(?:cll|cl|cra|cr|kr|dg|tv|av|avenida|calle|carrera|diagonal|transversal|circular)"

BUILTIN_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "address",
        re.compile(
            rf"\b{_ADDRESS_KEYWORDS}s?\.?\s{{0,2}}\d+(?:\s?[a-z]{{1,2}}\b)?"
            r"(?:\s?(?:#|no\.?|n°|nro\.?|num\.?)\s?\d+\s?[a-z]?(?:\s?-\s?\d+)?)?",
            re.IGNORECASE,
        ),
        "[DIRECCION]",
    ),
    (
        "house_number",
        re.compile(r"(?<!\w)#\s?\d+\s?-\s?\d+\b"),
        "[DIRECCION]",
    ),
    (
        "plate",
        # Uppercase plates may carry separators, lowercase ones only match
        # in compact form so ordinary words followed by numbers stay intact.
        re.compile(r"\b(?:[A-Z]{3}\s?[–-]?\s?\d{3}|[a-z]{3}\d{3})\b"),
        "[PLACA]",
    ),
    (
        "serial",
        re.compile(r"\b(?i:serial|serie|c[oó]d(?:igo)?)\.{0,2}\s{0,2}[A-Z0-9]{2,}(?:\s[A-Z0-9]{2,})?\b"),
        "[SERIE]",
    ),
    (
        "mobile_phone",
        re.compile(r"\b3\d{9}\b"),
        "[TELEFONO]",
    ),
    (
        "landline_phone",
        re.compile(r"\b[2-8]\d{6}\b"),
        "[TELEFONO]",
    ),
    (
        "document",
        re.compile(r"\b\d{8,11}\b"),
        "[DOCUMENTO]",
    ),
]


def load_term_rules(path: str | Path) -> list[TermRule]:
    """Read term rules from a JSON file with a top level ``terms`` list."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "terms" not in payload or not isinstance(payload["terms"], list):
        raise ValueError("Term file must be a JSON object with a 'terms' list")
    rules = []
    for entry in payload["terms"]:
        rules.append(
            TermRule(
                pattern=entry["pattern"],
                replacement=entry["replacement"],
                regex=bool(entry.get("regex", False)),
            )
        )
    return rules


def scrub_text(text: str, term_rules: list[TermRule]) -> tuple[str, Counter]:
    """Scrub one text. Returns the clean text and a counter of hits per rule."""
    hits: Counter = Counter()
    clean = text
    for rule in term_rules:
        clean, n = _compile_term(rule).subn(rule.replacement, clean)
        if n:
            hits[f"term:{rule.pattern}"] += n
    for name, pattern, replacement in BUILTIN_RULES:
        clean, n = pattern.subn(replacement, clean)
        if n:
            hits[name] += n
    return clean, hits


def scrub_series(series: pd.Series, term_rules: list[TermRule]) -> tuple[pd.Series, Counter]:
    """Scrub a pandas Series of texts, passing null values through untouched."""
    totals: Counter = Counter()
    values = []
    for value in series:
        if pd.isna(value):
            values.append(value)
            continue
        clean, hits = scrub_text(str(value), term_rules)
        totals.update(hits)
        values.append(clean)
    return pd.Series(values, index=series.index, name=series.name), totals
