"""L1 · compliance gate.

This layer is a **gate, not a score** (design §11②). A hit does not subtract
points; it zeroes the reward and turns the request into a `reject`. Compliance
incidents and merely-ugly output are not the same kind of error, and averaging
them together is how you end up shipping the first one.

Two properties the acceptance criteria pin down, and why each is hard:

* **A1.1 — recall >= 0.99.** Missing a violation is an incident, so matching
  tolerates obfuscation: NFKC folding, zero-width stripping, and opt-in
  separator-tolerant matching for terms that get spelled around.

* **A1.2 — precision >= 0.90.** Recall is easy to buy with substring matching,
  and it bankrupts precision: `ass` inside `assassin`, `hell` inside `shell`.
  So matching splits by script — word-boundary for Latin, substring for CJK,
  which has no word delimiters — and separator tolerance is opt-in per term
  rather than applied across the board.

The image half of the gate is deliberately *not* silently absent. A verdict
records whether the visual check actually ran; `ComplianceVerdict.passed` only
speaks for the checks that were performed. Callers that need the full gate must
consult `gate_complete`. A gate that reports green because half of it was never
wired is the exact failure mode this project exists to avoid.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol, runtime_checkable

from ..spec.models import BannedTerm, ContentPolicy

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)

# Ranges where text has no word delimiters, so \b is meaningless and substring
# matching is the correct (and safe) strategy.
_SCRIPTLESS_RANGES = (
    (0x3040, 0x30FF),  # kana
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK unified
    (0xAC00, 0xD7AF),  # hangul
    (0xF900, 0xFAFF),  # CJK compatibility
)

# Bounded separator run allowed between characters of an evasion-checked term.
# Unbounded would let a term match across an entire paragraph.
_SEP = r"[\W_]{0,3}"


def normalize(text: str) -> str:
    """Fold the cheap obfuscations away without destroying token structure.

    NFKC collapses full-width and other compatibility forms; zero-width
    characters are dropped outright. Whitespace and punctuation are *kept*,
    because word boundaries are what protect precision.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    return folded.casefold()


def _is_scriptless(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _SCRIPTLESS_RANGES)


def _needs_substring_match(term: str) -> bool:
    return any(_is_scriptless(ch) for ch in term)


def _compile(term: BannedTerm) -> re.Pattern[str]:
    folded = normalize(term.term)
    substring = _needs_substring_match(folded)

    if term.evasion:
        chars = [re.escape(c) for c in folded if not c.isspace()]
        body = _SEP.join(chars)
        # Even under evasion matching, a Latin term must not match inside a
        # longer alphanumeric run — otherwise precision collapses.
        if not substring:
            body = rf"(?<![0-9a-z]){body}(?![0-9a-z])"
        return re.compile(body)

    body = re.escape(folded)
    if substring:
        return re.compile(body)
    return re.compile(rf"(?<![0-9a-z]){body}(?![0-9a-z])")


@dataclass(frozen=True, slots=True)
class ComplianceHit:
    term: str
    category: str
    field: str
    span: tuple[int, int]
    """Offsets into the *normalized* text, not the raw input."""


@dataclass(frozen=True, slots=True)
class ComplianceVerdict:
    passed: bool
    """True when every check that ran found nothing. Says nothing about checks
    that did not run — see `gate_complete`."""

    hits: tuple[ComplianceHit, ...] = ()
    policy_version: str = ""
    text_checked: bool = True
    image_checked: bool = False

    @property
    def gate_complete(self) -> bool:
        """Whether both halves of the gate actually ran."""
        return self.text_checked and self.image_checked

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({h.category for h in self.hits}))

    def explain(self) -> str:
        if not self.hits:
            state = "clean" if self.gate_complete else "clean (text only — image check did not run)"
            return f"compliance: {state}"
        parts = ", ".join(f"{h.category}:{h.term!r} in {h.field}" for h in self.hits)
        return f"compliance: BLOCKED [{parts}]"


@runtime_checkable
class ImageComplianceChecker(Protocol):
    """Visual half of the gate. Implemented in a later task; the protocol exists
    now so the absence is explicit rather than silent."""

    def check(self, image_path: str, categories: Iterable[str]) -> tuple[ComplianceHit, ...]:
        ...


@dataclass
class ComplianceGate:
    policy: ContentPolicy
    image_checker: ImageComplianceChecker | None = None
    _patterns: list[tuple[BannedTerm, re.Pattern[str]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._patterns = [(t, _compile(t)) for t in self.policy.terms]

    def check_text(self, fields: Mapping[str, str]) -> tuple[ComplianceHit, ...]:
        """Scan named copy fields. Field names are kept so the verdict can say
        *where* the violation is, which is what makes a rejection actionable."""
        hits: list[ComplianceHit] = []
        for name, raw in fields.items():
            if not raw:
                continue
            text = normalize(raw)
            for term, pattern in self._patterns:
                for m in pattern.finditer(text):
                    hits.append(
                        ComplianceHit(
                            term=term.term,
                            category=term.category,
                            field=name,
                            span=m.span(),
                        )
                    )
        return tuple(hits)

    def check(
        self,
        fields: Mapping[str, str] | None = None,
        image_path: str | None = None,
    ) -> ComplianceVerdict:
        hits: list[ComplianceHit] = []

        text_checked = fields is not None
        if fields is not None:
            hits.extend(self.check_text(fields))

        image_checked = False
        if image_path is not None and self.image_checker is not None:
            hits.extend(
                self.image_checker.check(image_path, self.policy.banned_visual_categories)
            )
            image_checked = True

        return ComplianceVerdict(
            passed=not hits,
            hits=tuple(hits),
            policy_version=self.policy.version,
            text_checked=text_checked,
            image_checked=image_checked,
        )
