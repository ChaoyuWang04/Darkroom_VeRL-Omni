"""Typed records for the spec registry.

Everything here is frozen. A spec is a fact looked up by an exact key, never
mutated in place — when a platform changes its rules you bump `version` and
write a new record. See docs/darkroom-project-design-v0.1.md §2: the spec table
is the one thing the flywheel corrects *without* retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


class SpecError(Exception):
    """Base for every spec-layer failure."""


class SpecNotFound(SpecError):
    """Exact-key lookup missed. Never fall back to a fuzzy match."""


class SpecExpired(SpecError):
    """The record is past `valid_to` — platform rules have almost certainly moved."""


class SpecInvalid(SpecError):
    """The record does not satisfy its own internal constraints."""


@dataclass(frozen=True, slots=True)
class SafeZone:
    """Margins, in pixels, that platform UI may cover. Content must stay clear."""

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    def __post_init__(self) -> None:
        for name in ("top", "bottom", "left", "right"):
            if getattr(self, name) < 0:
                raise SpecInvalid(f"safe_zone.{name} must be >= 0")


@dataclass(frozen=True, slots=True)
class LogoRule:
    anchor: str = "any"
    min_size_pct: float = 0.0
    """Minimum logo extent as a fraction of the canvas's shorter side."""

    ANCHORS = (
        "any",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "top_center",
        "bottom_center",
        "center",
    )

    def __post_init__(self) -> None:
        if self.anchor not in self.ANCHORS:
            raise SpecInvalid(f"unknown logo anchor {self.anchor!r}")
        if not 0.0 <= self.min_size_pct <= 1.0:
            raise SpecInvalid("logo.min_size_pct must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PlacementSpec:
    """One (platform, size) combination: the geometry half of `deliverable`."""

    platform: str
    size: str
    width: int
    height: int
    safe_zone: SafeZone
    text_max_ratio: float
    required_elements: tuple[str, ...]
    logo: LogoRule
    source: str
    version: str
    valid_to: date

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise SpecInvalid("canvas dimensions must be positive")
        if self.size != f"{self.width}x{self.height}":
            raise SpecInvalid(f"size {self.size!r} disagrees with {self.width}x{self.height}")
        if not 0.0 < self.text_max_ratio <= 1.0:
            raise SpecInvalid("text_max_ratio must be in (0, 1]")
        if not self.required_elements:
            raise SpecInvalid("required_elements must not be empty")
        sz = self.safe_zone
        if sz.left + sz.right >= self.width or sz.top + sz.bottom >= self.height:
            raise SpecInvalid("safe zone leaves no usable area")

    @property
    def key(self) -> tuple[str, str]:
        return (self.platform, self.size)

    @property
    def ratio(self) -> float:
        return self.width / self.height

    @property
    def usable_box(self) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) of the region content may occupy.

        Right/bottom are exclusive, matching PIL and most detector conventions.
        This is what L3's geometry checks measure against.
        """
        sz = self.safe_zone
        return (sz.left, sz.top, self.width - sz.right, self.height - sz.bottom)

    @property
    def usable_area(self) -> int:
        left, top, right, bottom = self.usable_box
        return (right - left) * (bottom - top)


@dataclass(frozen=True, slots=True)
class BannedTerm:
    """One entry of the L1 blocklist.

    `evasion` opts the term into separator-tolerant matching (f.u.c.k). It costs
    precision, so it is opt-in per term rather than global — A1.2 requires
    precision >= 0.90 and a blanket evasion pass will not hold that line.
    """

    term: str
    category: str
    evasion: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not self.term.strip():
            raise SpecInvalid("banned term must not be blank")
        if not self.category.strip():
            raise SpecInvalid(f"banned term {self.term!r} has no category")


@dataclass(frozen=True, slots=True)
class ContentPolicy:
    """Region-scoped content rules. Text terms plus visual categories."""

    region: str
    terms: tuple[BannedTerm, ...]
    banned_visual_categories: tuple[str, ...]
    source: str
    version: str
    valid_to: date

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for t in self.terms:
            folded = t.term.casefold()
            if folded in seen:
                raise SpecInvalid(f"duplicate banned term {t.term!r}")
            seen.add(folded)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({t.category for t in self.terms}))


@dataclass(frozen=True, slots=True)
class BrandSpec:
    """Per-advertiser guideline. Deliberately separate from PlacementSpec:
    brand rules travel with the advertiser, placement rules with the platform.
    """

    brand_id: str
    primary_hex: str
    delta_e_max: float
    font_family: str
    logo_asset: str = ""

    def __post_init__(self) -> None:
        h = self.primary_hex
        if not (h.startswith("#") and len(h) == 7):
            raise SpecInvalid(f"primary_hex must look like #RRGGBB, got {h!r}")
        try:
            int(h[1:], 16)
        except ValueError as exc:
            raise SpecInvalid(f"primary_hex {h!r} is not hexadecimal") from exc
        if self.delta_e_max <= 0:
            raise SpecInvalid("delta_e_max must be positive")

    @property
    def primary_rgb(self) -> tuple[int, int, int]:
        h = self.primary_hex[1:]
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
