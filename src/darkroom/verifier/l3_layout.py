"""L3 · layout geometry.

The only layer of the verifier with no model in it. Everything here is integer
arithmetic over boxes, which is why its acceptance bar is 1.00 (A1.6): the other
layers have recognition error, this one has only bugs.

**Pure function over boxes.** Where the boxes come from — OCR, an object
detector, or the programmatic renderer that placed them and therefore knows
their coordinates exactly — is deliberately not this module's problem. That
decoupling is what lets T3.1's renderer check its own output (A3.1), and it lets
this logic be tested exhaustively without an image existing.

**Hard vs soft.** Not every rule violation means rework. A wrong canvas size is
rejected outright and text buried under platform UI comes straight back from a
reviewer — those are hard. Exceeding the text-area guidance does neither: Meta
retired the hard 20% rule in 2020, and heavy text now costs delivery, not
approval. Folding a soft violation into `deliverable` would fail creatives the
platform would happily run.

Three arithmetic traps this module is written against, all of which produce a
plausible-looking wrong number rather than a crash:

1. **Text area must be a union, not a sum.** Overlapping text boxes double-count
   the shared pixels, inflating the ratio and failing good images. Computed here
   by coordinate compression — exact integers, no rasterisation, no float.
2. **Boundary convention.** Right and bottom are exclusive throughout, matching
   PIL and most detectors. A box flush against the safe-zone edge is inside it.
3. **Which denominator.** The text ratio is over the *full canvas*, not the
   usable area — that is how the platform guidance reads. The two differ by ~30%
   on a 9:16 placement, which is the difference between passing and failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..spec.models import PlacementSpec

TEXT_LABEL = "text"

OVERLAP_LIMIT = 0.30
"""Fraction of the smaller element that may be covered before it counts as
occlusion. 0.30 is a starting value — T1.6's annotations are what will tell us
where a human reviewer actually draws this line."""


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in pixels. `right` and `bottom` are exclusive."""

    left: int
    top: int
    right: int
    bottom: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.right < self.left or self.bottom < self.top:
            raise ValueError(f"inverted box: {self}")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def intersection_area(self, other: "Box") -> int:
        w = min(self.right, other.right) - max(self.left, other.left)
        h = min(self.bottom, other.bottom) - max(self.top, other.top)
        return w * h if w > 0 and h > 0 else 0

    def is_inside(self, box: tuple[int, int, int, int]) -> bool:
        left, top, right, bottom = box
        return (
            self.left >= left
            and self.top >= top
            and self.right <= right
            and self.bottom <= bottom
        )

    def overflow(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """How far the box pokes out of `box` on each side (0 when inside).

        Returned so a violation can say *how badly*, not just *that* — a 2px
        bleed and a headline sitting fully under the caption bar are the same
        boolean but very different rework.
        """
        left, top, right, bottom = box
        return (
            max(0, left - self.left),
            max(0, top - self.top),
            max(0, self.right - right),
            max(0, self.bottom - bottom),
        )


def union_area(boxes: Sequence[Box]) -> int:
    """Exact area covered by the union of `boxes`.

    Coordinate compression: the distinct x and y edges cut the plane into a grid
    of cells, each cell is entirely inside or entirely outside every box, so
    summing the covered cells is exact. O(n^2) cells for n boxes, which is
    nothing at the scale of one creative, and — unlike rasterising — it cannot
    drift by a pixel.
    """
    boxes = [b for b in boxes if b.area > 0]
    if not boxes:
        return 0

    xs = sorted({b.left for b in boxes} | {b.right for b in boxes})
    ys = sorted({b.top for b in boxes} | {b.bottom for b in boxes})

    total = 0
    for xi in range(len(xs) - 1):
        x0, x1 = xs[xi], xs[xi + 1]
        for yi in range(len(ys) - 1):
            y0, y1 = ys[yi], ys[yi + 1]
            for b in boxes:
                if b.left <= x0 and b.right >= x1 and b.top <= y0 and b.bottom >= y1:
                    total += (x1 - x0) * (y1 - y0)
                    break
    return total


@dataclass(frozen=True, slots=True)
class LayoutViolation:
    rule: str
    severity: str
    """"hard" — the creative comes back for rework. "soft" — it ships but pays."""
    detail: str
    measured: float = 0.0
    limit: float = 0.0

    def __str__(self) -> str:
        return f"[{self.severity}] {self.rule}: {self.detail}"


@dataclass(frozen=True, slots=True)
class LayoutVerdict:
    violations: tuple[LayoutViolation, ...]
    text_ratio: float
    text_area: int
    canvas_area: int
    max_pairwise_overlap: float
    checked_rules: tuple[str, ...]

    @property
    def hard_violations(self) -> tuple[LayoutViolation, ...]:
        return tuple(v for v in self.violations if v.severity == "hard")

    @property
    def soft_violations(self) -> tuple[LayoutViolation, ...]:
        return tuple(v for v in self.violations if v.severity == "soft")

    @property
    def passed(self) -> bool:
        """Hard violations only. Soft ones cost score, not shippability."""
        return not self.hard_violations

    def explain(self) -> str:
        if not self.violations:
            return f"layout: clean (text {self.text_ratio:.1%})"
        return "layout: " + " | ".join(str(v) for v in self.violations)


@dataclass
class LayoutChecker:
    spec: PlacementSpec

    # ------------------------------------------------------------- rules

    def _check_canvas(self, size: tuple[int, int]) -> list[LayoutViolation]:
        w, h = size
        if (w, h) == (self.spec.width, self.spec.height):
            return []
        return [
            LayoutViolation(
                rule="canvas_size",
                severity="hard",
                detail=f"expected {self.spec.width}x{self.spec.height}, got {w}x{h}",
            )
        ]

    def _check_safe_zone(self, boxes: Sequence[Box]) -> list[LayoutViolation]:
        usable = self.spec.usable_box
        out: list[LayoutViolation] = []
        for b in boxes:
            if b.is_inside(usable):
                continue
            l, t, r, bo = b.overflow(usable)
            worst = max(l, t, r, bo)
            side = ("left", "top", "right", "bottom")[[l, t, r, bo].index(worst)]
            out.append(
                LayoutViolation(
                    rule="safe_zone",
                    severity="hard",
                    detail=f"{b.label or 'element'} crosses the {side} safe zone by {worst}px",
                    measured=float(worst),
                    limit=0.0,
                )
            )
        return out

    def _check_text_ratio(self, boxes: Sequence[Box]) -> tuple[list[LayoutViolation], int, float]:
        text_boxes = [b for b in boxes if b.label == TEXT_LABEL]
        area = union_area(text_boxes)
        canvas = self.spec.width * self.spec.height
        ratio = area / canvas if canvas else 0.0
        if ratio <= self.spec.text_max_ratio:
            return [], area, ratio
        return (
            [
                LayoutViolation(
                    rule="text_ratio",
                    severity=self.spec.text_ratio_severity,
                    detail=f"text covers {ratio:.1%} of the canvas, guidance is "
                    f"{self.spec.text_max_ratio:.0%}",
                    measured=ratio,
                    limit=self.spec.text_max_ratio,
                )
            ],
            area,
            ratio,
        )

    def _anchor_region(self, anchor: str) -> tuple[float, float, float, float]:
        """The (x0, y0, x1, y1) band a logo's centre must fall in.

        Bands are thirds of the canvas, and the edge bands run to two-thirds
        rather than one — lenient enough to survive detector jitter and ordinary
        design variation, strict enough that a bottom-right logo still fails a
        top_left rule. Anything tighter would fail on placement noise; anything
        looser would stop being a rule.
        """
        w, h = float(self.spec.width), float(self.spec.height)
        tx, ty = w / 3, h / 3
        x_bands = {"left": (0.0, 2 * tx), "right": (tx, w), "center": (tx, 2 * tx)}
        y_bands = {"top": (0.0, 2 * ty), "bottom": (ty, h), "center": (ty, 2 * ty)}

        if anchor == "center":
            vert = horiz = "center"
        else:
            vert, _, horiz = anchor.partition("_")

        x0, x1 = x_bands.get(horiz, (0.0, w))
        y0, y1 = y_bands.get(vert, (0.0, h))
        return (x0, y0, x1, y1)

    def _check_logo(self, boxes: Sequence[Box]) -> list[LayoutViolation]:
        rule = self.spec.logo
        logos = [b for b in boxes if b.label == "logo"]
        if not logos:
            return []  # presence is L3's element check (T1.4b), not geometry's
        out: list[LayoutViolation] = []
        logo = max(logos, key=lambda b: b.area)

        min_side = min(self.spec.width, self.spec.height)
        required = rule.min_size_pct * min_side
        extent = max(logo.width, logo.height)
        if extent < required:
            out.append(
                LayoutViolation(
                    rule="logo_min_size",
                    severity="hard",
                    detail=f"logo is {extent}px, needs {required:.0f}px "
                    f"({rule.min_size_pct:.0%} of the short side)",
                    measured=float(extent),
                    limit=required,
                )
            )

        if rule.anchor != "any":
            cx, cy = logo.center
            x0, y0, x1, y1 = self._anchor_region(rule.anchor)
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                out.append(
                    LayoutViolation(
                        rule="logo_anchor",
                        severity="hard",
                        detail=f"logo centre ({cx:.0f}, {cy:.0f}) is outside the "
                        f"{rule.anchor} region",
                    )
                )
        return out

    def _check_overlap(self, boxes: Sequence[Box]) -> tuple[list[LayoutViolation], float]:
        """Pairwise overlap between non-text elements, normalised by the smaller
        box. Feeds `element_overlap_cap` — a CTA sitting on the product is a
        rework even when every other rule passes."""
        elements = [b for b in boxes if b.label and b.label != TEXT_LABEL and b.area > 0]
        worst = 0.0
        worst_pair = ("", "")
        for i, a in enumerate(elements):
            for b in elements[i + 1 :]:
                inter = a.intersection_area(b)
                if not inter:
                    continue
                frac = inter / min(a.area, b.area)
                if frac > worst:
                    worst, worst_pair = frac, (a.label, b.label)
        if worst <= OVERLAP_LIMIT:
            return [], worst
        return (
            [
                LayoutViolation(
                    rule="element_overlap",
                    severity="hard",
                    detail=f"{worst_pair[0]} and {worst_pair[1]} overlap by {worst:.0%}",
                    measured=worst,
                    limit=OVERLAP_LIMIT,
                )
            ],
            worst,
        )

    # ------------------------------------------------------------- entry

    def check(self, boxes: Iterable[Box], canvas: tuple[int, int] | None = None) -> LayoutVerdict:
        boxes = list(boxes)
        violations: list[LayoutViolation] = []
        rules: list[str] = []

        if canvas is not None:
            violations += self._check_canvas(canvas)
            rules.append("canvas_size")

        violations += self._check_safe_zone(boxes)
        rules.append("safe_zone")

        ratio_v, text_area, text_ratio = self._check_text_ratio(boxes)
        violations += ratio_v
        rules.append("text_ratio")

        violations += self._check_logo(boxes)
        rules.append("logo")

        overlap_v, worst_overlap = self._check_overlap(boxes)
        violations += overlap_v
        rules.append("element_overlap")

        return LayoutVerdict(
            violations=tuple(violations),
            text_ratio=text_ratio,
            text_area=text_area,
            canvas_area=self.spec.width * self.spec.height,
            max_pairwise_overlap=worst_overlap,
            checked_rules=tuple(rules),
        )
