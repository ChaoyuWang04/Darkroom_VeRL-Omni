"""T1.4a acceptance: A1.6 (geometry accuracy = 1.00) and its sub-criteria.

1.00 is only a reasonable bar because this layer is arithmetic — there is no
recognition error to absorb, so anything below 1.00 is a bug. The tests are
therefore written against the three ways integer geometry produces a
*plausible wrong number* rather than a crash:

  A1.6a  three-point boundary — exactly on the line, 1px in, 1px out
  A1.6b  union area — overlapping boxes must not double-count
  A1.6c  every placement in the registry is checkable
  A1.6d  determinism
  A1.6e  every violation explains itself
"""

from __future__ import annotations

import math

import pytest

from darkroom.spec import SpecRegistry
from darkroom.verifier import Box, LayoutChecker, union_area


@pytest.fixture(scope="module")
def registry() -> SpecRegistry:
    return SpecRegistry()


@pytest.fixture(scope="module")
def checker(registry: SpecRegistry) -> LayoutChecker:
    # 9:16 Meta: safe zone (60, 250) .. (960, 1600) on a 1080x1920 canvas.
    return LayoutChecker(registry.get_placement("meta", "1080x1920", strict_expiry=False))


def _rules(verdict) -> set[str]:
    return {v.rule for v in verdict.violations}


# =============================================================== A1.6b union


def test_a1_6b_disjoint_boxes_sum() -> None:
    assert union_area([Box(0, 0, 10, 10), Box(20, 20, 30, 30)]) == 200


def test_a1_6b_overlapping_boxes_are_not_double_counted() -> None:
    """The trap: 100 + 100 naively, but they share a 25px^2 corner. A sum here
    inflates the text ratio and fails images that are actually fine."""
    assert union_area([Box(0, 0, 10, 10), Box(5, 5, 15, 15)]) == 175


def test_a1_6b_fully_contained_box_adds_nothing() -> None:
    assert union_area([Box(0, 0, 100, 100), Box(10, 10, 20, 20)]) == 10_000


def test_a1_6b_identical_boxes_count_once() -> None:
    b = Box(3, 4, 13, 24)
    assert union_area([b, b, b]) == 200


def test_a1_6b_edge_touching_boxes_do_not_overlap() -> None:
    """Right/bottom are exclusive, so [0,10) and [10,20) abut without sharing a
    pixel. Getting this backwards silently loses or gains a row per box pair."""
    assert union_area([Box(0, 0, 10, 10), Box(10, 0, 20, 10)]) == 200


def test_a1_6b_three_way_overlap_is_exact() -> None:
    # Cross shape: a 30x10 bar and a 10x30 bar sharing a 10x10 centre.
    assert union_area([Box(0, 10, 30, 20), Box(10, 0, 20, 30)]) == 300 + 300 - 100


def test_a1_6b_degenerate_boxes_are_ignored() -> None:
    assert union_area([Box(5, 5, 5, 50), Box(0, 0, 10, 10)]) == 100
    assert union_area([]) == 0


# ======================================================== A1.6a safe zone


def test_a1_6a_box_flush_against_the_safe_zone_is_inside(checker: LayoutChecker) -> None:
    """Exactly-on-the-line is the case every off-by-one bug gets wrong."""
    left, top, right, bottom = checker.spec.usable_box
    flush = Box(left, top, right, bottom, label="text")
    assert "safe_zone" not in _rules(checker.check([flush]))


def test_a1_6a_one_pixel_inside_passes(checker: LayoutChecker) -> None:
    left, top, right, bottom = checker.spec.usable_box
    inside = Box(left + 1, top + 1, right - 1, bottom - 1, label="text")
    assert "safe_zone" not in _rules(checker.check([inside]))


@pytest.mark.parametrize("side", ["left", "top", "right", "bottom"])
def test_a1_6a_one_pixel_outside_fails_on_every_side(
    checker: LayoutChecker, side: str
) -> None:
    left, top, right, bottom = checker.spec.usable_box
    coords = {"left": left, "top": top, "right": right, "bottom": bottom}
    coords[side] += -1 if side in ("left", "top") else 1
    box = Box(coords["left"], coords["top"], coords["right"], coords["bottom"], label="text")

    verdict = checker.check([box])
    assert "safe_zone" in _rules(verdict), f"1px over the {side} edge was not caught"
    assert not verdict.passed
    v = next(v for v in verdict.violations if v.rule == "safe_zone")
    assert v.measured == 1.0
    assert side in v.detail


def test_safe_zone_violation_is_hard(checker: LayoutChecker) -> None:
    """Text under the platform UI comes back from a reviewer — it is rework,
    not a delivery tax."""
    verdict = checker.check([Box(0, 0, 500, 100, label="text")])
    assert verdict.hard_violations
    assert not verdict.passed


# ====================================================== A1.6a text ratio


def test_a1_6a_text_ratio_exactly_at_the_limit_passes(checker: LayoutChecker) -> None:
    spec = checker.spec
    height = int(spec.height * spec.text_max_ratio)
    verdict = checker.check([Box(0, 0, spec.width, height, label="text")])
    assert verdict.text_ratio == pytest.approx(spec.text_max_ratio, abs=1e-9)
    assert "text_ratio" not in _rules(verdict), "at-the-limit must not violate"


def test_a1_6a_text_ratio_one_row_over_the_limit_violates(checker: LayoutChecker) -> None:
    spec = checker.spec
    height = int(spec.height * spec.text_max_ratio) + 1
    verdict = checker.check([Box(0, 0, spec.width, height, label="text")])
    assert "text_ratio" in _rules(verdict)


def test_text_ratio_violation_is_soft_and_does_not_block(checker: LayoutChecker) -> None:
    """Meta retired the hard 20% rule in 2020. Gating on it would fail creatives
    the platform runs happily, which is the wrong direction of error for a
    metric called `deliverable`."""
    spec = checker.spec
    # Kept inside the safe zone so text_ratio is the *only* rule that fires —
    # otherwise a safe-zone violation would supply the hard failure and the
    # assertion below would prove nothing.
    left, top, right, _ = spec.usable_box
    tall = Box(left, top, right, top + int(spec.height * 0.5), label="text")
    verdict = checker.check([tall])

    assert [v.rule for v in verdict.violations] == ["text_ratio"]
    assert verdict.soft_violations and not verdict.hard_violations
    assert verdict.passed, "a soft violation must not block shippability"


def test_a1_6a_text_ratio_denominator_is_the_full_canvas(checker: LayoutChecker) -> None:
    """Canvas vs usable area differ by ~30% on 9:16 — the gap between passing
    and failing. Pinning the denominator stops that drifting silently."""
    spec = checker.spec
    box = Box(60, 250, 960, 700, label="text")
    verdict = checker.check([box])
    assert verdict.canvas_area == spec.width * spec.height
    assert verdict.text_ratio == pytest.approx(box.area / (spec.width * spec.height))
    assert verdict.text_ratio != pytest.approx(box.area / spec.usable_area)


def test_overlapping_text_uses_union_not_sum(checker: LayoutChecker) -> None:
    a = Box(100, 300, 500, 500, label="text")
    b = Box(300, 400, 700, 600, label="text")
    verdict = checker.check([a, b])
    assert verdict.text_area == union_area([a, b])
    assert verdict.text_area < a.area + b.area


# ============================================================ A1.6a logo


def test_a1_6a_logo_exactly_at_minimum_size_passes(checker: LayoutChecker) -> None:
    spec = checker.spec
    required = math.ceil(spec.logo.min_size_pct * min(spec.width, spec.height))
    logo = Box(100, 300, 100 + required, 300 + required, label="logo")
    assert "logo_min_size" not in _rules(checker.check([logo]))


def test_a1_6a_logo_one_pixel_under_minimum_fails(checker: LayoutChecker) -> None:
    spec = checker.spec
    required = math.ceil(spec.logo.min_size_pct * min(spec.width, spec.height))
    logo = Box(100, 300, 100 + required - 1, 300 + required - 1, label="logo")
    assert "logo_min_size" in _rules(checker.check([logo]))


def test_a1_6a_fractional_size_requirement_rounds_up_not_down(
    checker: LayoutChecker,
) -> None:
    """6% of a 1080px short side is 64.8px, and a logo cannot be 64.8px wide.
    The requirement is therefore met at 65 and missed at 64 — flooring instead
    would quietly admit every logo that is fractionally too small. Pinned here
    because the first version of this test made exactly that mistake."""
    spec = checker.spec
    exact = spec.logo.min_size_pct * min(spec.width, spec.height)
    assert exact != int(exact), "pick a placement with a fractional requirement"

    just_under = Box(100, 300, 100 + int(exact), 300 + int(exact), label="logo")
    just_over = Box(100, 300, 100 + math.ceil(exact), 300 + math.ceil(exact), label="logo")
    assert "logo_min_size" in _rules(checker.check([just_under]))
    assert "logo_min_size" not in _rules(checker.check([just_over]))


def test_logo_anchor_rejects_the_opposite_corner(checker: LayoutChecker) -> None:
    spec = checker.spec  # anchor is top_left
    required = math.ceil(spec.logo.min_size_pct * min(spec.width, spec.height))
    wrong = Box(800, 1400, 800 + required, 1400 + required, label="logo")
    assert "logo_anchor" in _rules(checker.check([wrong]))

    right = Box(100, 300, 100 + required, 300 + required, label="logo")
    assert "logo_anchor" not in _rules(checker.check([right]))


def test_absent_logo_is_not_a_geometry_violation(checker: LayoutChecker) -> None:
    """Presence is the element check's job (T1.4b). Geometry only rules on a
    logo that is actually there — conflating the two would report a missing
    logo as a placement error."""
    assert "logo_anchor" not in _rules(checker.check([Box(100, 300, 400, 400, label="text")]))
    assert "logo_min_size" not in _rules(checker.check([]))


# ========================================================= A1.6a overlap


def test_disjoint_elements_do_not_trip_overlap(checker: LayoutChecker) -> None:
    boxes = [
        Box(100, 300, 400, 600, label="product"),
        Box(500, 300, 800, 600, label="cta"),
    ]
    assert "element_overlap" not in _rules(checker.check(boxes))


def test_heavy_occlusion_is_caught(checker: LayoutChecker) -> None:
    boxes = [
        Box(100, 300, 500, 700, label="product"),
        Box(100, 300, 400, 600, label="cta"),  # 90000 / 90000 of the smaller box
    ]
    verdict = checker.check(boxes)
    assert "element_overlap" in _rules(verdict)
    assert verdict.max_pairwise_overlap == pytest.approx(1.0)


def test_overlap_is_normalised_by_the_smaller_element(checker: LayoutChecker) -> None:
    """A small badge fully on top of a large product is total occlusion of the
    badge. Normalising by the larger box would report it as negligible."""
    boxes = [
        Box(100, 300, 900, 1100, label="product"),
        Box(200, 400, 260, 460, label="cta"),
    ]
    assert checker.check(boxes).max_pairwise_overlap == pytest.approx(1.0)


def test_text_is_excluded_from_element_overlap(checker: LayoutChecker) -> None:
    """Text over a product is normal design, not occlusion."""
    boxes = [
        Box(100, 300, 900, 1100, label="product"),
        Box(200, 400, 800, 500, label="text"),
    ]
    assert "element_overlap" not in _rules(checker.check(boxes))


# =========================================================== canvas size


def test_canvas_mismatch_is_hard(checker: LayoutChecker) -> None:
    verdict = checker.check([], canvas=(1080, 1921))
    assert "canvas_size" in _rules(verdict)
    assert not verdict.passed


def test_canvas_match_passes(checker: LayoutChecker) -> None:
    assert "canvas_size" not in _rules(checker.check([], canvas=(1080, 1920)))


def test_canvas_not_checked_when_not_supplied(checker: LayoutChecker) -> None:
    assert "canvas_size" not in checker.check([]).checked_rules


# ================================================ A1.6c / A1.6d / A1.6e


def test_a1_6c_every_registered_placement_is_checkable(registry: SpecRegistry) -> None:
    for platform, size in registry.placements:
        spec = registry.get_placement(platform, size, strict_expiry=False)
        left, top, right, bottom = spec.usable_box
        c = LayoutChecker(spec)
        clean = c.check(
            [Box(left, top, right, min(bottom, top + 10), label="text")],
            canvas=(spec.width, spec.height),
        )
        assert clean.passed, f"{platform}/{size}: {clean.explain()}"


def test_a1_6d_repeated_calls_agree(checker: LayoutChecker) -> None:
    boxes = [
        Box(60, 250, 900, 500, label="text"),
        Box(100, 600, 400, 900, label="product"),
        Box(70, 260, 200, 390, label="logo"),
    ]
    first = checker.check(boxes, canvas=(1080, 1920))
    for _ in range(5):
        assert checker.check(boxes, canvas=(1080, 1920)) == first


def test_a1_6e_every_violation_carries_a_reason(checker: LayoutChecker) -> None:
    """An unexplained failure cannot be bucketed in the rework report, and
    cannot be debugged when the reward moves the wrong way."""
    verdict = checker.check(
        [Box(0, 0, 1080, 900, label="text"), Box(0, 0, 10, 10, label="logo")],
        canvas=(1080, 1000),
    )
    assert verdict.violations
    for v in verdict.violations:
        assert v.rule and v.severity in ("hard", "soft") and v.detail
    assert verdict.explain()


def test_clean_layout_reports_clean(checker: LayoutChecker) -> None:
    spec = checker.spec
    required = math.ceil(spec.logo.min_size_pct * min(spec.width, spec.height))
    boxes = [
        Box(70, 260, 70 + required, 260 + required, label="logo"),
        Box(100, 500, 900, 700, label="text"),
        Box(100, 800, 900, 1500, label="product"),
    ]
    verdict = checker.check(boxes, canvas=(spec.width, spec.height))
    assert verdict.passed and not verdict.violations, verdict.explain()
    assert "clean" in verdict.explain()
