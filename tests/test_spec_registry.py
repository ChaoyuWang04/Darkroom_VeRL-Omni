"""T1.1 acceptance: A1.8 (exact lookup = 1.00), A1.9 (coverage >= 9), A1.10 (schema clean)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from darkroom.spec import SpecExpired, SpecNotFound, SpecRegistry

PLATFORMS = ("meta", "tiktok", "applovin")


@pytest.fixture(scope="module")
def registry() -> SpecRegistry:
    return SpecRegistry()


# --------------------------------------------------------------- A1.10 schema


def test_a1_10_every_spec_file_parses_and_validates(registry: SpecRegistry) -> None:
    """Construction runs every dataclass validator. Zero errors is the bar."""
    assert registry.placements, "no placements loaded"
    assert registry.regions, "no policies loaded"
    assert registry.brands, "no brands loaded"


def test_usable_area_is_positive_everywhere(registry: SpecRegistry) -> None:
    for platform, size in registry.placements:
        spec = registry.get_placement(platform, size, strict_expiry=False)
        left, top, right, bottom = spec.usable_box
        assert right > left and bottom > top, f"{platform}/{size} has no usable area"
        assert spec.usable_area <= spec.width * spec.height


# ------------------------------------------------------------- A1.9 coverage


def test_a1_9_placement_coverage_at_least_nine(registry: SpecRegistry) -> None:
    assert len(registry.placements) >= 9, f"only {len(registry.placements)} placements"


def test_a1_9_at_least_three_platforms(registry: SpecRegistry) -> None:
    platforms = {p for p, _ in registry.placements}
    assert platforms >= set(PLATFORMS), f"missing platforms: {set(PLATFORMS) - platforms}"


# ---------------------------------------------------- A1.8 exact lookup = 1.00


def test_a1_8_every_known_key_returns_its_own_record(registry: SpecRegistry) -> None:
    """The metric behind A1.8: for all known keys, the returned record is the
    one that key names. Anything less than 1.00 means a whole batch of images
    got judged against another placement's rules."""
    for platform, size in registry.placements:
        spec = registry.get_placement(platform, size, strict_expiry=False)
        assert spec.platform == platform and spec.size == size


def test_a1_8_unknown_key_raises_and_never_falls_back(registry: SpecRegistry) -> None:
    """A near-miss must fail loudly. Returning the closest match is the single
    most dangerous behaviour this layer could have."""
    with pytest.raises(SpecNotFound):
        registry.get_placement("meta", "1081x1080", strict_expiry=False)
    with pytest.raises(SpecNotFound):
        registry.get_placement("META", "1080x1080", strict_expiry=False)
    with pytest.raises(SpecNotFound):
        registry.get_placement("meta_ads", "1080x1080", strict_expiry=False)
    with pytest.raises(SpecNotFound):
        registry.get_policy("CN")
    with pytest.raises(SpecNotFound):
        registry.get_brand("demo")


# ------------------------------------------------------------------- expiry


def test_expired_spec_raises_by_default(registry: SpecRegistry) -> None:
    spec = registry.get_placement("meta", "1080x1080", strict_expiry=False)
    with pytest.raises(SpecExpired):
        registry.get_placement("meta", "1080x1080", as_of=spec.valid_to + timedelta(days=1))


def test_expiry_can_be_waived_deliberately(registry: SpecRegistry) -> None:
    spec = registry.get_placement(
        "meta", "1080x1080", as_of=date(2099, 1, 1), strict_expiry=False
    )
    assert spec.platform == "meta"


def test_every_spec_carries_provenance(registry: SpecRegistry) -> None:
    """source/version/valid_to are what make the flywheel able to correct a rule
    without retraining. A record without them cannot be superseded safely."""
    for platform, size in registry.placements:
        spec = registry.get_placement(platform, size, strict_expiry=False)
        assert spec.source and spec.version and spec.valid_to


# ------------------------------------------------------------ policy extends


def test_cn_policy_inherits_the_global_baseline(registry: SpecRegistry) -> None:
    glob = registry.get_policy("global", strict_expiry=False)
    cn = registry.get_policy("cn", strict_expiry=False)
    global_terms = {t.term for t in glob.terms}
    cn_terms = {t.term for t in cn.terms}
    assert global_terms <= cn_terms, "cn lost part of the global baseline"
    assert "最好" in cn_terms and "最好" not in global_terms
    assert "absolute_claim" in cn.categories


def test_inherited_visual_categories_are_merged_without_duplicates(
    registry: SpecRegistry,
) -> None:
    cn = registry.get_policy("cn", strict_expiry=False)
    cats = cn.banned_visual_categories
    assert len(cats) == len(set(cats))
    assert "nudity" in cats and "national_symbol_misuse" in cats


def test_brand_hex_parses_to_rgb(registry: SpecRegistry) -> None:
    brand = registry.get_brand("demo_casual_game")
    assert brand.primary_rgb == (0xE6, 0x39, 0x46)
