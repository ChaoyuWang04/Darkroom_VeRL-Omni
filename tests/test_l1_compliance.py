"""T1.2 acceptance: A1.1 (recall >= 0.99) and A1.2 (precision >= 0.90).

The real numbers come from the 200-image annotation set in T1.6. What this file
does is hold the two properties that make those numbers achievable at all:

* recall survives obfuscation (full-width, zero-width, separators),
* precision survives the substring trap (`ass` in `assassin`),

plus the structural guarantee that a half-wired gate reports itself as
half-wired rather than green.
"""

from __future__ import annotations

from datetime import date

import pytest

from darkroom.spec import BannedTerm, ContentPolicy, SpecRegistry
from darkroom.verifier import ComplianceGate, ComplianceHit, normalize


@pytest.fixture(scope="module")
def gate() -> ComplianceGate:
    registry = SpecRegistry()
    return ComplianceGate(policy=registry.get_policy("cn", strict_expiry=False))


def _blocked(gate: ComplianceGate, **fields: str) -> bool:
    return not gate.check(fields).passed


# ------------------------------------------------------- A1.1 recall side

VIOLATIONS = [
    ("headline", "限时5折 最好的游戏"),          # 广告法绝对化用语
    ("headline", "国家级品质保证"),
    ("subline", "保本理财 零风险"),
    ("subline", "根治失眠 无副作用"),
    ("headline", "Watch free porn now"),
    ("cta", "Buy cocaine online"),
    ("headline", "underage girls"),
    ("subline", "包赢 百分百中奖"),
]


@pytest.mark.parametrize("field,text", VIOLATIONS)
def test_a1_1_plain_violations_are_caught(gate: ComplianceGate, field: str, text: str) -> None:
    assert _blocked(gate, **{field: text}), f"missed: {text!r}"


OBFUSCATED = [
    "ＷＡＴＣＨ ＦＲＥＥ ＰＯＲＮ",          # full-width
    "watch free p​orn",               # zero-width joiner inside the term
    "p.o.r.n now",                         # separator evasion (term is opt-in)
    "P-O-R-N",
    "最​好的游戏",                     # zero-width inside a CJK term
]


@pytest.mark.parametrize("text", OBFUSCATED)
def test_a1_1_obfuscation_does_not_buy_a_pass(gate: ComplianceGate, text: str) -> None:
    """Recall is the expensive metric here: a miss is a compliance incident,
    so the cheap spelling-around tricks have to fold."""
    assert _blocked(gate, headline=text), f"evasion succeeded: {text!r}"


# ---------------------------------------------------- A1.2 precision side

# Every string here contains a banned term as a *substring* but is innocent.
# Naive `in` matching fails all of them, which is exactly how a blocklist ends
# up with 99% recall and unusable precision.
SUBSTRING_TRAPS = [
    "Play as an assassin",        # contains "ass"-like fragments
    "Seashells by the seashore",  # "hell"
    "Analyse your stats",         # "anal"
    "Grape harvest festival",     # "rape"
    "Scunthorpe United fixtures", # the canonical case
    "Classic puzzle game",
    "Massive update out now",
]


@pytest.mark.parametrize("text", SUBSTRING_TRAPS)
def test_a1_2_innocent_substrings_are_not_blocked(gate: ComplianceGate, text: str) -> None:
    verdict = gate.check({"headline": text})
    assert verdict.passed, f"false positive on {text!r}: {verdict.explain()}"


CLEAN_COPY = [
    {"headline": "限时5折", "subline": "仅剩3天", "cta": "立即下载"},
    {"headline": "50% OFF TODAY", "subline": "Ends Sunday", "cta": "Download now"},
    {"headline": "新赛季开启", "subline": "登录送英雄", "cta": "马上体验"},
    {"headline": "Free to play", "subline": "No download required", "cta": "Play now"},
]


@pytest.mark.parametrize("fields", CLEAN_COPY)
def test_a1_2_ordinary_ad_copy_passes(gate: ComplianceGate, fields: dict[str, str]) -> None:
    verdict = gate.check(fields)
    assert verdict.passed, verdict.explain()


def test_a1_2_word_boundary_logic_holds_against_real_traps() -> None:
    """The traps above only prove precision if the blocklist actually contains
    the trapping term — otherwise they pass for the wrong reason. This builds a
    policy out of exactly the short terms that break naive matching, so the
    word-boundary mechanism is what is under test."""
    policy = ContentPolicy(
        region="synthetic",
        terms=(
            BannedTerm(term="ass", category="adult"),
            BannedTerm(term="hell", category="violence"),
            BannedTerm(term="anal", category="adult"),
            BannedTerm(term="rape", category="violence"),
            BannedTerm(term="cum", category="adult"),
        ),
        banned_visual_categories=(),
        source="test",
        version="t",
        valid_to=date(2099, 1, 1),
    )
    g = ComplianceGate(policy=policy)

    innocent = [
        "Play as an assassin",
        "Seashells by the seashore",
        "Analyse your stats",
        "Grape harvest festival",
        "Scunthorpe United",
        "Cucumber salad recipe",
        "Class assignment",
        "Shell company",
        "Document circumference",
    ]
    for text in innocent:
        v = g.check({"headline": text})
        assert v.passed, f"false positive on {text!r}: {v.explain()}"

    # ...and the same terms must still fire when they stand alone.
    for text in ["what the hell", "ass kicking action", "anal fissure clinic"]:
        assert not g.check({"headline": text}).passed, f"missed: {text!r}"


def test_cjk_terms_match_as_substrings_since_there_are_no_word_breaks() -> None:
    """Latin gets \\b; CJK cannot — there are no delimiters, so a term embedded
    in a longer run must still match. Applying \\b uniformly would silently
    disable every Chinese entry in the blocklist."""
    policy = ContentPolicy(
        region="synthetic-cjk",
        terms=(BannedTerm(term="保本", category="financial_claim"),),
        banned_visual_categories=(),
        source="test",
        version="t",
        valid_to=date(2099, 1, 1),
    )
    g = ComplianceGate(policy=policy)
    assert not g.check({"headline": "本产品保本保息稳健理财"}).passed


def test_evasion_matching_stays_inside_word_boundaries(gate: ComplianceGate) -> None:
    """Separator tolerance must not leak across a longer alphanumeric run —
    that is the failure mode that would sink precision."""
    assert gate.check({"headline": "spornographic is not a word"}).passed


# --------------------------------------------------------------- structure


def test_verdict_names_the_offending_field_and_category(gate: ComplianceGate) -> None:
    """An unexplained rejection is not actionable — the rework report buckets by
    reason, so the verdict must carry one."""
    verdict = gate.check({"headline": "限时5折", "subline": "保本稳赚"})
    assert not verdict.passed
    fields = {h.field for h in verdict.hits}
    assert fields == {"subline"}
    assert "financial_claim" in verdict.categories
    assert verdict.policy_version


def test_gate_reports_itself_incomplete_until_the_image_check_is_mounted(
    gate: ComplianceGate,
) -> None:
    """The dangerous failure is a gate that reads green because half of it was
    never wired. `passed` speaks only for checks that ran; `gate_complete` is
    what a deliverable claim must consult."""
    verdict = gate.check({"headline": "限时5折"})
    assert verdict.passed
    assert verdict.text_checked
    assert not verdict.image_checked
    assert not verdict.gate_complete
    assert "image check did not run" in verdict.explain()


def test_gate_becomes_complete_once_an_image_checker_is_supplied() -> None:
    class StubChecker:
        def check(self, image_path: str, categories):  # noqa: ANN001, ARG002
            return (ComplianceHit("nudity", "adult", "image", (0, 0)),)

    registry = SpecRegistry()
    g = ComplianceGate(
        policy=registry.get_policy("global", strict_expiry=False),
        image_checker=StubChecker(),
    )
    verdict = g.check({"headline": "限时5折"}, image_path="/tmp/x.png")
    assert verdict.gate_complete
    assert not verdict.passed
    assert verdict.categories == ("adult",)


def test_empty_fields_are_skipped_not_matched(gate: ComplianceGate) -> None:
    assert gate.check({"headline": "", "subline": None or ""}).passed


def test_normalize_is_idempotent() -> None:
    for s in ["ＦＲＥＥ", "a​b", "MiXeD Case", "限时5折"]:
        assert normalize(normalize(s)) == normalize(s)
