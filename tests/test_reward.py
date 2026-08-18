"""T1.8 acceptance: A1.8a–A1.8i and A1.13.

The three that actually catch bugs are the gate short-circuit, the cap ceiling,
and monotonicity. The last one is the quiet one: a sign flip, a swapped weight,
or capping before rather than after aggregation will not crash and will not look
wrong in a single example — it will just train the model somewhere else.
"""

from __future__ import annotations

import random

import pytest

from darkroom.verifier.reward import (
    CAPS,
    COMPONENT_WEIGHTS,
    HARD_CHECKS,
    RewardAssembler,
    RewardConfigError,
)

ALL_PASS = dict.fromkeys(HARD_CHECKS, True)


@pytest.fixture(scope="module")
def rw() -> RewardAssembler:
    return RewardAssembler()


def _full(value: float = 1.0) -> dict[str, float]:
    return dict.fromkeys(COMPONENT_WEIGHTS, value)


# ============================================================ A1.8a weights


def test_a1_8a_weights_sum_to_one() -> None:
    assert sum(COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)


def test_a1_8a_bad_weights_are_rejected_at_construction() -> None:
    with pytest.raises(RewardConfigError):
        RewardAssembler(weights={"text_render": 0.5, "layout": 0.2})
    with pytest.raises(RewardConfigError):
        RewardAssembler(weights={"text_render": 1.5, "layout": -0.5})


def test_aesthetic_stays_a_monitor_not_a_steering_wheel() -> None:
    """0.05 is a design commitment, not a tuning choice: a learned aesthetic
    reward with real weight is the first thing reward hacking eats."""
    assert COMPONENT_WEIGHTS["aesthetic"] <= 0.05


# ============================================================== A1.8b range


def test_a1_8b_reward_stays_in_range_over_random_inputs(rw: RewardAssembler) -> None:
    rng = random.Random(20260818)
    names = list(COMPONENT_WEIGHTS)
    cap_names = list(CAPS)
    for _ in range(10_000):
        scores = {
            n: (None if rng.random() < 0.2 else rng.random()) for n in names
        }
        caps = [c for c in cap_names if rng.random() < 0.15]
        checks = {n: rng.choice([True, False, None]) for n in HARD_CHECKS}
        out = rw.assemble(scores, checks, caps)
        assert 0.0 <= out.reward <= 1.0
        assert 0.0 <= out.uncapped_reward <= 1.0
        assert 0.0 <= out.coverage <= 1.0


def test_out_of_range_component_score_is_rejected(rw: RewardAssembler) -> None:
    with pytest.raises(RewardConfigError):
        rw.assemble({"layout": 1.4}, ALL_PASS)
    with pytest.raises(RewardConfigError):
        rw.assemble({"layout": -0.1}, ALL_PASS)


def test_unknown_names_are_rejected_rather_than_ignored(rw: RewardAssembler) -> None:
    """A typo that is silently dropped would remove a component from the reward
    without anyone noticing."""
    with pytest.raises(RewardConfigError):
        rw.assemble({"text_rendering": 1.0}, ALL_PASS)
    with pytest.raises(RewardConfigError):
        rw.assemble(_full(), ALL_PASS, ["gibberish"])
    with pytest.raises(RewardConfigError):
        rw.assemble(_full(), {"complience": True})


# ======================================================== A1.8c gate blocks


def test_a1_8c_gate_zeroes_reward_regardless_of_everything_else(
    rw: RewardAssembler,
) -> None:
    """One-vote veto: no combination of other scores can lift a non-compliant
    image off zero."""
    rng = random.Random(7)
    for _ in range(500):
        scores = {n: rng.random() for n in COMPONENT_WEIGHTS}
        checks = {**ALL_PASS, "compliance": False}
        out = rw.assemble(scores, checks)
        assert out.reward == 0.0
        assert out.gate_blocked
        assert out.deliverable is False


def test_gate_blocked_is_distinguishable_from_merely_scoring_zero(
    rw: RewardAssembler,
) -> None:
    blocked = rw.assemble(_full(1.0), {**ALL_PASS, "compliance": False})
    scored_zero = rw.assemble(_full(0.0), ALL_PASS)
    assert blocked.reward == scored_zero.reward == 0.0
    assert blocked.gate_blocked and not scored_zero.gate_blocked
    assert "gate blocked" in blocked.explain()


def test_unknown_compliance_does_not_zero_the_reward(rw: RewardAssembler) -> None:
    """Blocking on *unknown* compliance would zero every reward for as long as
    the image checker is unwired, killing all gradient during development. The
    gate fires on known-failure only; the uncertainty shows up in deliverable."""
    out = rw.assemble(_full(0.8), {**ALL_PASS, "compliance": None})
    assert out.reward > 0
    assert not out.gate_blocked
    assert out.deliverable is None


# ========================================================= A1.8d cap ceiling


def test_a1_8d_cap_is_a_ceiling_that_cannot_be_bought_off(rw: RewardAssembler) -> None:
    """The whole reason caps are not deductions: a perfect score everywhere else
    must not purchase tolerance for the defect."""
    perfect = rw.assemble(_full(1.0), ALL_PASS)
    assert perfect.reward == pytest.approx(1.0)

    capped = rw.assemble(_full(1.0), ALL_PASS, ["gibberish_text"])
    assert capped.reward == pytest.approx(CAPS["gibberish_text"])
    assert capped.uncapped_reward == pytest.approx(1.0)


def test_a1_8d_the_lowest_ceiling_binds(rw: RewardAssembler) -> None:
    out = rw.assemble(_full(1.0), ALL_PASS, ["template_collapse", "gibberish_text"])
    assert out.reward == pytest.approx(min(CAPS["template_collapse"], CAPS["gibberish_text"]))
    assert out.binding_cap is not None
    assert out.binding_cap.name == "gibberish_text"


def test_cap_does_not_raise_a_score_already_below_it(rw: RewardAssembler) -> None:
    """min(), not assignment. A defect must never improve the reward."""
    low = rw.assemble(_full(0.05), ALL_PASS)
    capped = rw.assemble(_full(0.05), ALL_PASS, ["template_collapse"])
    assert capped.reward == pytest.approx(low.reward)
    assert capped.reward < CAPS["template_collapse"]


def test_uncapped_reward_is_retained_for_diagnosis(rw: RewardAssembler) -> None:
    """A hard ceiling can flatten a whole GRPO group onto one value and erase the
    within-group variance. Keeping the raw score is what lets a monitor measure
    how often that happens, so the ceiling-vs-multiplicative choice can be made
    on evidence."""
    a = rw.assemble(_full(0.9), ALL_PASS, ["gibberish_text"])
    b = rw.assemble(_full(0.3), ALL_PASS, ["gibberish_text"])
    assert a.reward == b.reward, "hard ceiling flattens — this is the known trade"
    assert a.uncapped_reward > b.uncapped_reward, "but the raw ordering survives"


# ================================================== A1.8e missing components


def test_a1_8e_missing_component_is_neither_zero_nor_one(rw: RewardAssembler) -> None:
    """Absent must not read as broken, and must not read as perfect — the second
    is the same failure as a gate reporting green because half of it never ran."""
    partial = rw.assemble({"text_render": 0.5, "layout": 0.5}, ALL_PASS)

    as_zero = rw.assemble({**_full(0.0), "text_render": 0.5, "layout": 0.5}, ALL_PASS)
    as_one = rw.assemble({**_full(1.0), "text_render": 0.5, "layout": 0.5}, ALL_PASS)

    assert partial.reward == pytest.approx(0.5)
    assert partial.reward > as_zero.reward
    assert partial.reward < as_one.reward


def test_a1_8e_renormalisation_is_over_the_weights_that_ran(rw: RewardAssembler) -> None:
    out = rw.assemble({"text_render": 1.0, "layout": 0.0}, ALL_PASS)
    w_t, w_l = COMPONENT_WEIGHTS["text_render"], COMPONENT_WEIGHTS["layout"]
    assert out.reward == pytest.approx(w_t / (w_t + w_l))
    assert out.coverage == pytest.approx(w_t + w_l)


def test_a1_8e_coverage_reports_how_much_of_the_verifier_ran(rw: RewardAssembler) -> None:
    assert rw.assemble(_full(1.0), ALL_PASS).coverage == pytest.approx(1.0)
    assert rw.assemble({}, ALL_PASS).coverage == 0.0
    assert set(rw.assemble({"layout": 1.0}, ALL_PASS).ran) == {"layout"}


def test_nothing_ran_scores_zero_without_dividing_by_zero(rw: RewardAssembler) -> None:
    out = rw.assemble({}, ALL_PASS)
    assert out.reward == 0.0 and out.coverage == 0.0


# ================================================ A1.8f deliverable honesty


def test_a1_8f_missing_hard_check_makes_deliverable_unknown(rw: RewardAssembler) -> None:
    for omitted in HARD_CHECKS:
        checks = {k: v for k, v in ALL_PASS.items() if k != omitted}
        out = rw.assemble(_full(1.0), checks)
        assert out.deliverable is None, f"omitting {omitted} produced {out.deliverable}"


def test_a1_8f_no_hard_checks_at_all_is_unknown_not_true(rw: RewardAssembler) -> None:
    assert rw.assemble(_full(1.0)).deliverable is None
    assert rw.assemble(_full(1.0), {}).deliverable is None


def test_a1_8f_deliverable_is_the_conjunction_of_every_hard_check(
    rw: RewardAssembler,
) -> None:
    assert rw.assemble(_full(1.0), ALL_PASS).deliverable is True
    for failed in HARD_CHECKS:
        checks = {**ALL_PASS, failed: False}
        assert rw.assemble(_full(1.0), checks).deliverable is False


def test_a1_8f_a_perfect_reward_can_still_be_undeliverable(rw: RewardAssembler) -> None:
    """The two outputs are independent by design. Collapsing them is what would
    reintroduce a binary signal."""
    out = rw.assemble(_full(1.0), {**ALL_PASS, "text_exact": False})
    assert out.reward == pytest.approx(1.0)
    assert out.deliverable is False


# ========================================================= A1.8g monotonic


def test_a1_8g_raising_any_component_never_lowers_the_reward(
    rw: RewardAssembler,
) -> None:
    """Guards against sign flips, swapped weights, and capping before rather
    than after aggregation — none of which crash, all of which train the model
    somewhere other than where we meant."""
    rng = random.Random(99)
    names = list(COMPONENT_WEIGHTS)
    for _ in range(2_000):
        scores = {n: rng.random() for n in names}
        caps = [c for c in CAPS if rng.random() < 0.2]
        before = rw.assemble(scores, ALL_PASS, caps).reward

        bumped = rng.choice(names)
        scores[bumped] = min(1.0, scores[bumped] + rng.uniform(0.01, 0.4))
        after = rw.assemble(scores, ALL_PASS, caps).reward

        assert after >= before - 1e-12, f"raising {bumped} lowered the reward"


def test_a1_8g_adding_a_cap_never_raises_the_reward(rw: RewardAssembler) -> None:
    rng = random.Random(1234)
    for _ in range(1_000):
        scores = {n: rng.random() for n in COMPONENT_WEIGHTS}
        caps = [c for c in CAPS if rng.random() < 0.3]
        base = rw.assemble(scores, ALL_PASS, caps).reward
        extra = rng.choice(list(CAPS))
        more = rw.assemble(scores, ALL_PASS, [*caps, extra]).reward
        assert more <= base + 1e-12


# ============================================ A1.8h / A1.13 explainability


def test_a1_8h_breakdown_reconstructs_the_reward(rw: RewardAssembler) -> None:
    scores = {"text_render": 0.8, "elements": 0.6, "layout": 1.0}
    out = rw.assemble(scores, ALL_PASS)
    ran_weight = sum(c.weight for c in out.components if c.ran)
    rebuilt = sum(c.contribution for c in out.components) / ran_weight
    assert rebuilt == pytest.approx(out.reward)


def test_a1_13_every_component_is_reported_even_when_it_did_not_run(
    rw: RewardAssembler,
) -> None:
    out = rw.assemble({"layout": 1.0}, ALL_PASS)
    assert {c.name for c in out.components} == set(COMPONENT_WEIGHTS)
    assert [c.name for c in out.components if not c.ran]


def test_top_deficits_rank_by_weighted_loss_not_raw_score(rw: RewardAssembler) -> None:
    """A near-miss on a heavy component outranks a middling score on a light
    one: 0.40 x 0.1 = 0.04 beats 0.05 x 0.5 = 0.025. Ranking by raw score would
    send the rework report chasing the aesthetic every time."""
    out = rw.assemble(
        {"text_render": 0.9, "elements": 1.0, "layout": 1.0, "brand": 1.0, "aesthetic": 0.5},
        ALL_PASS,
    )
    assert out.top_deficits(1)[0].name == "text_render"


def test_top_deficits_still_surface_a_totally_failed_light_component(
    rw: RewardAssembler,
) -> None:
    """The converse, and it is correct: 0.05 x 1.0 = 0.05 does outrank
    0.40 x 0.1 = 0.04. Weighted loss is the ordering, all the way down —
    pinned because the first version of the test above assumed otherwise."""
    out = rw.assemble(
        {"text_render": 0.9, "elements": 1.0, "layout": 1.0, "brand": 1.0, "aesthetic": 0.0},
        ALL_PASS,
    )
    assert out.top_deficits(1)[0].name == "aesthetic"


def test_explain_covers_every_state(rw: RewardAssembler) -> None:
    assert "gate blocked" in rw.assemble(
        _full(1.0), {**ALL_PASS, "compliance": False}
    ).explain()
    assert "capped" in rw.assemble(_full(1.0), ALL_PASS, ["safe_zone"]).explain()
    assert "coverage" in rw.assemble({"layout": 1.0}, ALL_PASS).explain()
    assert "deliverable unknown" in rw.assemble(_full(1.0), {}).explain()
    assert "not deliverable" in rw.assemble(
        _full(1.0), {**ALL_PASS, "layout_hard": False}
    ).explain()


# ======================================================== A1.8i determinism


def test_a1_8i_repeated_assembly_agrees(rw: RewardAssembler) -> None:
    scores = {"text_render": 0.42, "elements": 0.13, "layout": 0.77, "brand": 0.5}
    caps = ["safe_zone", "element_overlap"]
    first = rw.assemble(scores, ALL_PASS, caps)
    for _ in range(10):
        assert rw.assemble(scores, ALL_PASS, caps) == first


def test_cap_order_does_not_matter(rw: RewardAssembler) -> None:
    a = rw.assemble(_full(1.0), ALL_PASS, ["safe_zone", "gibberish_text"])
    b = rw.assemble(_full(1.0), ALL_PASS, ["gibberish_text", "safe_zone"])
    assert a.reward == b.reward


def test_duplicate_caps_are_idempotent(rw: RewardAssembler) -> None:
    once = rw.assemble(_full(1.0), ALL_PASS, ["safe_zone"])
    twice = rw.assemble(_full(1.0), ALL_PASS, ["safe_zone", "safe_zone"])
    assert once.reward == twice.reward
    assert len(twice.caps_hit) == 1
