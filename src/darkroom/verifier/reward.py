"""Reward assembly — the layer verdicts become one number here.

Two outputs, and keeping them apart is the whole point of this module:

* ``deliverable`` — boolean, the business anchor. Can this ship untouched?
* ``reward`` — continuous in [0, 1], the training signal.

The tempting shortcut is ``reward = 1.0 if deliverable else 0.0``, and it would
quietly destroy training. An image whose copy renders at 90% and one that is
pure gibberish are the same boolean, so a group of 16 samples that all fall
short scores 16 zeroes: no within-group variance, no advantage, no gradient.
GRPO learns nothing from a prompt it always fails. The reward therefore has to
stay continuous *inside* the failing region, which is exactly where a
shippability flag carries no information.

Three further decisions worth stating, because each has a plausible wrong
version:

**Caps are ceilings, not deductions.** A deduction can be earned back
elsewhere, which teaches the model to trade: "perfect layout buys me a little
gibberish." A ceiling cannot be bought off — however good everything else is,
it cannot break through. That is the right semantics for a defect as opposed to
a preference.

**Ceilings introduce their own zero-gradient trap**, and it is not hypothetical:
early in training whole groups may carry the same defect, get flattened onto the
same ceiling, and lose within-group variance again. The alternative —
multiplicative capping (``cap * r``) — preserves ordering but turns "ceiling"
back into "discount". This module implements the hard ceiling and makes the
failure *observable* instead of guessing: ``uncapped_reward`` is retained, so a
monitor can measure how often a group is cap-flattened and we can switch on
evidence rather than on taste.

**A component that did not run is neither 0 nor 1.** Scoring it 0 makes every
image look broken; scoring it 1 makes every image look fine, which is the same
failure as a compliance gate reporting green because half of it was never wired.
Missing components are excluded and the remaining weights renormalised, with
``coverage`` reporting how much of the intended weight actually ran — and a hard
check that did not run makes ``deliverable`` *unknown*, never ``True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

COMPONENT_WEIGHTS: dict[str, float] = {
    "text_render": 0.40,
    "elements": 0.25,
    "layout": 0.20,
    "brand": 0.10,
    "aesthetic": 0.05,
}
"""Provisional. The *structure* is what this module fixes; these numbers are a
starting guess and are meant to move once T1.6's annotations and S2's baseline
say where the signal actually is. Aesthetic sits at 0.05 deliberately — it is a
monitor against collapse, and anything more invites the reward face."""

CAPS: dict[str, float] = {
    "gibberish_text": 0.15,
    "text_overflow": 0.20,
    "safe_zone": 0.25,
    "element_overlap": 0.30,
    "template_collapse": 0.35,
}
"""Ceiling imposed when the named defect is present."""

HARD_CHECKS: tuple[str, ...] = (
    "compliance",
    "text_exact",
    "elements_present",
    "layout_hard",
)
"""Every conjunct of `deliverable`. A name absent from the caller's mapping is
treated as *did not run* rather than *passed* — forgetting to wire one up must
not silently manufacture a shippable verdict."""


class RewardConfigError(ValueError):
    """Weights or caps are not a usable configuration."""


@dataclass(frozen=True, slots=True)
class ComponentScore:
    name: str
    weight: float
    score: float | None
    """None means the component did not run."""

    @property
    def ran(self) -> bool:
        return self.score is not None

    @property
    def contribution(self) -> float:
        """Weighted value before renormalisation."""
        return 0.0 if self.score is None else self.weight * self.score

    @property
    def deficit(self) -> float:
        """Weighted points lost. Sorting by this answers "what is costing the
        most", which is what the rework report buckets on."""
        return 0.0 if self.score is None else self.weight * (1.0 - self.score)


@dataclass(frozen=True, slots=True)
class CapHit:
    name: str
    ceiling: float


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    reward: float
    uncapped_reward: float
    deliverable: bool | None
    """None when a hard check did not run — unknown, never optimistically True."""

    components: tuple[ComponentScore, ...]
    caps_hit: tuple[CapHit, ...]
    hard_checks: tuple[tuple[str, bool | None], ...]
    gate_blocked: bool
    coverage: float
    """Fraction of the intended weight that actually ran. 1.0 once every layer
    is implemented; below that the reward is a partial view and says so."""

    @property
    def capped(self) -> bool:
        return bool(self.caps_hit)

    @property
    def binding_cap(self) -> CapHit | None:
        """The cap that actually set the ceiling, if any."""
        return min(self.caps_hit, key=lambda c: c.ceiling) if self.caps_hit else None

    @property
    def ran(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components if c.ran)

    def top_deficits(self, n: int = 3) -> tuple[ComponentScore, ...]:
        ordered = sorted(self.components, key=lambda c: c.deficit, reverse=True)
        return tuple(c for c in ordered[:n] if c.deficit > 0)

    def explain(self) -> str:
        if self.gate_blocked:
            return "reward 0.000 — compliance gate blocked"
        bits = [f"reward {self.reward:.3f}"]
        if self.capped:
            cap = self.binding_cap
            assert cap is not None
            bits.append(f"capped at {cap.ceiling:.2f} by {cap.name} (raw {self.uncapped_reward:.3f})")
        if self.coverage < 1.0:
            bits.append(f"coverage {self.coverage:.0%} — {', '.join(self.ran)} only")
        if self.deliverable is None:
            bits.append("deliverable unknown")
        elif not self.deliverable:
            failed = [n for n, ok in self.hard_checks if ok is False]
            bits.append(f"not deliverable ({', '.join(failed)})")
        if top := self.top_deficits(2):
            bits.append("costliest: " + ", ".join(f"{c.name} -{c.deficit:.3f}" for c in top))
        return " | ".join(bits)


@dataclass
class RewardAssembler:
    weights: Mapping[str, float] = None  # type: ignore[assignment]
    caps: Mapping[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = dict(COMPONENT_WEIGHTS)
        if self.caps is None:
            self.caps = dict(CAPS)

        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise RewardConfigError(f"component weights must sum to 1.0, got {total}")
        if any(w < 0 for w in self.weights.values()):
            raise RewardConfigError("component weights must be non-negative")
        if any(not 0.0 <= c <= 1.0 for c in self.caps.values()):
            raise RewardConfigError("cap ceilings must lie in [0, 1]")

    def assemble(
        self,
        scores: Mapping[str, float | None],
        hard_checks: Mapping[str, bool | None] | None = None,
        caps_triggered: Iterable[str] = (),
    ) -> RewardBreakdown:
        unknown = set(scores) - set(self.weights)
        if unknown:
            raise RewardConfigError(f"unknown reward components: {sorted(unknown)}")

        components = tuple(
            ComponentScore(name=name, weight=weight, score=_validate(scores.get(name), name))
            for name, weight in self.weights.items()
        )

        # Renormalise over what ran. Absent components are excluded rather than
        # defaulted, so a partially-wired verifier reports a smaller view of the
        # image rather than a confidently wrong score of it.
        ran_weight = sum(c.weight for c in components if c.ran)
        total_weight = sum(c.weight for c in components)
        base = (
            sum(c.contribution for c in components) / ran_weight if ran_weight > 0 else 0.0
        )
        coverage = ran_weight / total_weight if total_weight else 0.0

        triggered = list(dict.fromkeys(caps_triggered))
        unknown_caps = set(triggered) - set(self.caps)
        if unknown_caps:
            raise RewardConfigError(f"unknown caps: {sorted(unknown_caps)}")
        hits = tuple(CapHit(name=n, ceiling=self.caps[n]) for n in triggered)

        checks = dict.fromkeys(HARD_CHECKS, None)
        if hard_checks:
            unknown_checks = set(hard_checks) - set(HARD_CHECKS)
            if unknown_checks:
                raise RewardConfigError(f"unknown hard checks: {sorted(unknown_checks)}")
            checks.update(hard_checks)

        gate_blocked = checks["compliance"] is False

        # Order matters: the gate zeroes outright, and only then do ceilings
        # apply. A ceiling on top of zero would still be zero, but expressing it
        # this way keeps "blocked" distinguishable from "capped to near-zero".
        if gate_blocked:
            reward = 0.0
        elif hits:
            reward = min(base, min(h.ceiling for h in hits))
        else:
            reward = base

        if any(ok is None for ok in checks.values()):
            deliverable: bool | None = None
        else:
            deliverable = all(bool(ok) for ok in checks.values())

        return RewardBreakdown(
            reward=reward,
            uncapped_reward=base,
            deliverable=deliverable,
            components=components,
            caps_hit=hits,
            hard_checks=tuple(checks.items()),
            gate_blocked=gate_blocked,
            coverage=coverage,
        )


def _validate(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise RewardConfigError(f"component {name!r} scored {v}, must be in [0, 1]")
    return v
