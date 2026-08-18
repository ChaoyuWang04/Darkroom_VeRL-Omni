"""Exact-key lookup over the YAML spec tree.

Two rules govern this module, both from docs/darkroom-project-design-v0.1.md §2:

1. **Zero recall error.** Lookups are exact. A miss raises; it never degrades to
   a nearest match. Returning PROD_B's rules for a PROD_A query is worse than
   returning nothing, because nothing is visible and wrong is not. Acceptance
   A1.8 pins this at 1.00.

2. **Expiry is enforced, not advisory.** Platform rules move. A spec past its
   `valid_to` raises by default, so a stale rule cannot quietly become training
   signal. Pass `strict_expiry=False` only when you know why.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .models import (
    BannedTerm,
    BrandSpec,
    ContentPolicy,
    LogoRule,
    PlacementSpec,
    SafeZone,
    SpecExpired,
    SpecInvalid,
    SpecNotFound,
)

DEFAULT_SPEC_ROOT = Path(__file__).resolve().parents[3] / "specs"


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise SpecInvalid(f"{where}: missing required field {key!r}")
    return mapping[key]


def _as_date(value: Any, where: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise SpecInvalid(f"{where}: {value!r} is not an ISO date") from exc


class SpecRegistry:
    """Loads the spec tree once, then answers exact-key questions."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_SPEC_ROOT
        if not self.root.is_dir():
            raise SpecInvalid(f"spec root {self.root} does not exist")
        self._placements: dict[tuple[str, str], PlacementSpec] = {}
        self._policies: dict[str, ContentPolicy] = {}
        self._brands: dict[str, BrandSpec] = {}
        self._load()

    # ---------------------------------------------------------------- loading

    def _load(self) -> None:
        self._load_placements(self.root / "placements")
        self._load_policies(self.root / "policies")
        self._load_brands(self.root / "brands")

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        if not isinstance(doc, dict):
            raise SpecInvalid(f"{path}: top level must be a mapping")
        return doc

    def _load_placements(self, folder: Path) -> None:
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("*.yaml")):
            doc = self._read(path)
            platform = _require(doc, "platform", str(path))
            source = _require(doc, "source", str(path))
            version = str(_require(doc, "version", str(path)))
            valid_to = _as_date(_require(doc, "valid_to", str(path)), str(path))

            for entry in _require(doc, "sizes", str(path)):
                where = f"{path}:{entry.get('size', '?')}"
                width, height = (int(v) for v in _require(entry, "size", where).split("x"))
                spec = PlacementSpec(
                    platform=platform,
                    size=entry["size"],
                    width=width,
                    height=height,
                    safe_zone=SafeZone(**entry.get("safe_zone", {})),
                    text_max_ratio=float(_require(entry, "text_max_ratio", where)),
                    text_ratio_severity=entry.get("text_ratio_severity", "soft"),
                    required_elements=tuple(_require(entry, "required_elements", where)),
                    logo=LogoRule(**entry.get("logo", {})),
                    source=source,
                    version=version,
                    valid_to=valid_to,
                )
                if spec.key in self._placements:
                    raise SpecInvalid(f"{where}: duplicate placement {spec.key}")
                self._placements[spec.key] = spec

    def _load_policies(self, folder: Path) -> None:
        """Load region policies, resolving `extends` so a shared rule lives in
        exactly one file. Without composition every region would carry its own
        copy of the baseline blocklist, and a rule change would need N edits —
        which is how blocklists drift out of agreement with each other."""
        if not folder.is_dir():
            return

        raw: dict[str, tuple[Path, dict[str, Any]]] = {}
        for path in sorted(folder.glob("*.yaml")):
            doc = self._read(path)
            region = _require(doc, "region", str(path))
            if region in raw:
                raise SpecInvalid(f"{path}: duplicate policy region {region!r}")
            raw[region] = (path, doc)

        resolving: set[str] = set()

        def resolve(region: str) -> ContentPolicy:
            if region in self._policies:
                return self._policies[region]
            if region in resolving:
                raise SpecInvalid(f"policy {region!r} is part of an `extends` cycle")
            if region not in raw:
                raise SpecInvalid(f"policy {region!r} extends an unknown region")
            resolving.add(region)
            path, doc = raw[region]

            terms: list[BannedTerm] = []
            visual: list[str] = []
            parent_name = doc.get("extends")
            if parent_name:
                parent = resolve(str(parent_name))
                terms.extend(parent.terms)
                visual.extend(parent.banned_visual_categories)

            own = {
                BannedTerm(
                    term=_require(t, "term", str(path)),
                    category=_require(t, "category", str(path)),
                    evasion=bool(t.get("evasion", False)),
                    note=t.get("note", ""),
                )
                for t in doc.get("terms", [])
            }
            inherited = {t.term.casefold() for t in terms}
            # A region may restate an inherited term to tighten it (e.g. turn on
            # evasion); the child's version wins, and the duplicate is dropped
            # so ContentPolicy's uniqueness check still holds.
            terms = [t for t in terms if t.term.casefold() not in {o.term.casefold() for o in own}]
            terms.extend(sorted(own, key=lambda t: t.term))
            del inherited

            visual.extend(doc.get("banned_visual_categories", []))

            policy = ContentPolicy(
                region=region,
                terms=tuple(terms),
                banned_visual_categories=tuple(dict.fromkeys(visual)),
                source=_require(doc, "source", str(path)),
                version=str(_require(doc, "version", str(path))),
                valid_to=_as_date(_require(doc, "valid_to", str(path)), str(path)),
            )
            resolving.discard(region)
            self._policies[region] = policy
            return policy

        for region in raw:
            resolve(region)

    def _load_brands(self, folder: Path) -> None:
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("*.yaml")):
            doc = self._read(path)
            brand = BrandSpec(
                brand_id=_require(doc, "brand_id", str(path)),
                primary_hex=_require(doc, "primary_hex", str(path)),
                delta_e_max=float(_require(doc, "delta_e_max", str(path))),
                font_family=_require(doc, "font_family", str(path)),
                logo_asset=doc.get("logo_asset", ""),
            )
            if brand.brand_id in self._brands:
                raise SpecInvalid(f"{path}: duplicate brand {brand.brand_id!r}")
            self._brands[brand.brand_id] = brand

    # ---------------------------------------------------------------- lookups

    @staticmethod
    def _check_expiry(spec: Any, label: str, as_of: date | None, strict: bool) -> None:
        if not strict or as_of is None:
            return
        if spec.valid_to < as_of:
            raise SpecExpired(
                f"{label} expired on {spec.valid_to} (asked as of {as_of}); "
                "re-verify the upstream rules and bump the record"
            )

    def get_placement(
        self,
        platform: str,
        size: str,
        *,
        as_of: date | None = None,
        strict_expiry: bool = True,
    ) -> PlacementSpec:
        try:
            spec = self._placements[(platform, size)]
        except KeyError:
            raise SpecNotFound(
                f"no placement spec for platform={platform!r} size={size!r}; "
                f"known: {sorted(self._placements)}"
            ) from None
        self._check_expiry(spec, f"placement {platform}/{size}", as_of, strict_expiry)
        return spec

    def get_policy(
        self,
        region: str,
        *,
        as_of: date | None = None,
        strict_expiry: bool = True,
    ) -> ContentPolicy:
        try:
            policy = self._policies[region]
        except KeyError:
            raise SpecNotFound(
                f"no content policy for region={region!r}; known: {sorted(self._policies)}"
            ) from None
        self._check_expiry(policy, f"policy {region}", as_of, strict_expiry)
        return policy

    def get_brand(self, brand_id: str) -> BrandSpec:
        try:
            return self._brands[brand_id]
        except KeyError:
            raise SpecNotFound(
                f"no brand spec for {brand_id!r}; known: {sorted(self._brands)}"
            ) from None

    # ------------------------------------------------------------- inventory

    @property
    def placements(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._placements))

    @property
    def regions(self) -> tuple[str, ...]:
        return tuple(sorted(self._policies))

    @property
    def brands(self) -> tuple[str, ...]:
        return tuple(sorted(self._brands))

    def __repr__(self) -> str:
        return (
            f"SpecRegistry(root={self.root}, placements={len(self._placements)}, "
            f"policies={len(self._policies)}, brands={len(self._brands)})"
        )
