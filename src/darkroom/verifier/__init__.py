"""Verifier layers. L1 gate / L2 text / L3 layout / L4 brand."""

from .l1_compliance import (  # noqa: F401
    ComplianceGate,
    ComplianceHit,
    ComplianceVerdict,
    normalize,
)
from .l3_layout import (  # noqa: F401
    Box,
    LayoutChecker,
    LayoutVerdict,
    LayoutViolation,
    union_area,
)
from .reward import (  # noqa: F401
    CAPS,
    COMPONENT_WEIGHTS,
    HARD_CHECKS,
    CapHit,
    ComponentScore,
    RewardAssembler,
    RewardBreakdown,
    RewardConfigError,
)
