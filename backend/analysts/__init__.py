# analysts/__init__.py
"""
Analysts package - Risk metric analysis implementations.

Each analyst extends the abstract Analyst base class and provides
classical and quantum analysis methods.
"""

from typing import Dict

from .base import Analyst, AnalysisParams, ClassicalResult, QuantumResult
from .var import VaRAnalyst
from .cvar import CVaRAnalyst
from .rvar import RVaRAnalyst


# Registry for easy access to analysts by metric name
ANALYSTS: Dict[str, Analyst] = {
    "VaR": VaRAnalyst(),
    "CVaR": CVaRAnalyst(),
    "RVaR": RVaRAnalyst(),
}


def get_analyst(metric: str) -> Analyst:
    """Get an analyst by metric name."""
    if metric not in ANALYSTS:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(ANALYSTS.keys())}")
    return ANALYSTS[metric]


__all__ = [
    "Analyst",
    "AnalysisParams",
    "ClassicalResult",
    "QuantumResult",
    "VaRAnalyst",
    "CVaRAnalyst",
    "RVaRAnalyst",
    "ANALYSTS",
    "get_analyst",
]
