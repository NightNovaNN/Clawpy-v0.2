# MIT License © 2025 ISD NightNova
"""
ClawPy v0.2 — A high-performance scientific, mathematical, AI, and computational toolkit.

Included modules:
- BasicMath
- AppliedMath
- AdvancedMath
- Calculus
- AI
- LeftMath
- MoreMath
- OtherMath
- ScientificMath
"""

from .basic_math import BasicMath
from .applied_math import AppliedMath
from .advanced_math import AdvancedMath
from .calculus import Calculus
from .ai import AI
from .left import LeftMath
from .more import MoreMath
from .other import OtherMath
from .scientific import ScientificMath

__version__ = "0.2"

__all__ = [
    "BasicMath",
    "AppliedMath",
    "AdvancedMath",
    "Calculus",
    "AI",
    "LeftMath",
    "MoreMath",
    "OtherMath",
    "ScientificMath",
    "__version__",
]
