"""Fine-tuning tasks for BCRFoundation."""

from . import classification
from . import integration
from . import specificity
from . import cross_modal
from . import cross_modal_zeroshot

__all__ = [
    "classification",
    "integration",
    "specificity",
    "cross_modal",
    "cross_modal_zeroshot",
]
