"""Bubble models for rare quantum fluctuations (toy models).

This subpackage provides upper bounds on complexity, rarity models,
phenomenological dynamics, experiment sweeps, plotting, and reporting
utilities. The models are deliberately simplified and are not intended
to describe real cosmology.
"""

from . import constants, bounds, rarity, dynamics, experiments, plot, report

__all__ = [
    "constants",
    "bounds",
    "rarity",
    "dynamics",
    "experiments",
    "plot",
    "report",
]
