"""
GeoMACA: Geographic Multi-Agent Credit Assignment

A novel credit assignment method that decomposes end-of-episode rewards across:
1. Geographic zones (which location?)
2. Time steps (which turn?)
3. Agents (which government agent?)

This is specifically designed for spatially-structured multi-agent environments
like urban planning simulations.
"""

from .model import GeoMACA, GeoMACAConfig
from .losses import GeoMACALosses
from .visualization import visualize_credit_assignment

__all__ = [
    'GeoMACA',
    'GeoMACAConfig',
    'GeoMACALosses',
    'visualize_credit_assignment',
]

