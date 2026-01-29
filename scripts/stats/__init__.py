"""
Statistics module for corpus analysis.
Computes statistics from raw XML files and processed TSV files.
"""

from .raw_stats import compute_raw_stats
from .processed_stats import compute_processed_stats
from .display import (
    display_raw_stats,
    display_processed_stats,
    display_comparison
)

__all__ = [
    'compute_raw_stats',
    'compute_processed_stats',
    'display_raw_stats',
    'display_processed_stats',
    'display_comparison'
]

__version__ = '1.0.0'