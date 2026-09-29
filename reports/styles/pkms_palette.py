# Copied verbatim from the PKMS vault house style (~/Documents/PKMS/reports/styles/pkms_palette.py); do not edit here.
"""House palette shared between Beamer (LaTeX) and matplotlib figures.

Single source of truth for accent colors so figures match the deck chrome.
Keep these hexes in sync with ``common.tex`` and ``report.mplstyle``.
"""

from __future__ import annotations

from pathlib import Path

INK = "#1A2A3A"
BLUE = "#1565C0"
GREEN = "#2E7D32"
RED = "#C62828"
AMBER = "#F9A825"
GRAY = "#616161"
SURFACE = "#F5F5F5"

# Default categorical cycle (matches axes.prop_cycle in report.mplstyle).
PALETTE = [BLUE, GREEN, RED, AMBER, GRAY]

MPLSTYLE = str(Path(__file__).with_name("report.mplstyle"))


def use_style() -> list[str]:
    """Apply the shared matplotlib style; return the palette list."""
    import matplotlib.pyplot as plt

    plt.style.use(MPLSTYLE)
    return PALETTE
