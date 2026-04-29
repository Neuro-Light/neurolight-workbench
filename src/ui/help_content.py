"""Plain-language help text for key analysis metrics and plots.

Keep these short: users should understand the takeaway in a few seconds.
"""

from __future__ import annotations

from typing import Final

HELP_TEXT: Final[dict[str, str]] = {
    # Lomb–Scargle
    "lomb_scargle.plot": (
        "Lomb–Scargle periodogram\n\n"
        "Shows how strong different repeating cycle lengths are in the ROI intensity trace.\n"
        "A taller peak suggests a more prominent rhythm at that frequency/period.\n\n"
        "Tip: In Period mode, the x-axis is cycle length (minutes per cycle).\n"
        "In Frequency mode, it is cycles per minute."
    ),
    "lomb_scargle.summary": (
        "Peak readout\n\n"
        "f*: peak frequency (where the periodogram is highest)\n"
        "P(f*): power at the peak (higher usually means a stronger rhythm)\n"
        "T*: peak period (cycle length) = 1 / f*"
    ),
    "lomb_scargle.interval": (
        "Sampling interval\n\n"
        "Time between frames (in minutes). This sets the time scale for the periodogram.\n"
        "Use the same interval that was used to acquire the images."
    ),
    "lomb_scargle.axis_mode": (
        "X-axis mode\n\n"
        "Frequency: cycles per minute.\n"
        "Period: minutes per cycle (cycle length).\n\n"
        "Both views show the same information, just in different units."
    ),
    # Rayleigh / Rao
    "rayleigh.plot": (
        "Peak times on a 24-hour circle\n\n"
        "Each dot is a neuron. Its angle shows when that neuron reaches its peak activity (modulo 24 hours).\n"
        "If dots cluster at a particular time-of-day, peak times are more synchronized."
    ),
    "rayleigh.stats": (
        "Rayleigh test (circular uniformity)\n\n"
        "Tests whether peak times are spread evenly across the day or clustered.\n"
        "r: clustering strength (0 = spread out, 1 = tightly clustered)\n"
        "p: smaller values suggest the peaks are not uniform (i.e., more clustered than expected by chance)."
    ),
    "rao.stats": (
        "Rao's spacing test (circular uniformity)\n\n"
        "Another test for whether peak times are evenly spaced around the 24-hour circle.\n"
        "U: spacing statistic (larger values indicate less-uniform spacing)\n"
        "p: smaller values suggest non-uniform peak times."
    ),
}


def get_help_text(help_id: str) -> str:
    """Return help text for a given ID, or a fallback."""
    return HELP_TEXT.get(help_id, "Help is not available for this item yet.")
