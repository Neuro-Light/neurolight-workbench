"""Shared UI constants for UI widgets."""

from __future__ import annotations

ROI_KEYS = ("roi_1", "roi_2")
ROI_DISPLAY_NAMES = {"roi_1": "ROI 1", "roi_2": "ROI 2"}

# Default time between frames (in minutes) used by widgets that
# interpret frame indices as time-of-day (e.g. Rayleigh and Lomb–Scargle).
DEFAULT_FRAME_INTERVAL_MINUTES = 30
