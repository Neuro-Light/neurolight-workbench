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
    "lomb_scargle.roi_toggles": (
        "ROI selectors\n\n"
        "Choose which ROI intensity traces to include in the periodogram.\n"
        "The color swatch matches the trace color used in other plots."
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
    # ROI Intensity
    "roi_intensity.plot": (
        "ROI Intensity Over Time\n\n"
        "Plots the average pixel intensity inside each selected ROI across frames.\n"
        "Use this to see how the ROI signal changes over the experiment.\n\n"
        "Tip: The x-axis is time (minutes) based on the experiment's frame interval."
    ),
    "roi_intensity.roi_toggles": (
        "ROI toggles\n\n"
        "Show or hide each ROI trace without deleting the ROI.\n"
        "This is useful when comparing one ROI at a time."
    ),
    "roi_intensity.peaks": (
        "Peaks / troughs markers\n\n"
        "Marks local maxima (peaks) and minima (troughs) in the ROI intensity trace.\n"
        "This helps you visually spot cycles and estimate timing between events.\n\n"
        "Note: markers are approximate and depend on signal noise and sampling."
    ),
    "roi_intensity.hover": (
        "Hover / selection readout\n\n"
        "Move the mouse over the plot to see the time and current intensity values.\n"
        "If peaks/troughs are enabled, hovering near a marker shows its time/value and spacing."
    ),
    # Neuron trajectories
    "neuron_trajectories.plot": (
        "Neuron intensity trajectories\n\n"
        "Shows per-neuron intensity over time (many thin lines) plus an optional average.\n"
        "Use ROI filtering and the Good/Bad toggles to focus on subsets."
    ),
    "neuron_trajectories.roi_filter": (
        "ROI filter\n\n"
        "If detection was run on both ROIs, this filters trajectories by which ROI each neuron belongs to."
    ),
    "neuron_trajectories.good_bad_avg": (
        "Good / Bad / Avg\n\n"
        "Good/Bad: show neurons classified by detection quality.\n"
        "Avg: overlay the mean trajectory of the currently displayed set."
    ),
    "neuron_trajectories.max_neurons": (
        "Max neurons\n\n"
        "Limits how many neuron traces are drawn to keep the plot responsive.\n"
        "Increasing this shows more detail but may slow rendering."
    ),
    "neuron_trajectories.smoothing": (
        "Smoothing (display only)\n\n"
        "Applies a moving-average to the displayed curves to reduce noise.\n"
        "Exported data is always the raw (unsmoothed) trajectories."
    ),
    "neuron_trajectories.peaks": (
        "Peaks / troughs on the average\n\n"
        "Marks peaks and troughs on the average trajectory to help spot cycles.\n"
        "Use 'Numbers' to label marker order over time."
    ),
    "neuron_trajectories.hover": (
        "Hover / selection readout\n\n"
        "Hover over the plot to see elapsed time and the mean intensity at that time.\n"
        "Clicking a peak/trough marker shows its exact values and spacing."
    ),
    # Neuron detection visualization
    "neuron_detection.plot": (
        "Detected neurons overlay\n\n"
        "Shows detected neuron locations (green = good, red = bad) over the first frame (or mean frame).\n"
        "This helps validate whether detection found plausible neuron positions inside the ROI."
    ),
    "neuron_detection.params": (
        "Detection parameters\n\n"
        "These settings control how neuron candidates are found and filtered.\n"
        "If you see too many false positives, raise thresholds; if you miss neurons, lower them."
    ),
    "neuron_detection.param.cell_size": (
        "Cell size (pixels)\n\n"
        "Approximate neuron diameter in pixels.\n"
        "Larger values favor larger cells; smaller values help find small cells but may increase false positives."
    ),
    "neuron_detection.param.max_neurons": (
        "Max neurons\n\n"
        "Upper limit on how many neuron candidates the detector will return.\n"
        "Lower this if detection is slow or you only need the strongest candidates."
    ),
    "neuron_detection.param.correlation_threshold": (
        "Correlation threshold\n\n"
        "Quality filter for detected neurons based on how consistent their signal is over time.\n"
        "Higher values are stricter (fewer neurons, higher confidence); lower values include more but may add noise."
    ),
    "neuron_detection.param.max_absent_frames": (
        "Max absent frames\n\n"
        "Marks a neuron as bad if its extracted intensity is zero for more than this many frames.\n"
        "Useful for filtering unstable detections or neurons that disappear due to motion/segmentation issues."
    ),
    "neuron_detection.param.peak_threshold": (
        "Peak threshold\n\n"
        "Relative threshold used when finding neuron candidates.\n"
        "Lower values can detect dimmer neurons but may increase false positives."
    ),
    "neuron_detection.param.max_projection": (
        "Max projection\n\n"
        "If enabled, detection uses the maximum intensity over time (good for flashing activity).\n"
        "If disabled, it uses an average-like view, which can be better for steady signals."
    ),
    "neuron_detection.param.smoothing_sigma": (
        "Smoothing (sigma)\n\n"
        "Gaussian blur applied before candidate detection.\n"
        "Higher sigma reduces noise and can help find dim peaks, but may merge nearby cells."
    ),
    "neuron_detection.param.detrending": (
        "Apply detrending\n\n"
        "Removes slow baseline drift in the intensity traces to emphasize rhythmic changes.\n"
        "Disable if you need the raw trend preserved."
    ),
}


def get_help_text(help_id: str) -> str:
    """Return help text for a given ID, or a fallback."""
    return HELP_TEXT.get(help_id, "Help is not available for this item yet.")
