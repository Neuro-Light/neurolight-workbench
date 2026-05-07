# 🧠 NeuroLight Workbench

[![CI](https://github.com/Neuro-Light/neurolight-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/Neuro-Light/neurolight-workbench/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Neuro-Light/neurolight-workbench/branch/main/graph/badge.svg)](https://codecov.io/gh/Neuro-Light/neurolight-workbench)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)

A PySide6 desktop application for automated circadian rhythm and neuron activity analysis. Built for neuroscience researchers, NeuroLight provides a reproducible, experiment-centric workflow for analyzing fluorescence image stacks of suprachiasmatic nucleus (SCN) tissue.

Sponsored by Dr. Allen (OHSU) and Dr. Doerry (NAU). Developed by team NeuroNauts as a capstone project.

---

## Key Features

- **Guided 6-Step Workflow** — Step-gated pipeline (Load → Edit → Align → ROI → Detect → Analyze) prevents out-of-order analysis
- **Dual SCN ROI Selection** — Define two independent regions of interest over SCN tissue for side-by-side circadian analysis
- **Image Culling Interface** — Mark and exclude poor-quality frames before alignment using a bracket-selection tool
- **Contrast & Exposure Editing** — Per-frame brightness/contrast adjustment so structures of interest are clearly visible
- **Image Alignment** — PyStackReg corrects for drift across image stacks before analysis
- **Automated Neuron Detection** — Adaptive thresholding via OpenCV identifies active neurons without manual marking
- **Circadian Period Estimation** — Lomb-Scargle periodogram extracts period from non-uniformly sampled fluorescence data
- **Rhythm Significance Testing** — Rayleigh and Rao circular statistics assess whether detected rhythms are statistically meaningful
- **Toggleable Peak/Trough Markers** — Visualize signal extrema across ROI intensity and trajectory plots
- **Time-Unit X-Axis** — Display plots in frames or real-world minutes based on acquisition start time and interval
- **EXIF-Based Timestamps** — Automatically extracts capture times from image metadata for accurate time-axis labeling
- **User Profiles** — Per-user login system for managing separate experiment sessions
- **CSV Export** — Export neuron peak and trough data for downstream analysis

---

## Architecture

NeuroLight follows a **Model-View-Presenter (MVP)** pattern with an **EventBus** for decoupled communication between components. UI presenters publish and subscribe to events rather than calling each other directly, keeping the analysis pipeline and interface layers independent.

### Analysis Pipeline

```
Load TIF Stack
     |
     v
Image Culling (exclude poor-quality frames)
     |
     v
Image Alignment (PyStackReg)
     |
     v
SCN ROI Selection (dual regions)
     |
     v
Neuron Detection (OpenCV adaptive thresholding)
     |
     v
Intensity Extraction per neuron per frame
     |
     v
EXIF Timestamp Extraction → time axis
     |
     v
Lomb-Scargle Periodogram → period estimate
     |
     v
Rayleigh + Rao Circular Statistics → significance
     |
     v
Plots + CSV Export
```

### Project Structure

```
neurolight-workbench/
├── src/
│   ├── main.py                    # Entry point
│   ├── core/
│   │   ├── image_processor.py     # Alignment + neuron detection pipeline
│   │   ├── alignment_mp.py        # Multiprocessing-safe alignment workers
│   │   ├── lomb_scargle.py        # Lomb-Scargle periodogram computation
│   │   ├── circular_stats.py      # Rayleigh + Rao circular statistics
│   │   ├── roi.py                 # ROI geometry and mask utilities
│   │   ├── data_analyzer.py       # Intensity extraction helpers
│   │   └── experiment_manager.py  # Session persistence (.nexp files)
│   ├── ui/
│   │   ├── main_window.py
│   │   ├── workflow.py            # Guided workflow stepper + step gating
│   │   ├── image_viewer.py        # Image navigation + ROI drawing + culling
│   │   ├── neuron_detection_widget.py
│   │   ├── roi_selection_dialog.py
│   │   ├── alignment_dialog.py
│   │   ├── rayleigh_plot.py
│   │   ├── lomb_scargle_plot.py
│   │   ├── roi_intensity_plot.py
│   │   ├── neuron_trajectory_plot.py
│   │   └── ...
│   └── utils/
│       ├── file_handler.py        # TIF stack I/O
│       └── image_utils.py         # NumPy-to-QImage conversion
├── tests/                         # pytest test suite (75%+ coverage)
├── .github/workflows/             # CI (tests + coverage) + CD (macOS build)
├── pyproject.toml
└── uv.lock
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI framework | PySide6 (Qt for Python) |
| Image processing | OpenCV, scikit-image |
| Image alignment | PyStackReg |
| Scientific computing | NumPy, SciPy, statsmodels |
| Circular statistics | SciPy (custom wrappers in `circular_stats.py`) |
| Periodogram | SciPy Lomb-Scargle |
| Plotting | Matplotlib |
| TIFF I/O | tifffile, Pillow |
| Testing | pytest + Codecov |
| Packaging | macOS code signing + notarization via GitHub Actions |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Install uv** (if not already installed)

   **Option A: Using pip**

   ```bash
   # Windows
   pip install uv

   # macOS
   brew install uv
   ```

   **Option B: Official installer**

   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**

   ```bash
   git clone https://github.com/Neuro-Light/neurolight-workbench
   cd neurolight-workbench
   ```

3. **Install dependencies**

   ```bash
   uv sync
   ```

4. **Launch the application**

   ```bash
   uv run python src/main.py
   ```

   Or activate the virtual environment manually:

   ```bash
   # macOS/Linux
   source .venv/bin/activate
   python src/main.py

   # Windows
   .venv\Scripts\activate
   python src/main.py
   ```

---

## 🧪 Testing with SCN Images

The **[GETTING_STARTED/](GETTING_STARTED/)** folder contains 50 sample SCN fluorescence images and a walkthrough for exploring the full analysis pipeline — ROI selection, neuron detection, periodogram, and circular statistics.

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request for significant changes. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE). You may choose either license.

---

## Acknowledgments

Sponsored by Dr. Allen (OHSU) and Dr. Doerry (NAU).

Built with [PySide6](https://doc.qt.io/qtforpython/), [OpenCV](https://opencv.org/), [PyStackReg](https://github.com/glichtner/pystackreg), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [Matplotlib](https://matplotlib.org/).
