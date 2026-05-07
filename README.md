# 🧠 Neurolight Workbench

[![CI](https://github.com/Neuro-Light/neurolight-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/Neuro-Light/neurolight-workbench/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![codecov](https://codecov.io/gh/Neuro-Light/neurolight-workbench/branch/main/graph/badge.svg)](https://codecov.io/gh/Neuro-Light/neurolight-workbench)

A powerful PySide6 desktop application for processing and analyzing large TIF image stacks with scientific rigor. Built for neuroscientists and researchers who need reproducible, shareable experiment workflows.

---

## ✨ Features

- 🔬 **Guided 6-Step Workflow** – Step-gated pipeline (Load → Edit → Align → ROI → Detect → Analyze) prevents out-of-order analysis
- 📸 **High-Volume Image Processing** – Handle 200+ TIF image stacks with LRU caching and folder/file drag-and-drop
- ✂️ **Image Culling Interface** – Mark and exclude poor-quality frames before alignment
- 🎨 **Contrast & Exposure Editing** – Per-frame brightness/contrast adjustment
- 📐 **Image Alignment** – Rigid-body and affine alignment via pystackreg, with optional multiprocessing
- 🎯 **Dual ROI Selection** – Draw and edit up to two polygon ROIs for side-by-side intensity comparison
- 🔭 **Neuron Detection** – Peak-based detection operating on cropped ROI regions with trajectory extraction
- 📊 **Rich Analysis Dashboard** – ROI Intensity, Neuron Trajectories, Lomb-Scargle Periodogram, and Rayleigh/Rao Circular Statistics
- 📈 **Toggleable Peak/Trough Markers** – Visualize signal extrema across all plot types
- 🕐 **Time-Unit X-Axis** – Display plots in frames or real-world minutes based on acquisition settings
- 📤 **CSV Export** – Export peak/trough timing tables from trajectory and intensity plots
- 💾 **Auto-Save** – Never lose your work with periodic session saving
- 🤝 **Collaboration Ready** – Share experiments as portable `.nexp` JSON files

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Install uv** (if not already installed)

   ```bash
   # Windows
   pip install uv

   # macOS
   brew install uv
   ```

   Or via the official installer:

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

   Or, after `uv pip install -e .`:

   ```bash
   uv run neurolight
   ```

For a step-by-step walkthrough with sample data, see [GETTING_STARTED/GETTING_STARTED.md](GETTING_STARTED/GETTING_STARTED.md).

---

## 📁 Project Structure

```
neurolight-workbench/
│
├── src/
│   ├── main.py                    # Application entry point
│   ├── core/                      # Domain logic and numerics (no Qt)
│   │   ├── experiment_manager.py  # Experiment session handling (.nexp files)
│   │   ├── image_processor.py     # Image processing + neuron detection pipeline
│   │   ├── alignment_mp.py        # Multiprocessing-safe alignment workers
│   │   ├── lomb_scargle.py        # Lomb-Scargle periodogram computation
│   │   ├── circular_stats.py      # Rayleigh + Rao circular uniformity tests
│   │   ├── roi.py                 # ROI geometry and mask utilities
│   │   ├── data_analyzer.py       # ROI intensity extraction helpers
│   │   └── gif_generator.py       # Animated GIF export
│   │
│   ├── ui/                        # PySide6 UI components
│   │   ├── main_window.py         # Main application window and orchestration
│   │   ├── workflow.py            # Guided workflow stepper + step gating
│   │   ├── image_viewer.py        # Image display, navigation, ROI tools, culling
│   │   ├── analysis_panel.py      # Tabbed analysis panel container
│   │   ├── roi_intensity_plot.py  # ROI mean intensity time series plot
│   │   ├── neuron_trajectory_plot.py # Per-neuron intensity trajectory plots
│   │   ├── lomb_scargle_plot.py   # Lomb-Scargle periodogram widget
│   │   ├── rayleigh_plot.py       # Rayleigh/Rao circular statistics plot
│   │   ├── neuron_detection_widget.py # Detection UI and results
│   │   ├── alignment_worker.py    # QThread alignment worker
│   │   ├── startup_dialog.py      # Experiment selection/creation screen
│   │   ├── app_settings.py        # App settings persistence
│   │   └── styles.py              # App theme + Matplotlib theme helpers
│   │
│   └── utils/
│       ├── file_handler.py        # TIF stack I/O
│       └── image_utils.py         # NumPy-to-QImage conversion
│
├── tests/                         # Pytest test suite
├── .github/workflows/             # CI/CD (lint, test, build, release)
├── neurolight.spec                # PyInstaller build spec
└── build-macos-*.sh               # macOS packaging scripts
```

### 🔧 Key Module Responsibilities

| Module | Purpose |
| --- | --- |
| **`experiment_manager.py`** | Create, load, save `.nexp` experiments; manage recent experiments list |
| **`image_processor.py`** | Image preprocessing, ROI cropping/masking, neuron detection pipeline |
| **`alignment_mp.py`** | Multiprocessing-safe pystackreg wrappers (isolated from Qt) |
| **`lomb_scargle.py`** | Lomb-Scargle periodogram via scipy; handles unevenly-sampled data |
| **`circular_stats.py`** | Rayleigh test and Rao spacing test for peak-time circular statistics |
| **`roi.py`** | `ROI` geometry (polygon/ellipse), mask generation, handle types |
| **`file_handler.py`** | Load/validate TIF stacks; frame access; EXIF timestamp extraction |
| **`workflow.py`** | `WorkflowStepper` — 6-step pipeline UI with step gating and status |
| **`analysis_panel.py`** | Tabs: Detection, ROI Intensity, Trajectories, Lomb-Scargle, Rayleigh/Rao |

---

## 🔬 Experiment Workflow

### Guided 6-Step Pipeline

| Step | Name | What happens |
| --- | --- | --- |
| 1 | **Load Image Stack** | Open TIF files or a folder via drag-and-drop or `File → Open Image Stack` |
| 2 | **Edit Contrast & Exposure** | Adjust brightness/contrast and cull poor-quality frames |
| 3 | **Align Images** | Run rigid-body or affine alignment; save aligned stack |
| 4 | **Select ROI** | Draw one or two polygon ROIs on the aligned images |
| 5 | **Detect Neurons** | Run peak-based neuron detection within the ROI |
| 6 | **Analyze Graphs** | Explore ROI intensity, trajectories, Lomb-Scargle, and Rayleigh/Rao plots |

### What is an Experiment?

An **experiment** is a complete research session stored as a `.nexp` JSON file containing:

- 📋 Metadata (name, description, principal investigator, acquisition start time and interval)
- 🖼️ Image stack information (path, dimensions, bit depth)
- ⚙️ Processing history (all operations and parameters)
- 🎯 ROI definitions (up to two polygon ROIs)
- 📈 Analysis results

### Sharing Experiments

1. Export the `.nexp` file from `users/<username>/experiments/`
2. Include the referenced image stack folder
3. Colleagues can load the experiment and reproduce your entire workflow

---

## 📖 Usage Guide

### Starting the Application

1. Application opens to the **Startup Dialog**
2. Choose your path:
   - 🆕 **Start New Experiment** – Enter metadata and create a fresh session
   - 📂 **Load Existing Experiment** – Browse for an existing `.nexp` file
   - 🕒 **Recent Experiments** – Quick access to your last experiments

### Main Application Window

**Left panel** – Image Viewer: drag-and-drop TIF files/folders, navigate frames, draw ROIs, and cull poor-quality frames.

**Right panel** – Analysis Dashboard:

| Tab | Content |
| --- | --- |
| **Detection** | Neuron detection parameters, run detection, view results |
| **ROI Intensity** | Mean intensity time series for ROI 1 and ROI 2 |
| **Trajectories** | Per-neuron intensity traces with peak/trough markers and CSV export |
| **Lomb–Scargle** | Periodogram for ROI intensity signals |
| **Rayleigh/Rao** | Per-day circular polar plots with statistical test results |

**Top bar** – Workflow Stepper: displays the 6 pipeline steps and locks downstream steps until prerequisites are complete.

---

## 🏗️ Architecture

### Design Principles

- **🧩 Modularity** – Independent, replaceable components
- **🔌 Extensibility** – Clear interfaces for adding new features
- **💼 Session Management** – All actions tied to experiment context
- **⚡ Performance** – Lazy loading, background threads, LRU cache, ROI-cropped detection
- **🛡️ Error Handling** – Graceful failures with user-friendly messages

### Application Flow

```
Launch → Startup Dialog → Create/Load Experiment → Main Window
  └─► WorkflowStepper gates each pipeline step
       └─► Auto-Save Loop (configurable interval)
```

---

## 🧪 Testing

```bash
uv sync --all-extras
uv run pytest tests/
```

With coverage:

```bash
uv run pytest tests/ -v --cov=src --cov-branch --cov-report=xml
```

The suite covers core algorithms, UI widgets (headless via `QT_QPA_PLATFORM=offscreen`), alignment/detection workers, and experiment persistence.

---

## 🚧 Future Roadmap

- 🔄 Experiment versioning and history
- ☁️ Cloud storage integration
- 📤 Export to standardized formats (HDF5, OME-TIFF)
- 🤝 Multi-user experiment comparison tools
- ⚙️ Batch processing across multiple experiments

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and the pull request process.

---

## 📝 License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE).

---

## 🙏 Acknowledgments

Built with:

- [PySide6](https://doc.qt.io/qtforpython/) – Qt for Python
- [OpenCV](https://opencv.org/) – Computer vision
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) – Scientific computing
- [Matplotlib](https://matplotlib.org/) – Plotting and visualization
- [pystackreg](https://github.com/glichtner/pystackreg) – Image stack alignment
- [tifffile](https://github.com/cgohlke/tifffile) / [scikit-image](https://scikit-image.org/) – TIFF I/O and image utilities
- [statsmodels](https://www.statsmodels.org/) / [imageio](https://imageio.readthedocs.io/) – Statistics and GIF export
- [Pillow](https://python-pillow.org/) – Image format support
- [uv](https://github.com/astral-sh/uv) – Package and environment management

---

<div align="center">

**Made with 🧠 for neuroscience research**

</div>
