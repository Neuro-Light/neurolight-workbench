# Build & Runtime Optimizations (macOS)

This document summarizes the optimizations implemented so far to improve:

- **Build correctness** (producing a real macOS `.app` and a valid installer `.pkg`)
- **Runtime reliability** (especially in PyInstaller-frozen builds)
- **Performance** (faster stack loading and lower peak memory during detection/alignment)
- **Safe parallelism** (guarded multiprocessing in a frozen GUI app)

The changes are intentionally conservative: defaults remain stable, with “risky” speedups gated behind environment variables.

---

## Goals and constraints

- **Target**: macOS app distributed via PyInstaller.
- **UI framework**: PySide6/Qt.
- **Image stacks**: primarily `.tif/.tiff`, sometimes `.gif`.
- **Frozen app constraints**:
  - Multiprocessing uses **`spawn`** on macOS.
  - Spawned workers can accidentally re-import GUI modules if not carefully separated, which can cause crashes/odd behavior.
  - Large intermediate NumPy arrays can trigger **OS-level termination** (no Python exception) due to memory pressure.

---

## 1) Packaging correctness: produce a real `.app` bundle

### Problem
The build script `build-macos-app.sh` expected `dist/Neurolight.app`, but the original `neurolight.spec` produced a **single executable** in `dist/` instead of a proper `.app` bundle. This mismatch often leads to downstream “installer” failures or confusing artifacts.

### Fix
Updated `neurolight.spec` to build a macOS bundle via:

- `EXE(..., exclude_binaries=True, ...)`
- `COLLECT(...)` (onedir layout)
- `BUNDLE(...)` producing `Neurolight.app`

This also resolves PyInstaller warnings about “onefile + .app bundles” and aligns the output with what macOS expects for GUI apps.

### Resulting outputs
- `dist/Neurolight.app` (launchable via Finder or `open`)
- `dist/Neurolight/` (onedir payload used by the bundle)

---

## 2) Installer correctness: generate a proper `.pkg`

### Problem
PyInstaller’s internal “PKG” step (`*.pkg` shown in logs) is **not** a macOS Installer package. It’s an internal archive used by the bootloader. Attempting to open it as an installer can cause Installer errors like:

- `com.apple.installer.pagecontroller error -1`

### Fix
Added a dedicated packaging script:

- `build-macos-pkg.sh`

It:
1. Stages `dist/Neurolight.app` into `build/pkgroot/Applications/Neurolight.app`
2. Runs `pkgbuild` to create a component package
3. Runs `productbuild` to create a product archive:
   - `dist/Neurolight-unsigned.pkg`

### Notes
- The produced pkg is **unsigned** and **not notarized**. It is suitable for local testing and development distribution.
- For broad distribution to other Macs, you typically need:
  - Developer ID signing
  - Notarization

---

## 3) Frozen build reliability: ensure Pillow loads TIFF/GIF plugins

### Problem
The viewer loads images via Pillow (`PIL.Image.open`). In frozen apps, Pillow’s format handlers are often imported dynamically (plugins), and PyInstaller can miss them. That manifests as:

- “images won’t load” even though the file picker works and permissions are granted

### Fix
Updated `neurolight.spec` to bundle Pillow’s plugins explicitly:

- Added `collect_submodules("PIL")`
- Explicitly included:
  - `PIL.TiffImagePlugin`
  - `PIL.GifImagePlugin`
  - `PIL.ImageSequence`
  - `PIL.ImageQt`
- Included `tifffile` as well (used elsewhere in the app for TIFF loading/alignment)

### Additional UX hardening
Updated `src/ui/image_viewer.py` to show a clear error dialog if an image fails to open (instead of silently doing nothing).

---

## 4) Performance: faster TIFF loading and faster full-stack loads

### 4.1 TIFF “fast path” with `tifffile`

#### Problem
Pillow can be slower for TIFF stacks and sometimes less ideal for 16-bit data.

#### Fix
Updated `src/utils/file_handler.py`:

- `get_image_at_index()` now tries `tifffile.imread` first for `.tif/.tiff`, falling back to Pillow.

This improves:
- time-to-preview for TIFFs
- consistency for 16-bit TIFF decoding

### 4.2 Parallel stack loading (threaded IO)

#### Problem
`get_all_frames_as_array()` previously loaded frames sequentially. For folders with many TIFFs, this is dominated by IO + decoding, and sequential loading is slow.

#### Fix
Updated `src/utils/file_handler.py`:

- `get_all_frames_as_array()` now uses a `ThreadPoolExecutor` (up to 8 workers) to load files concurrently.
- Falls back to sequential loading on any unexpected exception.

Why threads here?
- File IO + native decoding frequently releases the GIL.
- Threads avoid the complexity/fragility of spawning processes in a GUI frozen app.

---

## 5) Multiprocessing: safer alignment parallelism in a frozen GUI app

### Problem
Alignment already used `ProcessPoolExecutor` for speed, but multiprocessing was disabled when frozen because spawned workers can import Qt/PySide modules and destabilize the app.

### Fix: move worker functions out of Qt modules

Created a non-Qt module:

- `src/core/alignment_mp.py`

It contains the multiprocessing worker functions:
- `register_pair(...)`
- `transform_frame(...)`

Then updated:
- `src/ui/alignment_worker.py`

to import those functions instead of defining them in a Qt-importing module.

This reduces the chance that spawned workers import PySide6/Qt.

### Frozen behavior & opt-in
We keep a conservative default:

- **When frozen**: multiprocessing is **disabled by default**
- **Opt-in** with:

```bash
NEUROLIGHT_ENABLE_MP=1 open dist/Neurolight.app
```

This allows experimentation with MP on macOS while minimizing risk for typical users.

---

## 6) Crash fix: reduce peak memory during neuron detection

### Symptom
With `NEUROLIGHT_ENABLE_MP=1`, neuron detection could “promptly crash” without a Python traceback. This is typical of **OS-level termination** due to memory pressure.

### Root cause (high peak memory)
The detection pipeline was allocating large arrays:

- In `ImageProcessor.detect_neurons_in_roi()` it created a full-size `roi_region_stack` with shape:
  - `(num_frames, height, width)`
- In `NeuronDetectionWidget._on_detection_finished()` it created another:
  - `np.zeros_like(self.frame_data)` (same large shape)

When combined with extra memory overhead from multiprocessing, this can exceed available memory.

### Fix: operate on ROI bounding box, avoid 3D visualization stacks

#### In `src/core/image_processor.py`
`detect_neurons_in_roi()` now:

- Computes the ROI bounding box from `roi_mask`
- Crops the frame stack to the ROI bounding box before preprocessing
- Runs peak detection and trajectory extraction in **cropped coordinates**
- Converts neuron locations back to **full-image coordinates** before returning

This typically reduces memory and compute by a factor proportional to ROI area vs full-frame area.

#### In `src/ui/neuron_detection_widget.py`
Post-processing now:

- Avoids allocating a full `(frames, height, width)` array for display.
- Computes mean/first-frame visualization only in the ROI crop.
- Embeds the crop into 2D full-size `mean_frame` and `_display_frame` arrays for consistent plotting.

---

## How to build & test (current workflow)

### Build the app bundle

```bash
./build-macos-app.sh
open dist/Neurolight.app
```

### Build an unsigned installer pkg

```bash
./build-macos-pkg.sh
open dist/Neurolight-unsigned.pkg
```

### Enable multiprocessing (opt-in, use with care)

```bash
NEUROLIGHT_ENABLE_MP=1 open dist/Neurolight.app
```

---

## Known trade-offs and next possible improvements

- **Threaded stack loading** improves IO-bound loading time but can increase instantaneous disk activity.
- **Multiprocessing** can speed up alignment but increases memory usage and can interact poorly with libraries that internally parallelize (BLAS/FFT). We kept it opt-in for frozen builds.
- **Detection** is now much lower-memory, but still does CPU-heavy loops for trajectories and correlations.

Potential future work:
- **Prefetch adjacent frames** into the existing LRU cache for smoother slider scrubbing.
- **Vectorize trajectory extraction** (reduce inner Python loops).
- **Cap BLAS thread counts** in frozen builds when MP is enabled (prevents oversubscription).

