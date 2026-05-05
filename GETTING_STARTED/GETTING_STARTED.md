# Getting Started with NeuroLight

This guide walks you through installing NeuroLight and running your first analysis using the included sample images.

---

## Prerequisites

- **Operating System:** macOS (primary), Windows
  - macOS builds are code-signed and notarized — you will not receive an "unidentified developer" warning
  - Windows builds are available as standard installers
- **No Python installation required** — the distributed app bundles its own runtime

---

## Installation

1. Go to the [NeuroLight Releases page](https://github.com/Neuro-Light/neurolight-workbench/releases)
2. Download the installer for your operating system:
   - **macOS:** `NeuroLight-x.x.x.dmg`
   - **Windows:** `NeuroLight-x.x.x-setup.exe`
3. Run the installer and follow the on-screen prompts
4. Launch **NeuroLight** from your Applications folder (macOS) or Start Menu (Windows)

---

## Test Images

This `GETTING_STARTED/` folder contains a `TEST_IMAGES.zip` folder with 120 sample fluorescence images you can use to explore the application right away.

**To use them:**

1. Download the entire `GETTING_STARTED/TEST_IMAGES.zip/` folder to your computer
2. Unzip the folder somewhere easy to access.
3. When NeuroLight prompts you to load images, navigate to that folder and select it
4. The app will load all 120 images as a stack — no other setup needed

---

## First Launch Walkthrough

### 1. Log In or Create a Profile

When NeuroLight opens, you will be prompted to log in or create a new user profile. Profiles keep your experiment sessions separate. Enter a username to get started.

### 2. Create a New Experiment

From the startup screen, click **New Experiment**. Give it a name (e.g., `test-run`) and confirm. NeuroLight will create a session file (`.nexp`) to store your work.

### 3. Load the Test Images

- In the main window, open **File > Open Image Stack** (or drag the `test_images/` folder into the image panel)
- Select the `test_images/` folder you downloaded
- The image viewer will populate with the first frame of the stack; use the slider or Previous/Next buttons to navigate

### 4. Align Images (Optional but Recommended)

Go to **Tools > Align Stack**. PyStackReg will correct for any drift between frames. This step improves the accuracy of neuron detection and intensity extraction.

### 5. Select SCN Regions of Interest

Click **Select ROIs** in the workflow panel. You will draw two rectangular regions over the suprachiasmatic nucleus (SCN) areas visible in the image. These define the regions analyzed for circadian activity.

### 6. Run Neuron Detection

Click **Detect Neurons**. NeuroLight uses adaptive thresholding to automatically identify active neurons within your selected ROIs — no manual marking required.

### 7. Run Circadian Analysis

Once detection is complete, click **Analyze**. NeuroLight will:

- Extract fluorescence intensity per neuron per frame
- Build a time axis from EXIF timestamps embedded in the images
- Compute the Lomb-Scargle periodogram to estimate the circadian period
- Run Rayleigh and Rao circular statistics to assess rhythm significance

Results appear in the analysis tabs on the right panel. You can export neuron peak/trough data as CSV via **File > Export**.

---

## What's Next

- See the main [README](../README.md) for an architecture overview and full tech stack details
- Open an issue on GitHub if you encounter a problem or have a feature request
