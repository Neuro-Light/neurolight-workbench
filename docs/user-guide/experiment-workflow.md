# Experiment Workflow

## What is an Experiment?

An **experiment** is a complete research session stored as a JSON file (`.nexp`) containing:

- 📋 Metadata (name, description, principal investigator, dates)
- 🖼️ Image stack information (path, dimensions, bit depth)
- ⚙️ Processing history (all operations and parameters)
- 📈 Analysis results (statistics, plots)
- 🎛️ Custom settings

## Creating a New Experiment

1. From the Startup Dialog, click **"Start New Experiment"**
2. Fill in the experiment metadata:
   - **Name** (required)
   - **Description** (optional)
   - **Principal Investigator** (optional)
3. Choose a save location (defaults to `experiments/` directory)
4. Click **Create** to begin

## Working with Image Stacks

- **Drag and drop** TIF files or folders onto the image viewer
- Navigate frames using **Previous/Next** buttons or the slider
- The frame counter shows your current position

## Sharing Experiments

To collaborate with colleagues:

1. Export the `.nexp` file from your `experiments/` directory
2. Include the referenced image stack folder
3. Colleagues can load the experiment and reproduce your entire workflow

**Note:** Both the `.nexp` file and the image stack folder must be shared.
