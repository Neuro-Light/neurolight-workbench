# Getting Started

## Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

1. **Install uv** (if not already installed)

   ```bash
   # Windows (PowerShell)
   pip install uv

   # macOS/Linux
   brew install uv
   ```

2. **Clone the repository**

   ```bash
   git clone https://github.com/Neuro-Light/neurolight-prototype
   cd neurolight-prototype
   ```

3. **Install dependencies**

   ```bash
   uv sync
   ```

4. **Launch the application**

   ```bash
   uv run python src/main.py
   ```

## First Steps

1. When the application starts, you'll see the **Startup Dialog**
2. Choose to create a new experiment or load an existing one
3. Load your TIF image stack using drag-and-drop
4. Start analyzing your data!

For more detailed information, see the [Experiment Workflow](experiment-workflow.md) guide.
