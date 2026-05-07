# Contributing to Neurolight Workbench

Thank you for your interest in contributing to Neurolight Workbench! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Linting and Formatting](#linting-and-formatting)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate in all interactions
- Accept constructive criticism gracefully
- Focus on what is best for the community and the project
- Show empathy towards other community members

## Getting Started

Before you begin contributing:

1. **Familiarize yourself with the project** – Read the [README.md](README.md) to understand Neurolight Workbench's purpose and architecture
2. **Check existing issues** – Browse [open issues](https://github.com/Neuro-Light/neurolight-workbench/issues) to see if your idea or bug has already been reported
3. **Join the discussion** – Comment on issues you're interested in working on to avoid duplicate efforts

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Fork the repository on GitHub

2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/neurolight-workbench.git
cd neurolight-workbench
```

3. Add the upstream repository:
```bash
git remote add upstream https://github.com/Neuro-Light/neurolight-workbench.git
```

4. Install all dependencies including test and dev extras:
```bash
uv sync --all-extras
```

5. Run the application to ensure everything works:
```bash
uv run python src/main.py
```

## How to Contribute

### Reporting Bugs

When reporting bugs, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Screenshots or error messages (if applicable)
- Your environment details (OS, Python version, etc.)
- Sample data or experiments that trigger the bug (if applicable)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- A clear description of the feature
- Use cases explaining why this feature would be valuable
- Potential implementation approach (if you have ideas)
- Any relevant mockups or examples from other applications

### Code Contributions

1. **Create a new branch** for your work:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

2. **Make your changes** following our [coding standards](#coding-standards)

3. **Test your changes** thoroughly

4. **Commit your changes** with clear, descriptive messages:
```bash
git commit -m "Add feature: brief description of what you added"
```

5. **Keep your fork updated**:
```bash
git fetch upstream
git rebase upstream/main
```

6. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

7. **Submit a Pull Request** (see [Pull Request Process](#pull-request-process))

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Maximum line length: **120 characters** (enforced by Ruff)
- Use type hints where appropriate

### Code Organization

- Keep functions focused and single-purpose
- Add docstrings to all public functions and classes
- Use descriptive comments for complex logic only; avoid narrating obvious code
- Follow the existing project structure:
  - `src/core/` – Pure domain logic and numerics (no Qt imports)
    - `experiment_manager.py` – Experiment session state
    - `image_processor.py` – Image processing and neuron detection
    - `alignment_mp.py` – Multiprocessing-safe alignment workers
    - `lomb_scargle.py` – Lomb-Scargle periodogram computation
    - `circular_stats.py` – Rayleigh and Rao circular statistics
    - `data_analyzer.py` – ROI intensity extraction helpers
    - `roi.py` – ROI geometry and mask utilities
    - `gif_generator.py` – Animated GIF export
  - `src/ui/` – PySide6 UI components and workers
  - `src/utils/` – Shared utilities
    - `file_handler.py` – TIF stack I/O
    - `image_utils.py` – NumPy-to-QImage conversion

### Example Function Documentation

```python
def load_image_stack(path: str, validate: bool = True) -> ImageStackHandler:
    """
    Load a TIF image stack from the specified path.

    Args:
        path: Absolute or relative path to the TIF file or directory
        validate: Whether to validate image dimensions and format

    Returns:
        ImageStackHandler containing the discovered image list

    Raises:
        FileNotFoundError: If the path does not exist
        ValueError: If images fail validation
    """
    ...
```

### UI Guidelines

- Maintain consistency with existing UI patterns
- Ensure all UI elements are accessible and clearly labeled
- Use Qt signals/slots for event handling
- Keep UI logic separate from business logic
- Long-running operations must run in a `QThread` worker, not the main thread

## Linting and Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

### Running Ruff locally

```bash
# Check for lint issues
uv run ruff check .

# Auto-fix lint issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# Check formatting without changing files
uv run ruff format --check .
```

### CI auto-fix behavior

On every pull request, the CI lint job automatically runs `ruff check --fix` and `ruff format`, then commits any fixes back to your branch with the commit message `style: auto-fix ruff lint and format`. This means:

- You do not need to have perfectly formatted code before pushing — CI will clean it up.
- If CI pushes a fix commit, pull it down before making further changes (`git pull`).
- The final `ruff check` and `ruff format --check` steps must still pass; auto-fix only handles automatically correctable issues.

The configured rules are `F` (Pyflakes), `E` (pycodestyle errors), and `I` (isort).

## Testing Guidelines

We use `pytest` for testing. All contributions should include tests where applicable.

### Running Tests

```bash
uv sync --all-extras  # Install test dependencies
uv run pytest tests/
```

Run with coverage:
```bash
uv run pytest tests/ -v --cov=src --cov-branch --cov-report=xml
```

Qt widget tests run headlessly via `QT_QPA_PLATFORM=offscreen` (set automatically in CI).

### Writing Tests

- Create test files in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- For Qt widget tests, use the `QApplication` fixture from `conftest.py`

Example:

```python
def test_experiment_creation_with_valid_data():
    """Test that experiments are created successfully with valid metadata."""
    # Arrange
    experiment_data = {
        "name": "Test Experiment",
        "description": "Test description"
    }

    # Act
    experiment = ExperimentManager.create_experiment(experiment_data)

    # Assert
    assert experiment.name == "Test Experiment"
    assert experiment.description == "Test description"
```

### Test Coverage

- Aim for meaningful test coverage of your changes
- Focus on testing critical paths and edge cases
- Test error handling and validation logic

## Pull Request Process

1. **Ensure your PR**:
   - Follows the coding standards
   - Includes tests for new functionality
   - Updates documentation if needed
   - Passes all existing tests

2. **PR Title**: Use a clear, descriptive title
   - `Feature: Add Lomb-Scargle period export`
   - `Fix: Resolve image loading crash with corrupted TIFs`
   - `Docs: Update installation instructions`

3. **PR Description**: Include:
   - Summary of changes
   - Related issue numbers (e.g., "Closes #123")
   - Testing performed
   - Screenshots (for UI changes)
   - Breaking changes (if any)

4. **Review Process**:
   - A maintainer will review your PR
   - Address any requested changes
   - Once approved, your PR will be merged

## Issue Guidelines

### Before Creating an Issue

- Search existing issues to avoid duplicates
- Check if your issue might be a question better suited for discussions

### Issue Templates

**Bug Report:**
```
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g. Windows 11, macOS 14]
 - Python version: [e.g. 3.10]
 - Neurolight version: [e.g. commit hash or tag]
```

**Feature Request:**
```
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request.
```

## Areas for Contribution

### Issue Labels and What They Mean

We label issues that are good for first-time contributors as [**`good first issue`**](https://github.com/Neuro-Light/neurolight-workbench/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). These usually do not require significant experience with the codebase.

We label issues we think are good opportunities for subsequent contributions as [**`help wanted`**](https://github.com/Neuro-Light/neurolight-workbench/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22). These require varying levels of experience.

**Please check in with us before starting work on an issue that has not been labeled as appropriate for community contribution.**

Outside of labeled issues, [**`bug`**](https://github.com/Neuro-Light/neurolight-workbench/issues?q=is%3Aissue+is%3Aopen+label%3Abug) issues are the best candidates for contribution. Issues labeled **`needs-decision`** or **`needs-design`** are not good candidates — please do not open pull requests for these without prior discussion.

**Please do not open pull requests for new features without prior discussion.** Adding a new feature to Neurolight Workbench creates a long-term maintenance burden and requires consensus from the team before implementation begins.

### Good First Issues

Look for issues tagged with `good first issue` — these are great for newcomers and typically include:

- Documentation improvements
- Adding code comments and docstrings
- Simple UI text or label improvements
- Example experiment templates
- Minor bug fixes with clear solutions

## Questions?

If you have questions about contributing:

- Open a discussion on [GitHub Discussions](https://github.com/Neuro-Light/neurolight-workbench/discussions)
- Comment on relevant issues
- Reach out to maintainers

## License

By contributing to Neurolight Workbench, you agree that your contributions will be licensed under the MIT License and Apache 2.0 License.

---

Thank you for contributing to Neurolight Workbench! Your efforts help advance neuroscience research tools for scientists worldwide.
