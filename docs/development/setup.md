# Development Setup

## Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Git

## Initial Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Neuro-Light/neurolight-prototype
   cd neurolight-prototype
   ```

2. **Install dependencies**

   ```bash
   uv sync --all-extras
   ```

3. **Install pre-commit hooks** (recommended)

   ```bash
   uv run pip install pre-commit
   uv run pre-commit install
   ```

   This will automatically run code quality checks before each commit.

## Running Checks Locally

### Linting

```bash
uv run ruff check .
uv run ruff format --check .
```

### Type Checking

```bash
uv run mypy src/
```

### Testing

```bash
uv run pytest tests/
```

### All Checks

```bash
uv run pre-commit run --all-files
```

## Workflow

1. Create a feature branch
2. Make your changes
3. Run pre-commit hooks (or they'll run automatically on commit)
4. Run tests: `uv run pytest tests/`
5. Ensure all CI checks pass
6. Submit a pull request
