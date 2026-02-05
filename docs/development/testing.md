# Testing

## Running Tests

```bash
# Install test dependencies
uv sync --all-extras

# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_experiment_manager.py
```

## Writing Tests

- Create test files in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names

## GUI Testing

GUI tests run in headless mode in CI. For local testing, ensure you have a display available or set:

```bash
export QT_QPA_PLATFORM=offscreen
```
