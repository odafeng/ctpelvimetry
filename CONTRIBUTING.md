# Contributing to ctpelvimetry

Thank you for considering contributing to **ctpelvimetry**! Here's how you can help.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/odafeng/ctpelvimetry.git
   cd ctpelvimetry
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[seg]"
   pip install pytest
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
2. Make your changes
3. Run tests:
   ```bash
   pytest tests/ -v
   ```
4. Commit with a descriptive message:
   ```bash
   git commit -m "Add: brief description of change"
   ```
5. Push and open a Pull Request

## Code Style

- **Docstrings**: NumPy style for all public functions
- **Naming**: `snake_case` for functions and variables
- **Type hints**: Encouraged but not required
- **Linting**: `ruff check ctpelvimetry/ tests/` runs in CI on every PR. Run it locally before pushing:
  ```bash
  pip install ruff
  ruff check ctpelvimetry/ tests/
  ```

## Reproducibility (Snapshot) Tests

`tests/test_reproducibility.py` runs a synthetic phantom through `run_combined_pelvimetry` and compares the output against `tests/golden/phantom_snapshot.json`. This catches numerical drift when measurement code changes — even a 0.5 mm shift in `Sacral_Length_mm` will fail the test.

If your change legitimately moves the numbers (improved landmark detection, bug fix in metric calculation, etc.), regenerate the snapshot:

```bash
pytest tests/test_reproducibility.py --update-snapshot
```

Then **commit the updated JSON alongside your code change**. The diff in `phantom_snapshot.json` is auditable evidence of which metrics moved by how much. Reviewers should examine snapshot diffs carefully — unintended drift is exactly what this test exists to catch.

The phantom is geometrically crude and only exercises sacrum-side metrics. A future fixture based on a real, license-clean CT (e.g. Medical Decathlon case) would extend coverage to ISD and inlet/outlet metrics; PRs welcome.

## Reporting Issues

Please use [GitHub Issues](https://github.com/odafeng/ctpelvimetry/issues) to report bugs or request features. Include:

- Python version and OS
- Steps to reproduce
- Expected vs. actual behaviour
- Error traceback (if applicable)

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
