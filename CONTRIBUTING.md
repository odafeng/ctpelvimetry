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

## Reporting Issues

Please use [GitHub Issues](https://github.com/odafeng/ctpelvimetry/issues) to report bugs or request features. Include:

- Python version and OS
- Steps to reproduce
- Expected vs. actual behaviour
- Error traceback (if applicable)

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
