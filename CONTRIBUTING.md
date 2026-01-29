# Contributing to AlphaGenome-Plus

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/alphagenome-plus.git
cd alphagenome-plus
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev,test,all]"
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_quantum_optimization.py

# Run with coverage
pytest tests/ --cov=alphagenome_plus --cov-report=html
```

### Code Formatting

We use Black and isort for code formatting:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

### Linting

```bash
# Run flake8
flake8 src/ tests/ --max-line-length=100

# Run mypy
mypy src/ --ignore-missing-imports
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, documented code
- Add tests for new functionality
- Update documentation as needed
- Follow existing code style

### 3. Test Your Changes

```bash
pytest tests/ -v
```

### 4. Commit and Push

```bash
git add .
git commit -m "feat: add new feature"
git push origin feature/your-feature-name
```

### 5. Create Pull Request

- Go to GitHub and create a PR
- Describe your changes clearly
- Reference any related issues

## Code Style Guidelines

### Python

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Maximum line length: 100 characters

### Documentation

- Use clear, concise language
- Include code examples
- Update README for new features
- Add inline comments for complex logic

### Testing

- Write unit tests for new functions
- Aim for >80% code coverage
- Test edge cases
- Use pytest fixtures for setup

## Pull Request Guidelines

### PR Title

Use conventional commits format:

- `feat: add new feature`
- `fix: resolve bug`
- `docs: update documentation`
- `test: add tests`
- `refactor: improve code structure`

### PR Description

Include:

1. **What**: Brief description of changes
2. **Why**: Motivation for changes
3. **How**: Technical details
4. **Testing**: How you tested
5. **Screenshots**: If applicable

### Checklist

- [ ] Tests pass locally
- [ ] Code is formatted (black, isort)
- [ ] Linting passes (flake8, mypy)
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)

## Adding New Features

### 1. Quantum Algorithms

For new quantum algorithms:

- Add module in `src/alphagenome_plus/quantum/`
- Include circuit visualization
- Provide classical baseline comparison
- Document quantum advantage claims

### 2. ML Models

For new ML models:

- Add to `src/alphagenome_plus/ml/`
- Include training script
- Provide pretrained weights (if applicable)
- Document hyperparameters

### 3. Analysis Tools

For new analysis tools:

- Add visualization examples
- Include sample data
- Provide Jupyter notebook tutorial

## Questions?

- Open an issue for bugs
- Start a discussion for feature requests
- Join our community forum

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
