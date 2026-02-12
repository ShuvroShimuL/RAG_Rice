# Contributing to RAG_Rice

Thank you for your interest in contributing to RAG_Rice! This document provides guidelines for contributing to this academic research project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Submitting Changes](#submitting-changes)
- [Academic Integrity](#academic-integrity)

## 🤝 Code of Conduct

This project is part of academic research at North South University. All contributors are expected to:

- Treat all participants with respect and professionalism
- Provide constructive feedback
- Focus on what is best for the community and research goals
- Show empathy towards other community members
- Maintain academic integrity and proper attribution

## 💡 How Can I Contribute?

### Reporting Bugs

If you find a bug, please create an issue with:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected behavior**
- **Actual behavior**
- **System information** (OS, Python version, etc.)
- **Screenshots** (if applicable)

### Suggesting Enhancements

We welcome suggestions for improvements! When suggesting enhancements:

- **Use a clear and descriptive title**
- **Provide detailed description** of the proposed feature
- **Explain why** this enhancement would be useful
- **Provide examples** of how it would work

### Pull Requests

We actively welcome pull requests for:

- Bug fixes
- Documentation improvements
- Performance optimizations
- New features (please discuss first)
- Test coverage improvements
- Code quality improvements

## 🛠️ Development Setup

### Prerequisites

```bash
Python 3.8+
Git
Virtual environment tool (venv)
```

### Setup Instructions

1. **Fork the repository**
   ```bash
   # Fork via GitHub UI
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/RAG_Rice.git
   cd RAG_Rice
   ```

3. **Set up upstream remote**
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/RAG_Rice.git
   ```

4. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

6. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

7. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

## 📝 Contribution Guidelines

### Code Style

- Follow **PEP 8** Python style guide
- Use **meaningful variable and function names**
- Add **docstrings** to all functions and classes
- Keep functions **small and focused** (single responsibility)
- Use **type hints** where appropriate

Example:
```python
def calculate_yield_prediction(
    region: str,
    variety: str,
    season: str,
    climate_data: dict
) -> float:
    """
    Predict rice yield based on input parameters.
    
    Args:
        region: Geographic region (e.g., 'Dhaka')
        variety: Rice variety (e.g., 'BRRI dhan28')
        season: Growing season (e.g., 'Boro')
        climate_data: Dictionary containing climate parameters
        
    Returns:
        Predicted yield in tons/hectare
        
    Raises:
        ValueError: If input parameters are invalid
    """
    # Implementation here
    pass
```

### Documentation

- Add docstrings to all public functions and classes
- Update README.md if adding new features
- Include inline comments for complex logic
- Update CHANGELOG.md for significant changes

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Include integration tests where appropriate

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_rag_system.py
```

### Commit Messages

Follow conventional commit format:

```
type(scope): subject

body

footer
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(rag): add multi-query expansion for better retrieval

Implemented query expansion using multiple reformulations
to improve document retrieval accuracy.

Closes #123

fix(ml): correct yield prediction for edge cases

Fixed bug where yield prediction failed for new rice varieties
not present in training data. Added fallback mechanism.

docs(readme): update installation instructions

Added troubleshooting section and clarified API key setup.
```

## 🚀 Submitting Changes

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   ```bash
   # Edit files
   git add .
   git commit -m "feat(scope): description"
   ```

3. **Keep your branch updated**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Go to GitHub and create PR from your fork
   - Fill in PR template with:
     - Description of changes
     - Related issues
     - Testing performed
     - Screenshots (if UI changes)

6. **Address review comments**
   - Respond to feedback
   - Make requested changes
   - Push updates to same branch

## 🎓 Academic Integrity

### Research Contributions

If your contribution involves research findings:

- **Clearly document** your methodology
- **Cite sources** appropriately
- **Provide data** or references for claims
- **Maintain reproducibility**

### Citation Requirements

When using this project in academic work, please cite:

```bibtex
@misc{shimul2025ragrice,
  author = {Shimul, M. Shamimul Haque Mondal},
  title = {Context-Aware Agricultural Chatbot: Integrating Document 
           Retrieval with Predictive Analytics for Rice Farming},
  year = {2025},
  school = {North South University},
  department = {Electrical and Computer Engineering},
  type = {Directed Research Project}
}
```

## 🤔 Questions?

If you have questions about contributing:

- **Open an issue** with the `question` label
- **Email the team**:
  - Student: shamimul.shimul@northsouth.edu
  - Supervisor: shahnewaz.siddique@northsouth.edu

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

## Development Roadmap

Current priorities for contributions:

### High Priority
- [ ] Bengali language support
- [ ] Additional rice variety data
- [ ] Improved yield prediction accuracy
- [ ] Web interface enhancements
- [ ] Mobile app development

### Medium Priority
- [ ] Real-time weather integration
- [ ] Farmer community features
- [ ] SMS gateway support
- [ ] Voice interface
- [ ] Offline mode

### Low Priority
- [ ] Satellite imagery analysis
- [ ] Drone integration
- [ ] Blockchain for supply chain
- [ ] IoT sensor integration

---

## Recognition

Contributors will be acknowledged in:
- Project README
- Research paper acknowledgments (if applicable)
- Future publications citing this work

Thank you for contributing to sustainable agriculture technology! 🌾

---

**Last Updated**: January 2025
