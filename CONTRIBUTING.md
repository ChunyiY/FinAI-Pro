# Contributing to FinAI Pro

Thank you for your interest in contributing to FinAI Pro! This document provides guidelines and instructions for contributing.

## 🎯 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- Environment details (Python version, OS, etc.)

### Suggesting Features

Feature suggestions are welcome! Please include:
- Clear description of the feature
- Use case and benefits
- Potential implementation approach (if you have ideas)

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add comments and docstrings
   - Update documentation if needed
   - Ensure all tests pass

4. **Commit your changes**
   ```bash
   git commit -m "Add: Description of your changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Include screenshots for UI changes

## 📋 Code Style

### Python Style Guide
- Follow PEP 8 style guide
- Use type hints for function parameters and return types
- Write docstrings for all functions and classes
- Keep functions focused and single-purpose

### Example:
```python
def get_stock_data(
    self, 
    symbol: str, 
    period: str = "1y"
) -> pd.DataFrame:
    """
    Fetch stock data from Alpha Vantage.
    
    Args:
        symbol: Stock ticker symbol
        period: Time period for data
    
    Returns:
        DataFrame with stock data
    """
    # Implementation
```

## 🧪 Testing

Before submitting:
- Run `python verify_code.py` to check syntax
- Test your changes manually
- Ensure no linter errors

## 📝 Documentation

- Update README.md if adding new features
- Add docstrings to new functions
- Update relevant documentation files

## ✅ Checklist

Before submitting a PR:
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No linter errors
- [ ] Commit messages are clear

Thank you for contributing! 🎉

