"""
Dependency Checker Script

Checks if all required dependencies are installed and provides installation instructions.
"""
import sys
import importlib

REQUIRED_PACKAGES = {
    'streamlit': 'streamlit',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'yfinance': 'yfinance',
    'plotly': 'plotly',
    'torch': 'torch',
    'sklearn': 'scikit-learn',
    'nltk': 'nltk',
    'textblob': 'textblob',
    'vaderSentiment': 'vaderSentiment',
    'scipy': 'scipy',
}

def check_package(import_name: str, package_name: str) -> tuple[bool, str]:
    """
    Check if a package is installed.
    
    Args:
        import_name: Name to use in import statement
        package_name: Package name for pip install
    
    Returns:
        Tuple of (is_installed, version_or_error)
    """
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'installed')
        return True, version
    except ImportError:
        return False, f"Not installed (install with: pip install {package_name})"

def main():
    """Main dependency check function."""
    print("=" * 60)
    print("Dependency Checker")
    print("=" * 60)
    print()
    
    all_ok = True
    missing_packages = []
    
    for import_name, package_name in REQUIRED_PACKAGES.items():
        is_installed, info = check_package(import_name, package_name)
        status = "✓" if is_installed else "✗"
        print(f"{status} {package_name:20s} - {info}")
        
        if not is_installed:
            all_ok = False
            missing_packages.append(package_name)
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✓ All dependencies are installed!")
        
        # Check NLTK data
        print()
        print("Checking NLTK data...")
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
                print("✓ NLTK punkt data found")
            except LookupError:
                print("⚠ NLTK punkt data missing - run: python setup.py")
            
            try:
                nltk.data.find('vader_lexicon')
                print("✓ NLTK vader_lexicon found")
            except LookupError:
                print("⚠ NLTK vader_lexicon missing - run: python setup.py")
        except ImportError:
            print("⚠ NLTK not installed")
        
        return 0
    else:
        print("✗ Missing dependencies detected!")
        print()
        print("To install all dependencies, run:")
        print("  pip install -r requirements.txt")
        print()
        print("Or install missing packages individually:")
        for pkg in missing_packages:
            print(f"  pip install {pkg}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

