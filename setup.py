"""
Setup Script - Download Required NLTK Data

Downloads necessary NLTK data for sentiment analysis.
"""
import sys

def check_nltk_installed():
    """Check if NLTK is installed."""
    try:
        import nltk
        return True, nltk
    except ImportError:
        return False, None

def download_nltk_data():
    """Download required NLTK data."""
    is_installed, nltk = check_nltk_installed()
    
    if not is_installed:
        print("=" * 60)
        print("ERROR: NLTK is not installed.")
        print("=" * 60)
        print("\nPlease install dependencies first:")
        print("  pip install -r requirements.txt")
        print("\nOr install NLTK directly:")
        print("  pip install nltk")
        print("\nThen run this script again.")
        sys.exit(1)
    
    # Handle SSL context for NLTK downloads
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    print("Downloading NLTK data...")
    print("-" * 60)
    
    try:
        nltk.download('punkt', quiet=True)
        print("✓ punkt downloaded successfully")
    except Exception as e:
        print(f"✗ punkt download failed: {e}")
    
    try:
        nltk.download('vader_lexicon', quiet=True)
        print("✓ vader_lexicon downloaded successfully")
    except Exception as e:
        print(f"✗ vader_lexicon download failed: {e}")
    
    try:
        nltk.download('stopwords', quiet=True)
        print("✓ stopwords downloaded successfully")
    except Exception as e:
        print(f"✗ stopwords download failed: {e}")
    
    print("-" * 60)
    print("NLTK data download completed!")

if __name__ == "__main__":
    download_nltk_data()
