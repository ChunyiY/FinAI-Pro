"""
Code Verification Script

Verifies that all Python files have correct syntax.
"""
import ast
import sys
from pathlib import Path

def verify_file(file_path: Path) -> bool:
    """Verify Python file syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        print(f"✓ {file_path.name} - Syntax OK")
        return True
    except SyntaxError as e:
        print(f"✗ {file_path.name} - Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"✗ {file_path.name} - Error: {e}")
        return False

def main():
    """Main verification function."""
    project_dir = Path(__file__).parent
    python_files = [
        'app.py',
        'data_fetcher.py',
        'real_data_fetcher.py',
        'smart_data_fetcher.py',
        'multi_source_fetcher.py',
        'demo_data.py',
        'stock_predictor.py',
        'sentiment_analyzer.py',
        'portfolio_optimizer.py',
        'utils.py',
        'setup.py'
    ]
    
    print("Verifying Python files...\n")
    all_ok = True
    
    for file_name in python_files:
        file_path = project_dir / file_name
        if file_path.exists():
            if not verify_file(file_path):
                all_ok = False
        else:
            print(f"⚠ {file_name} - File not found")
    
    print("\n" + "="*50)
    if all_ok:
        print("✓ All files verified successfully!")
        return 0
    else:
        print("✗ Some files have errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())

