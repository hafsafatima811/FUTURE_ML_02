#!/usr/bin/env python3
"""
Setup script for Python 3.13
Run this first after extracting the project
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        return False
    print("SUCCESS!")
    return True

def main():
    print("\n" + "="*60)
    print("SUPPORT TICKET CLASSIFIER - PYTHON 3.13 SETUP")
    print("="*60)

    # Check Python version
    print(f"\nPython version: {sys.version}")

    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ required!")
        sys.exit(1)

    # Step 1: Create virtual environment
    if not os.path.exists('venv'):
        if not run_command(f'"{sys.executable}" -m venv venv', "Step 1: Creating virtual environment..."):
            sys.exit(1)
    else:
        print("\nVirtual environment already exists. Skipping creation.")

    # Step 2: Determine pip path
    if os.name == 'nt':  # Windows
        pip_path = 'venv\\Scripts\\pip.exe'
        python_path = 'venv\\Scripts\\python.exe'
    else:  # Mac/Linux
        pip_path = 'venv/bin/pip'
        python_path = 'venv/bin/python'

    # Step 3: Upgrade pip
    run_command(f'"{python_path}" -m pip install --upgrade pip', "Step 2: Upgrading pip...")

    # Step 4: Install requirements
    if not run_command(f'"{python_path}" -m pip install -r requirements.txt', "Step 3: Installing dependencies..."):
        sys.exit(1)

    # Step 5: Download NLTK data
    if not run_command(f'"{python_path}" download_nltk.py', "Step 4: Downloading NLTK data..."):
        sys.exit(1)

    # Step 6: Verify installation
    print(f"\n{'='*60}")
    print("Step 5: Verifying installation...")
    print(f"{'='*60}")

    test_script = '''
import flask
import sklearn
import pandas
import numpy
import nltk
print(f"Flask: {flask.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"NLTK: {nltk.__version__}")
print("\nAll dependencies installed successfully!")
'''

    with open('test_imports.py', 'w') as f:
        f.write(test_script)

    run_command(f'"{python_path}" test_imports.py', "")

    # Final message
    print(f"\n{'='*60}")
    print("SETUP COMPLETE!")
    print(f"{'='*60}")
    print("\nTo run the application:")
    print(f'  1. Activate virtual environment:')
    if os.name == 'nt':
        print(f'     venv\\Scripts\\activate')
    else:
        print(f'     source venv/bin/activate')
    print(f'  2. Run: python app.py')
    print(f'  3. Open browser: http://localhost:5000')
    print(f"\n{'='*60}")

if __name__ == '__main__':
    main()
