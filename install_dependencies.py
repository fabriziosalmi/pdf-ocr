#!/usr/bin/env python3
"""
Dependency installer for OCR PDF to DOCX converter
This script helps install the necessary dependencies for the OCR application.
"""

import os
import sys
import subprocess
import platform
import argparse

def print_step(message):
    """Print a formatted step message"""
    print(f"\n\033[1;34m==>\033[0m \033[1m{message}\033[0m")

def print_success(message):
    """Print a success message"""
    print(f"\033[1;32m✓\033[0m \033[1m{message}\033[0m")

def print_error(message):
    """Print an error message"""
    print(f"\033[1;31m✗\033[0m \033[1;31m{message}\033[0m")

def print_info(message):
    """Print an info message"""
    print(f"\033[1;33mℹ\033[0m {message}")

def run_command(command, shell=False, check=True):
    """Run a command and return its output"""
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            check=check, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Command failed with error code {e.returncode}: {e.stderr}"
    except Exception as e:
        return False, str(e)

def check_python_version():
    """Check if Python version is compatible"""
    print_step("Checking Python version")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print_error(f"Python version 3.7+ is required. Detected {sys.version.split()[0]}")
        return False
    
    print_success(f"Python version is compatible: {sys.version.split()[0]}")
    return True

def check_pip():
    """Check if pip is installed"""
    print_step("Checking pip installation")
    
    success, output = run_command([sys.executable, "-m", "pip", "--version"], check=False)
    if not success:
        print_error("pip is not installed or not working properly")
        return False
    
    print_success(f"pip is installed: {output.strip()}")
    return True

def install_core_requirements():
    """Install core Python dependencies"""
    print_step("Installing core Python dependencies")
    
    # Install pdf2image specifically (since it was missing in the test)
    print_info("Installing pdf2image...")
    success, output = run_command([sys.executable, "-m", "pip", "install", "pdf2image>=1.16.0"])
    if not success:
        print_error(f"Failed to install pdf2image: {output}")
        return False
    print_success("pdf2image installed successfully")
    
    # Install pytesseract (core OCR library)
    print_info("Installing pytesseract...")
    success, output = run_command([sys.executable, "-m", "pip", "install", "pytesseract>=0.3.8"])
    if not success:
        print_error(f"Failed to install pytesseract: {output}")
        return False
    print_success("pytesseract installed successfully")
    
    # Install other core dependencies
    print_info("Installing other core dependencies...")
    requirements = [
        "Flask>=2.0.0",
        "werkzeug>=2.0.0",
        "python-docx>=0.8.11", 
        "Pillow>=8.0.0",
        "gunicorn>=20.1.0",
        "python-dotenv>=0.19.0"
    ]
    
    for req in requirements:
        print_info(f"Installing {req}...")
        success, output = run_command([sys.executable, "-m", "pip", "install", req])
        if not success:
            print_error(f"Failed to install {req}: {output}")
            return False
    
    print_success("Core Python dependencies installed successfully")
    return True

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print_step("Checking Tesseract OCR installation")
    
    success, output = run_command(["tesseract", "--version"], check=False)
    if not success:
        print_error("Tesseract OCR is not installed or not in PATH")
        system = platform.system().lower()
        if system == "darwin":
            print_info("Install Tesseract on macOS with: brew install tesseract")
        elif system == "linux":
            print_info("Install Tesseract on Linux with: sudo apt-get install tesseract-ocr")
        elif system == "windows":
            print_info("Install Tesseract on Windows from: https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    print_success(f"Tesseract OCR is installed: {output.splitlines()[0] if output else 'Unknown version'}")
    return True

def check_poppler():
    """Check if Poppler is installed"""
    print_step("Checking Poppler installation")
    
    # Different commands for different platforms
    if platform.system().lower() == "windows":
        success, output = run_command(["where", "pdftoppm.exe"], check=False)
    else:
        success, output = run_command(["which", "pdftoppm"], check=False)
    
    if not success:
        print_error("Poppler is not installed or not in PATH")
        system = platform.system().lower()
        if system == "darwin":
            print_info("Install Poppler on macOS with: brew install poppler")
        elif system == "linux":
            print_info("Install Poppler on Linux with: sudo apt-get install poppler-utils")
        elif system == "windows":
            print_info("Install Poppler on Windows from: https://github.com/oschwartz10612/poppler-windows/releases")
        return False
    
    print_success("Poppler is installed")
    return True

def install_specific_ocr_engine(engine):
    """Install a specific OCR engine"""
    print_step(f"Installing {engine} OCR engine")
    
    req_file = f"requirements-{engine}.txt"
    if os.path.exists(req_file):
        print_info(f"Installing from {req_file}...")
        success, output = run_command([sys.executable, "-m", "pip", "install", "-r", req_file])
        if not success:
            print_error(f"Failed to install {engine} requirements: {output}")
            return False
    else:
        # Install based on engine name if requirements file doesn't exist
        packages = []
        if engine == "easyocr":
            packages = ["easyocr>=1.5.0", "torch>=1.7.0", "torchvision>=0.8.1"]
        elif engine == "paddleocr":
            packages = ["paddleocr>=2.0.0", "paddlepaddle>=2.0.0"]
        elif engine == "kraken":
            packages = ["kraken>=4.0.0"]
        elif engine == "pyocr":
            packages = ["pyocr>=0.8.0"]
        
        if packages:
            for pkg in packages:
                print_info(f"Installing {pkg}...")
                success, output = run_command([sys.executable, "-m", "pip", "install", pkg])
                if not success:
                    print_error(f"Failed to install {pkg}: {output}")
                    return False
        else:
            print_error(f"No installation method defined for {engine}")
            return False
    
    print_success(f"{engine} OCR engine dependencies installed successfully")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Install dependencies for OCR PDF to DOCX converter")
    parser.add_argument('--engine', choices=['tesseract', 'easyocr', 'paddleocr', 'kraken', 'pyocr', 'all'], 
                        default='tesseract', help="Specify OCR engine to install dependencies for")
    args = parser.parse_args()
    
    print("\n\033[1;36mOCR PDF to DOCX Converter - Dependency Installer\033[0m")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check pip
    if not check_pip():
        return 1
    
    # Install core requirements (including pdf2image)
    if not install_core_requirements():
        return 1
    
    # Check system dependencies
    check_tesseract()
    check_poppler()
    
    # Install specific OCR engine if requested
    if args.engine == 'all':
        for engine in ['easyocr', 'paddleocr', 'kraken', 'pyocr']:
            install_specific_ocr_engine(engine)
    elif args.engine != 'tesseract':  # tesseract is part of core requirements
        install_specific_ocr_engine(args.engine)
    
    print("\n\033[1;32m✓ Installation completed\033[0m")
    print("You can now run the application with: python app.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
