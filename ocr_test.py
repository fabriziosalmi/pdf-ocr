#!/usr/bin/env python3
"""
Test script to verify OCR functionality.
Run with: python ocr_test.py
"""
import os
import sys
import subprocess
import platform
from PIL import Image, ImageDraw, ImageFont

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

# Logging levels
DEBUG = 0
INFO = 1
WARNING = 2
ERROR = 3
LOG_LEVEL = INFO  # Set default log level

def log(message, level=INFO):
    """Print a formatted log message with color and level."""
    if level < LOG_LEVEL:
        return
    
    if level == DEBUG:
        color = CYAN
        level_str = "DEBUG"
    elif level == INFO:
        color = GREEN
        level_str = "INFO"
    elif level == WARNING:
        color = YELLOW
        level_str = "WARNING"
    elif level == ERROR:
        color = RED
        level_str = "ERROR"
    else:
        color = RESET
        level_str = "UNKNOWN"
    
    print(f"{color}[{level_str}]{RESET} {message}")

def print_header(message):
    """Print a formatted header message"""
    print("=" * 60)
    print(message)
    print("=" * 60)

def test_tesseract():
    """Test if Tesseract OCR is working properly"""
    log("Testing Tesseract OCR installation...", INFO)
    
    try:
        # First, check if pytesseract is installed
        try:
            import pytesseract
            log(f"{GREEN}✓{RESET} pytesseract module is installed", INFO)
        except ImportError:
            log(f"{RED}✗{RESET} pytesseract module is not installed. Install it with: pip install pytesseract", ERROR)
            return False
            
        # Check Tesseract version
        try:
            version = pytesseract.get_tesseract_version()
            log(f"{GREEN}✓{RESET} Tesseract version: {version}", INFO)
        except Exception as e:
            log(f"{RED}✗{RESET} Failed to get Tesseract version: {e}", ERROR)
            log("\nThis usually means Tesseract is not installed or not in your PATH.", WARNING)
            system = platform.system().lower()
            if system == "darwin":
                log("  - macOS: Install with 'brew install tesseract'", WARNING)
            elif system == "linux":
                log("  - Linux: Install with 'sudo apt-get install tesseract-ocr'", WARNING)
            elif system == "windows":
                log("  - Windows: Install from https://github.com/UB-Mannheim/tesseract/wiki", WARNING)
                log("    Make sure to check 'Add to PATH' during installation", WARNING)
            return False
        
        # Try to get list of available languages
        try:
            result = subprocess.run(['tesseract', '--list-langs'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                log(f"{GREEN}✓{RESET} Available languages:", INFO)
                for lang in result.stdout.strip().split('\n')[1:]:
                    log(f"  - {lang}", INFO)
            else:
                log(f"{YELLOW}✗{RESET} Error getting languages: {result.stderr}", WARNING)
        except Exception as e:
            log(f"{YELLOW}✗{RESET} Error checking languages: {e}", WARNING)
            
        # Create a simple test image with text
        try:
            # Create a white image
            img = Image.new('RGB', (400, 100), color='white')
            d = ImageDraw.Draw(img)
            
            # Try to use a standard font
            font = None
            try:
                # Try to find a font that exists on most systems
                system_fonts = []
                if platform.system().lower() == "darwin":  # macOS
                    system_fonts = [
                        "/System/Library/Fonts/Helvetica.ttc",
                        "/System/Library/Fonts/Arial.ttf",
                        "/System/Library/Fonts/Times.ttc"
                    ]
                elif platform.system().lower() == "windows":
                    system_fonts = [
                        "C:\\Windows\\Fonts\\arial.ttf",
                        "C:\\Windows\\Fonts\\times.ttf"
                    ]
                else:  # Linux
                    system_fonts = [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
                    ]
                
                for font_path in system_fonts:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, 36)
                        break
            except Exception:
                pass
            
            # Add text (with or without font)
            if font:
                d.text((20, 30), "Tesseract OCR Test", fill="black", font=font)
            else:
                d.text((20, 30), "Tesseract OCR Test", fill="black")
            
            # Save the image
            img.save("test_ocr.png")
            log(f"{GREEN}✓{RESET} Created test image: test_ocr.png", INFO)
            
            # Try OCR on the image
            text = pytesseract.image_to_string(Image.open("test_ocr.png"))
            log(f"OCR result: '{text.strip()}'", DEBUG)
            
            # Attempt with explicit config for better results
            if "Tesseract OCR Test" not in text:
                log("Trying with explicit configuration...", INFO)
                text = pytesseract.image_to_string(
                    Image.open("test_ocr.png"),
                    config="--psm 6 --oem 3"
                )
                log(f"Second OCR result: '{text.strip()}'", DEBUG)
            
            if "Tesseract OCR Test" in text:
                log(f"{GREEN}✅{RESET} OCR TEST PASSED: Text was correctly recognized", INFO)
            else:
                log(f"{RED}❌{RESET} OCR TEST FAILED: Text was not correctly recognized", ERROR)
                log("\nTroubleshooting tips:", WARNING)
                log("1. Make sure you have a newer version of Tesseract (4.0+)", WARNING)
                log("2. Install additional languages if needed", WARNING)
                log("3. Try running 'tesseract test_ocr.png stdout' directly in terminal", WARNING)
                
            # Keep the image for investigation
            log(f"Test image saved as 'test_ocr.png' for your reference", INFO)
            
            return "Tesseract OCR Test" in text
        except Exception as e:
            log(f"{RED}✗{RESET} Error during image creation test: {e}", ERROR)
            return False
        
    except Exception as e:
        log(f"{RED}❌{RESET} Tesseract OCR test failed: {e}", ERROR)
        log("\nPossible solutions:", WARNING)
        log("1. Make sure Tesseract is installed:", WARNING)
        log("   - macOS: brew install tesseract", WARNING)
        log("   - Ubuntu/Debian: sudo apt-get install tesseract-ocr", WARNING)
        log("   - Windows: Install from https://github.com/UB-Mannheim/tesseract/wiki", WARNING)
        log("2. Verify it's in your PATH by running 'tesseract --version' in terminal", WARNING)
        log("3. Check that pytesseract is correctly installed: pip install pytesseract", WARNING)
        return False

def test_pdf_to_image():
    """Test PDF to image conversion with pdf2image"""
    log("\nTesting PDF to image conversion...", INFO)
    
    try:
        try:
            # Check if pdf2image is installed
            try:
                import pdf2image
                log(f"{GREEN}✓{RESET} pdf2image module is installed", INFO)
            except ImportError:
                log(f"{RED}✗{RESET} pdf2image module is not installed. Install it with: pip install pdf2image", ERROR)
                log("  Then run the script again to complete the test.", WARNING)
                return False
            
            # Check Poppler
            try:
                from pdf2image.exceptions import PDFInfoNotInstalledError
                
                # Create a tiny test PDF
                test_pdf_path = "test_pdf.pdf"
                img = Image.new('RGB', (100, 100), color='white')
                img.save(test_pdf_path)
                
                # Try to get info using pdf2image
                try:
                    pdf2image.pdfinfo_from_path(test_pdf_path)
                    log(f"{GREEN}✓{RESET} Poppler is installed and working", INFO)
                    
                    # Try to convert a page
                    try:
                        images = pdf2image.convert_from_path(test_pdf_path, dpi=72)
                        if images and len(images) > 0:
                            log(f"{GREEN}✓{RESET} Successfully converted PDF to {len(images)} image(s)", INFO)
                            # Save the first image for reference
                            images[0].save("test_pdf_conversion.png")
                            log(f"{GREEN}✓{RESET} Saved converted image as 'test_pdf_conversion.png'", INFO)
                        else:
                            log(f"{RED}✗{RESET} PDF conversion failed - no images returned", ERROR)
                            return False
                    except Exception as e:
                        log(f"{RED}✗{RESET} PDF conversion failed: {e}", ERROR)
                        return False
                        
                except PDFInfoNotInstalledError:
                    log(f"{RED}✗{RESET} Poppler is not installed or not in PATH", ERROR)
                    system = platform.system().lower()
                    if system == "darwin":
                        log("  - macOS: Install with 'brew install poppler'", WARNING)
                    elif system == "linux":
                        log("  - Linux: Install with 'sudo apt-get install poppler-utils'", WARNING)
                    elif system == "windows":
                        log("  - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases", WARNING)
                        log("    and add the bin directory to your PATH", WARNING)
                    return False
                except Exception as e:
                    log(f"{RED}✗{RESET} Error testing Poppler: {e}", ERROR)
                    return False
                
                # Clean up
                try:
                    os.remove(test_pdf_path)
                    os.remove("test_pdf_conversion.png")
                except OSError as e:
                    log(f"{YELLOW}✗{RESET} Could not clean up temporary files: {e}", WARNING)
                
                return True
                
            except ImportError:
                log(f"{RED}✗{RESET} Could not properly test Poppler installation", ERROR)
                return False
                
        except Exception as e:
            log(f"{RED}✗{RESET} Error importing pdf2image: {e}", ERROR)
            log("  Make sure you have installed all dependencies: pip install pdf2image", WARNING)
            return False
            
    except Exception as e:
        log(f"{RED}✗{RESET} PDF to image test failed: {e}", ERROR)
        return False

def test_dependencies():
    """Test both OCR dependencies and recommend fixes"""
    print_header("OCR PDF to DOCX Test Utility")
    
    tesseract_result = test_tesseract()
    pdf_result = test_pdf_to_image()
    
    print("\nTest Summary:")
    print(f"Tesseract OCR: {GREEN}✅ Passed{RESET} "
          f"if {GREEN if tesseract_result else RED}{tesseract_result}{RESET} else "
          f"{RED}❌ Failed{RESET}")
    print(f"PDF to Image: {GREEN}✅ Passed{RESET} "
          f"if {GREEN if pdf_result else RED}{pdf_result}{RESET} else "
          f"{RED}❌ Failed{RESET}")
    
    # Give combined advice
    if not (tesseract_result and pdf_result):
        print("\nRecommended fixes:")
        
        if not pdf_result:
            print("\n1. Fix PDF conversion:")
            print("   - Install pdf2image: pip install pdf2image")
            print("   - Install Poppler:")
            print("     • macOS: brew install poppler")
            print("     • Linux: sudo apt-get install poppler-utils")
            print("     • Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases")
            print("       and add the bin directory to your PATH")
        
        if not tesseract_result:
            print("\n2. Fix Tesseract OCR:")
            print("   - Install pytesseract: pip install pytesseract")
            print("   - Install Tesseract:")
            print("     • macOS: brew install tesseract")
            print("     • Linux: sudo apt-get install tesseract-ocr")
            print("     • Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        
        print("\nFor automatic installation of all dependencies:")
        print("python install_dependencies.py")
        
        return False
    
    print("\n✅ All tests passed! Your system is ready to run the OCR PDF to DOCX converter.")
    return True

if __name__ == "__main__":
    success = test_dependencies()
    sys.exit(0 if success else 1)
