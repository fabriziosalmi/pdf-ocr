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

def print_header(message):
    """Print a formatted header message"""
    print("=" * 60)
    print(message)
    print("=" * 60)

def test_tesseract():
    """Test if Tesseract OCR is working properly"""
    print("Testing Tesseract OCR installation...")
    
    try:
        # First, check if pytesseract is installed
        try:
            import pytesseract
            print(f"✓ pytesseract module is installed")
        except ImportError:
            print("✗ pytesseract module is not installed. Install it with: pip install pytesseract")
            return False
            
        # Check Tesseract version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract version: {version}")
        except Exception as e:
            print(f"✗ Failed to get Tesseract version: {e}")
            print("\nThis usually means Tesseract is not installed or not in your PATH.")
            system = platform.system().lower()
            if system == "darwin":
                print("  - macOS: Install with 'brew install tesseract'")
            elif system == "linux":
                print("  - Linux: Install with 'sudo apt-get install tesseract-ocr'")
            elif system == "windows":
                print("  - Windows: Install from https://github.com/UB-Mannheim/tesseract/wiki")
                print("    Make sure to check 'Add to PATH' during installation")
            return False
        
        # Try to get list of available languages
        try:
            result = subprocess.run(['tesseract', '--list-langs'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Available languages:")
                for lang in result.stdout.strip().split('\n')[1:]:
                    print(f"  - {lang}")
            else:
                print(f"✗ Error getting languages: {result.stderr}")
        except Exception as e:
            print(f"✗ Error checking languages: {e}")
            
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
            print("✓ Created test image: test_ocr.png")
            
            # Try OCR on the image
            text = pytesseract.image_to_string(Image.open("test_ocr.png"))
            print(f"OCR result: '{text.strip()}'")
            
            # Attempt with explicit config for better results
            if "Tesseract OCR Test" not in text:
                print("Trying with explicit configuration...")
                text = pytesseract.image_to_string(
                    Image.open("test_ocr.png"),
                    config="--psm 6 --oem 3"
                )
                print(f"Second OCR result: '{text.strip()}'")
            
            if "Tesseract OCR Test" in text:
                print("✅ OCR TEST PASSED: Text was correctly recognized")
            else:
                print("❌ OCR TEST FAILED: Text was not correctly recognized")
                print("\nTroubleshooting tips:")
                print("1. Make sure you have a newer version of Tesseract (4.0+)")
                print("2. Install additional languages if needed")
                print("3. Try running 'tesseract test_ocr.png stdout' directly in terminal")
                
            # Keep the image for investigation
            print(f"Test image saved as 'test_ocr.png' for your reference")
            
            return "Tesseract OCR Test" in text
        except Exception as e:
            print(f"✗ Error during image creation test: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Tesseract OCR test failed: {e}")
        print("\nPossible solutions:")
        print("1. Make sure Tesseract is installed:")
        print("   - macOS: brew install tesseract")
        print("   - Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("   - Windows: Install from https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Verify it's in your PATH by running 'tesseract --version' in terminal")
        print("3. Check that pytesseract is correctly installed: pip install pytesseract")
        return False

def test_pdf_to_image():
    """Test PDF to image conversion with pdf2image"""
    print("\nTesting PDF to image conversion...")
    
    try:
        try:
            # Check if pdf2image is installed
            try:
                import pdf2image
                print(f"✓ pdf2image module is installed")
            except ImportError:
                print("✗ pdf2image module is not installed. Install it with: pip install pdf2image")
                print("  Then run the script again to complete the test.")
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
                    print("✓ Poppler is installed and working")
                    
                    # Try to convert a page
                    try:
                        images = pdf2image.convert_from_path(test_pdf_path, dpi=72)
                        if images and len(images) > 0:
                            print(f"✓ Successfully converted PDF to {len(images)} image(s)")
                            # Save the first image for reference
                            images[0].save("test_pdf_conversion.png")
                            print("✓ Saved converted image as 'test_pdf_conversion.png'")
                        else:
                            print("✗ PDF conversion failed - no images returned")
                            return False
                    except Exception as e:
                        print(f"✗ PDF conversion failed: {e}")
                        return False
                        
                except PDFInfoNotInstalledError:
                    print("✗ Poppler is not installed or not in PATH")
                    system = platform.system().lower()
                    if system == "darwin":
                        print("  - macOS: Install with 'brew install poppler'")
                    elif system == "linux":
                        print("  - Linux: Install with 'sudo apt-get install poppler-utils'")
                    elif system == "windows":
                        print("  - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases")
                        print("    and add the bin directory to your PATH")
                    return False
                except Exception as e:
                    print(f"✗ Error testing Poppler: {e}")
                    return False
                
                # Clean up
                try:
                    os.remove(test_pdf_path)
                    os.remove("test_pdf_conversion.png")
                except:
                    pass
                
                return True
                
            except ImportError:
                print("✗ Could not properly test Poppler installation")
                return False
                
        except Exception as e:
            print(f"✗ Error importing pdf2image: {e}")
            print("  Make sure you have installed all dependencies: pip install pdf2image")
            return False
            
    except Exception as e:
        print(f"✗ PDF to image test failed: {e}")
        return False

def test_dependencies():
    """Test both OCR dependencies and recommend fixes"""
    print_header("OCR PDF to DOCX Test Utility")
    
    tesseract_result = test_tesseract()
    pdf_result = test_pdf_to_image()
    
    print("\nTest Summary:")
    print(f"Tesseract OCR: {'✅ Passed' if tesseract_result else '❌ Failed'}")
    print(f"PDF to Image: {'✅ Passed' if pdf_result else '❌ Failed'}")
    
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
