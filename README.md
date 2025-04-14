# PDF to DOCX/TXT/MD/HTML OCR Converter

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A web-based application built with Flask to convert PDF documents into editable formats (DOCX, TXT, Markdown, HTML) using Optical Character Recognition (OCR). It supports multiple OCR engines and provides options for image preprocessing to improve accuracy.

![Screenshot](placeholder.png) <!-- Add a relevant screenshot here -->

## Features

*   **Web Interface:** Simple drag-and-drop or browse interface for uploading PDF files.
*   **Multiple Output Formats:** Convert PDFs to:
    *   Microsoft Word (`.docx`)
    *   Plain Text (`.txt`)
    *   Markdown (`.md`)
    *   HTML (`.html`)
*   **Multiple OCR Engines:** Choose between:
    *   **Tesseract:** (Default) Widely used, supports many languages.
    *   **EasyOCR:** Often good for complex layouts or noisy images.
    *   **PyOCR:** A wrapper that can use Tesseract or Cuneiform.
*   **Language Support:** Select the document language for better Tesseract accuracy (including multi-language support).
*   **Image Preprocessing:** Enhance image quality before OCR with options like:
    *   Grayscale conversion
    *   Sharpening
    *   Denoising
    *   Deskewing (straightening tilted text)
    *   Thresholding (creating high-contrast images)
    *   Border removal
    *   Contrast adjustment
    *   DPI selection (300 or 600 DPI)
    *   Preset profiles for common scenarios (scanned documents, low quality, etc.).
*   **Quality Settings:** Choose between standard (faster) and high quality (slower, potentially more accurate) OCR processing.
*   **Background Processing:** Handles conversions asynchronously, allowing users to monitor progress.
*   **Progress Tracking:** Real-time status updates during conversion.
*   **Dependency Checker:** Includes a script to help install necessary system and Python dependencies.
*   **In-App Installation Guide:** Provides installation instructions directly within the web interface via a modal.
*   **Automatic Cleanup:** Periodically removes old uploaded and converted files.

## Installation

### Prerequisites

1.  **Python:** Version 3.7 or higher.
2.  **pip:** Python package installer (usually comes with Python).
3.  **Tesseract OCR Engine:**
    *   **macOS:** `brew install tesseract tesseract-lang`
    *   **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install tesseract-ocr libtesseract-dev tesseract-ocr-all` (Install desired language packs, e.g., `tesseract-ocr-fra`)
    *   **Windows:** Download installer from [UB Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki). **Ensure Tesseract is added to your system's PATH.**
4.  **Poppler PDF Rendering Library:**
    *   **macOS:** `brew install poppler`
    *   **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install poppler-utils`
    *   **Windows:** Download latest binaries from [Poppler for Windows Releases](https://github.com/oschwartz10612/poppler-windows/releases/). Extract and add the `bin/` directory to your system's PATH.

### Using the Installer Script (Recommended)

The easiest way to install Python dependencies and check system prerequisites is using the provided script:

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/your-username/ocr-pdf-docx.git # Replace with your repo URL
cd ocr-pdf-docx

# Run the installer script
python install_dependencies.py

# To install optional OCR engines (EasyOCR, PyOCR):
# python install_dependencies.py --engine easyocr
# python install_dependencies.py --engine pyocr
# python install_dependencies.py --engine all
```

The script will:
*   Check your Python version.
*   Install core Python packages (`Flask`, `pytesseract`, `python-docx`, `pdf2image`, `Pillow`, etc.).
*   Check if Tesseract and Poppler are accessible via the system PATH.
*   Optionally install dependencies for EasyOCR and PyOCR.

### Manual Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ocr-pdf-docx.git # Replace with your repo URL
    cd ocr-pdf-docx
    ```

2.  **Install core Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **(Optional) Install dependencies for other OCR engines:**
    ```bash
    # For EasyOCR (may require manual PyTorch installation first - see https://pytorch.org/)
    pip install -r requirements-easyocr.txt

    # For PyOCR
    pip install -r requirements-pyocr.txt
    ```

4.  **Verify System Dependencies:** Ensure Tesseract and Poppler are installed and accessible in your system's PATH.

## Usage

1.  **Start the Flask application:**
    ```bash
    python app.py
    ```
    The application will be available at `http://127.0.0.1:8011` (or the configured host/port).

2.  **Open the web interface:** Navigate to the URL in your web browser.

3.  **Upload PDF:** Drag and drop a PDF file onto the designated area or click to browse.

4.  **Configure Options (Optional):**
    *   Click the "Show" button next to "Options" to expand the settings panel.
    *   **Preprocessing:** Enable and configure image enhancement options or select a preset profile. Adjust DPI if needed.
    *   **Processing:** Select the desired OCR Engine (Tesseract, EasyOCR, PyOCR), Document Language (for Tesseract), and OCR Quality.
    *   **Output:** Choose the desired Output Format (DOCX, TXT, MD, HTML).

5.  **Convert:** Click the "Upload and Convert" button.

6.  **Monitor Progress:** You will be redirected to a status page showing the conversion progress.

7.  **Download:** Once complete, click the "Download" button on the success page.

8.  **Convert Another:** Click "Convert Another File" to return to the upload page.

## Configuration

The application can be configured using environment variables (e.g., in a `.env` file):

*   `FLASK_ENV`: Set to `development` for debug mode, `production` otherwise.
*   `SECRET_KEY`: A strong, random secret key for session management. If not set, a temporary one is generated.
*   `PORT`: The port the application runs on (default: `8011`).

Example `.env` file:
