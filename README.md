# PDF to DOCX OCR Converter

A web application that converts PDF documents to editable DOCX files using Optical Character Recognition (OCR). This tool is useful for extracting text from scanned documents, making them searchable and editable.

![PDF to DOCX OCR Converter](https://via.placeholder.com/800x400?text=PDF+to+DOCX+OCR+Converter)

## Features

- Convert PDF documents to editable DOCX or TXT files
- Multiple OCR engine support (Tesseract, EasyOCR, PaddleOCR, Kraken, PyOCR)
- Multi-language support
- Adjustable quality settings
- Preprocessing options for improved accuracy
- Web-based interface
- Background processing for large documents
- Progress tracking

![screenshot](https://github.com/fabriziosalmi/ocr-pdf-docx/blob/main/screenshot.png?raw=true)

## Installation

### Prerequisites

The application requires the following dependencies:

- Python 3.7+
- Flask
- Poppler (for PDF conversion)
- Tesseract OCR (for text recognition)

### Platform-specific Installation

#### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Poppler
brew install poppler

# Install Tesseract OCR
brew install tesseract

# Install Python dependencies
pip install -r requirements.txt
```

#### Ubuntu/Debian

```bash
# Install Poppler
sudo apt-get update
sudo apt-get install poppler-utils

# Install Tesseract OCR
sudo apt-get install tesseract-ocr

# For additional languages (optional)
sudo apt-get install tesseract-ocr-all
# Or for specific languages (e.g., French)
sudo apt-get install tesseract-ocr-fra

# Install Python dependencies
pip install -r requirements.txt
```

#### Windows

1. **Install Poppler**:
   - Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
   - Extract and add the `bin` directory to your PATH

2. **Install Tesseract OCR**:
   - Download the installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - During installation, check "Add to PATH"

3. **Install Python dependencies**:
   ```
   pip install -r requirements.txt
   ```

#### Docker (Alternative)

```bash
# Build the Docker image
docker build -t ocr-pdf-docx .

# Run the container
docker run -p 8011:8011 -e DOCKER_ENV=true ocr-pdf-docx
```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:8011`

3. Upload a PDF file and select your preferred OCR engine and settings

4. Click "Upload and Convert" and wait for the process to complete

5. Download the resulting DOCX file

## OCR Engines

The application supports multiple OCR engines, each with different strengths:

- **Tesseract OCR**: Fast and supports many languages
- **EasyOCR**: Good for complex layouts and multiple languages
- **PaddleOCR**: High accuracy for Asian languages
- **Kraken OCR**: Specialized for historical documents
- **PyOCR**: Python wrapper for Tesseract and Cuneiform

## Additional Language Support

For Tesseract OCR, you can install additional language packs:

- **macOS**: `brew install tesseract-lang`
- **Linux**: `sudo apt-get install tesseract-ocr-[lang]` (replace [lang] with language code)
- **Windows**: During Tesseract installation, select additional languages

## Troubleshooting

### Common Issues

1. **"Poppler is not installed" error**:
   - Ensure Poppler is installed and added to your PATH
   - For detailed installation guide, click "View installation guides" on the home page

2. **"Tesseract OCR is not installed" error**:
   - Ensure Tesseract is installed and added to your PATH
   - Check if the correct version is installed using `tesseract --version`

3. **Slow processing for large files**:
   - Try using standard quality instead of high quality
   - Consider splitting large PDFs into smaller files

### Advanced Configuration

You can modify the following environment variables:

- `PORT`: Change the default port (default: 8011)
- `FLASK_ENV`: Set to "development" for debug mode
- `SECRET_KEY`: Set a custom secret key for session management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [pdf2image](https://github.com/Belval/pdf2image)
- [python-docx](https://github.com/python-openxml/python-docx)
- [Flask](https://flask.palletsprojects.com/)
- [TailwindCSS](https://tailwindcss.com/)
