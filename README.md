# OCR PDF to DOCX Converter

A web application to convert PDF files to editable DOCX documents using various OCR engines.

## Docker Setup (Recommended)

The easiest way to run this application is using Docker, which includes all dependencies.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Running with Docker

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ocr-pdf-docx.git
   cd ocr-pdf-docx
   ```

2. Start the application using Docker Compose:
   ```
   docker-compose up -d
   ```
   
   Or use the helper script:
   ```
   chmod +x run_docker.sh
   ./run_docker.sh
   ```

3. Access the application at http://localhost:8011

4. To stop the application:
   ```
   docker-compose down
   ```

## Manual Setup

If you prefer to run without Docker, you'll need to install the required dependencies manually.

### Dependencies

- Python 3.8+
- Poppler Utils
- Tesseract OCR
- Various OCR engines (optional)

See the installation guide in the application for detailed instructions.

### Installation

1. Install the required system dependencies (see above)

2. Install Python requirements:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Access the application at http://localhost:8011
