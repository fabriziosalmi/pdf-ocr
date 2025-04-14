import os
import io
import sys
import subprocess
import threading
import time
from flask import Flask, request, render_template, send_file, flash, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from docx import Document
from PIL import Image
import uuid
import time

# Import the progress manager
from progress_manager import ProgressManager
from websocket_server import run_websocket_server

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB limit
app.secret_key = os.environ.get('SECRET_KEY', 'MSlCVqRL7LEbkeriYRLc4jNE7LSWUaWt')  # Change this in production!

# Docker-specific configuration
DOCKER_ENV = os.environ.get('DOCKER_ENV', 'false').lower() == 'true'

# Create progress manager instance
progress_manager = ProgressManager()

# Start WebSocket server in a separate thread
websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
websocket_thread.start()

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        # In Docker, we can assume dependencies are installed
        if DOCKER_ENV:
            return True, "Running in Docker, dependencies assumed to be installed"

        # Check for poppler on macOS using Homebrew
        if sys.platform == 'darwin':  # macOS
            try:
                # Try to import pdf2image
                from pdf2image import convert_from_path
                # Test with a simple pdf conversion call (with a non-existent file is fine)
                convert_from_path.get_page_count('test.pdf')
            except Exception as e:
                if "Unable to get page count. Is poppler installed and in PATH?" in str(e):
                    return False, "Poppler is not installed or not in PATH. Install it with 'brew install poppler'"

        # Check for Tesseract
        try:
            output = subprocess.check_output(['tesseract', '--version'], stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "Tesseract OCR is not installed or not in PATH. On macOS, install it with 'brew install tesseract'"

        return True, "All dependencies are properly installed"
    except Exception as e:
        return False, f"Error checking dependencies: {str(e)}"

@app.route('/')
def index():
    # Check if dependencies are properly installed
    deps_installed, message = check_dependencies()
    if not deps_installed:
        flash(message, 'error')
    return render_template('index.html')

def process_pdf_with_progress(pdf_path, conversion_id, ocr_engine="tesseract", language="eng", quality="standard"):
    """Process PDF with progress tracking"""
    try:
        # Import here to avoid issues if not installed
        from pdf2image import convert_from_path

        # Update progress to starting
        progress_manager.update_progress(conversion_id, 5, "processing", "Converting PDF to images...")
        progress_manager.broadcast_progress(conversion_id)

        # Get total page count for progress calculation
        total_pages = 0
        try:
            from pdf2image.pdf2image import pdfinfo_from_path
            pdf_info = pdfinfo_from_path(pdf_path)
            total_pages = pdf_info["Pages"]
        except:
            # Fall back to counting after conversion
            pass

        # Convert PDF to images
        dpi = 300
        if quality == "high":
            dpi = 600

        images = convert_from_path(pdf_path, dpi=dpi)

        if total_pages == 0:
            total_pages = len(images)

        # Update progress
        progress_manager.update_progress(conversion_id, 20, "processing", f"PDF converted to {total_pages} images. Starting OCR...")
        progress_manager.broadcast_progress(conversion_id)

        # Create a DOCX document
        document = Document()
        full_text = ""

        # Perform OCR on each image
        for i, image in enumerate(images):
            # Calculate current progress (20-90%)
            page_progress = int(20 + (70 * (i / total_pages)))
            progress_manager.update_progress(conversion_id, page_progress, "processing", f"Processing page {i+1} of {total_pages}...")
            progress_manager.broadcast_progress(conversion_id)

            # Save image temporarily to perform OCR
            temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{conversion_id}_page_{i}.png')
            image.save(temp_image_path, 'PNG')

            # Use appropriate OCR engine based on selection
            text = ""
            if ocr_engine == "tesseract":
                # Use pytesseract with specified language
                config = f"--oem 1 --psm 3 -l {language}"
                text = pytesseract.image_to_string(Image.open(temp_image_path), config=config)
            elif ocr_engine == "easyocr":
                # Use EasyOCR if selected
                import easyocr
                reader = easyocr.Reader([language])
                result = reader.readtext(temp_image_path, detail=0)
                text = '\n'.join(result)
            # ... Add other OCR engines here
            else:
                # Default to tesseract
                text = pytesseract.image_to_string(Image.open(temp_image_path))

            full_text += text + "\n\n" # Add page break representation

            # Add paragraph for each page with a page break
            document.add_paragraph(text)
            if i < len(images) - 1:  # Don't add page break after the last page
                document.add_page_break()

            os.remove(temp_image_path) # Clean up temp image

        # Update progress
        progress_manager.update_progress(conversion_id, 95, "finalizing", "Creating DOCX document...")
        progress_manager.broadcast_progress(conversion_id)

        # Save DOCX to file system
        orig_filename = session.get('orig_filename', 'document.pdf')
        output_filename = os.path.splitext(orig_filename)[0] + '.docx'
        docx_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{conversion_id}_{output_filename}")
        document.save(docx_path)

        # Store path in session for download
        session['docx_path'] = docx_path
        session['output_filename'] = output_filename

        # Final progress update
        progress_manager.update_progress(conversion_id, 100, "complete", "Conversion complete!")
        progress_manager.broadcast_progress(conversion_id)

        return True, docx_path, output_filename

    except Exception as e:
        error_message = str(e)
        progress_manager.update_progress(conversion_id, 0, "error", f"Error: {error_message}")
        progress_manager.broadcast_progress(conversion_id)
        return False, None, error_message

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check dependencies first
    deps_installed, message = check_dependencies()
    if not deps_installed:
        flash(message, 'error')
        flash("Please install the required dependencies before proceeding", 'error')
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        # Generate unique ID for this conversion
        conversion_id = str(uuid.uuid4())
        session['conversion_id'] = conversion_id

        # Save original filename for later use
        orig_filename = file.filename
        session['orig_filename'] = orig_filename

        # Get OCR options from form
        ocr_engine = request.form.get('ocr-engine', 'tesseract')
        language = request.form.get('language', 'eng')
        quality = request.form.get('ocr-quality', 'standard')

        # Create unique filename to avoid collisions
        filename = f"{conversion_id}_{secure_filename(file.filename)}"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        session['pdf_path'] = pdf_path

        # Initialize progress
        progress_manager.update_progress(conversion_id, 0, "starting", "Preparing to process...")

        # Process in background thread to avoid blocking
        def process_thread():
            try:
                success, docx_path, output = process_pdf_with_progress(
                    pdf_path, conversion_id, ocr_engine, language, quality)
                if not success and os.path.exists(pdf_path):
                    os.remove(pdf_path)
            except Exception as e:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                progress_manager.update_progress(conversion_id, 0, "error", f"Error: {str(e)}")
                progress_manager.broadcast_progress(conversion_id)

        # Start processing thread
        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()

        # Redirect to progress page
        return redirect(url_for('progress', conversion_id=conversion_id))

    else:
        flash('Invalid file type. Please upload a PDF.', 'error')
        return redirect(url_for('index'))

@app.route('/progress/<conversion_id>')
def progress(conversion_id):
    """Show progress page for a specific conversion"""
    # Make sure the conversion ID matches the one in session
    if 'conversion_id' not in session or session['conversion_id'] != conversion_id:
        flash('Invalid conversion session', 'error')
        return redirect(url_for('index'))

    # Get current progress
    progress_data = progress_manager.get_progress(conversion_id)

    # If conversion is complete, redirect to success page
    if progress_data['status'] == 'complete':
        return redirect(url_for('success'))

    # Get original filename for display
    orig_filename = session.get('orig_filename', 'document.pdf')

    return render_template('progress.html',
                          conversion_id=conversion_id,
                          filename=orig_filename,
                          progress=progress_data)

@app.route('/api/progress/<conversion_id>')
def get_progress(conversion_id):
    """API endpoint to get current progress"""
    progress_data = progress_manager.get_progress(conversion_id)
    return jsonify(progress_data)

@app.route('/success')
def success():
    # Check if we have valid session data
    if 'docx_path' not in session or 'output_filename' not in session:
        flash('No conversion data found. Please upload a file first.', 'error')
        return redirect(url_for('index'))

    return render_template('success.html', filename=session['output_filename'])

@app.route('/download')
def download_file():
    # Check if we have valid session data
    if 'docx_path' not in session or 'output_filename' not in session:
        flash('No conversion data found. Please upload a file first.', 'error')
        return redirect(url_for('index'))

    docx_path = session['docx_path']
    output_filename = session['output_filename']

    # Check if the file exists
    if not os.path.exists(docx_path):
        flash('The converted file is no longer available.', 'error')
        return redirect(url_for('index'))

    return send_file(
        docx_path,
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        as_attachment=True,
        download_name=output_filename
    )

@app.route('/new_conversion')
def new_conversion():
    # Clean up session data and files
    if 'pdf_path' in session and os.path.exists(session['pdf_path']):
        os.remove(session['pdf_path'])

    if 'docx_path' in session and os.path.exists(session['docx_path']):
        os.remove(session['docx_path'])

    # Clear session data
    session.clear()

    return redirect(url_for('index'))

if __name__ == '__main__':
    # Add initial dependency check
    deps_installed, message = check_dependencies()
    if not deps_installed:
        print(f"Warning: {message}")
        print("The application will start but may not work correctly until all dependencies are installed.")

    # Use 0.0.0.0 to make the server accessible externally in the container
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 8011))
    debug = os.environ.get('FLASK_ENV') == 'development'

    app.run(debug=debug, host=host, port=port)