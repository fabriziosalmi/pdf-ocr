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

# Removed ProgressManager and websocket_server imports

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB limit
app.secret_key = os.environ.get('SECRET_KEY', 'MSlCVqRL7LEbkeriYRLc4jNE7LSWUaWt')  # Change this in production!

# Docker-specific configuration
DOCKER_ENV = os.environ.get('DOCKER_ENV', 'false').lower() == 'true'

# Removed ProgressManager instance and WebSocket thread start

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
    """Process PDF (progress parts removed)"""
    try:
        # Import PDF conversion library
        from pdf2image import convert_from_path, pdfinfo_from_path

        # Removed progress update

        # Get total page count (optional, less critical now)
        total_pages = 0
        try:
            pdf_info = pdfinfo_from_path(pdf_path)
            total_pages = pdf_info["Pages"]
        except Exception as e:
            app.logger.warning(f"Could not get page count via pdfinfo: {e}. Will count after conversion.")
            # Fall back to counting after conversion

        # Convert PDF to images
        dpi = 300
        if quality == "high":
            dpi = 600

        images = convert_from_path(pdf_path, dpi=dpi)

        if total_pages == 0:
            total_pages = len(images)

        # Removed progress update

        # Initialize OCR engine reader/tool (outside the loop for efficiency)
        ocr_reader = None
        ocr_tool = None # For PyOCR
        engine_initialized = False
        init_error_msg = None

        try:
            # ... (Engine initialization logic remains the same) ...
            if ocr_engine == "easyocr":
                import easyocr
                # Map common 3-letter codes to 2-letter if needed
                lang_map = {'eng': 'en', 'fra': 'fr', 'deu': 'de', 'spa': 'es', 'ita': 'it', 'por': 'pt', 'chi_sim': 'ch_sim', 'chi_tra': 'ch_tra', 'jpn': 'ja', 'kor': 'ko', 'rus': 'ru', 'ara': 'ar', 'hin': 'hi'}
                easyocr_lang = lang_map.get(language, language) # Use mapped or original
                # Handle multiple languages if '+' is present (basic split)
                langs_to_load = easyocr_lang.split('+')
                ocr_reader = easyocr.Reader(langs_to_load)
                engine_initialized = True
            elif ocr_engine == "paddleocr":
                from paddleocr import PaddleOCR
                # PaddleOCR uses different lang codes, map common ones
                lang_map = {'eng': 'en', 'fra': 'fr', 'deu': 'german', 'spa': 'es', 'ita': 'it', 'por': 'pt', 'chi_sim': 'ch', 'chi_tra': 'chinese_cht', 'jpn': 'japan', 'kor': 'korean', 'rus': 'ru', 'ara': 'ar', 'hin': 'hi'}
                paddle_lang = lang_map.get(language, 'en') # Default to english if map fails
                ocr_reader = PaddleOCR(use_angle_cls=True, lang=paddle_lang)
                engine_initialized = True
            elif ocr_engine == "kraken":
                from kraken import binarization, pageseg, rpred
                from kraken.lib import models
                # Kraken uses model files, assumes default model for the language if not specified
                # This is a simplified setup; real usage might need specific model selection
                model_path = models.load_any(language + '.mlmodel') # Assumes models are available
                if not model_path:
                     raise ImportError(f"Kraken model for language '{language}' not found.")
                ocr_reader = rpred.rpred(model_path, None) # None for device selection (auto)
                engine_initialized = True
            elif ocr_engine == "calamari":
                raise NotImplementedError("Calamari OCR requires specific model checkpoint configuration not yet implemented.")
                # ... calamari init placeholder ...
            elif ocr_engine == "pyocr":
                import pyocr
                import pyocr.builders
                tools = pyocr.get_available_tools()
                if len(tools) == 0:
                    raise ImportError("No PyOCR tools (Tesseract, Cuneiform) found. Is Tesseract installed?")
                ocr_tool = tools[0] # Use the first available tool (likely Tesseract)
                engine_initialized = True
            elif ocr_engine == "ocrmypdf":
                 raise NotImplementedError("OCRmyPDF integration requires a different processing pipeline (PDF-to-PDF).")
            elif ocr_engine == "tesseract":
                engine_initialized = True
            else:
                 raise ValueError(f"Unknown OCR engine: {ocr_engine}")

        except ImportError as ie:
            init_error_msg = f"Error initializing {ocr_engine}: Required library not installed ({ie})."
        except Exception as e:
            init_error_msg = f"Error initializing {ocr_engine}: {str(e)}"

        if not engine_initialized:
            # Removed progress update
            if os.path.exists(pdf_path): os.remove(pdf_path) # Clean up uploaded file
            return False, None, init_error_msg or f"Failed to initialize OCR engine '{ocr_engine}'."

        # Create a DOCX document
        document = Document()
        full_text = ""

        # Perform OCR on each image
        for i, image in enumerate(images):
            # Removed progress update

            # Save image temporarily to perform OCR (needed by most engines)
            temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{conversion_id}_page_{i}.png')
            # Ensure image is in RGB format for engines that require it
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(temp_image_path, 'PNG')

            text = ""
            try:
                # ... (OCR engine processing logic remains the same) ...
                if ocr_engine == "tesseract":
                    config = f"--oem 1 --psm 3 -l {language}" # Default config
                    app.logger.info(f"Processing page {i+1} with Tesseract. Language: '{language}', Config: '{config}'")
                    try:
                        # Explicitly open the image here for clarity
                        img_to_process = Image.open(temp_image_path)
                        text = pytesseract.image_to_string(img_to_process, config=config)
                        app.logger.info(f"Tesseract processing for page {i+1} completed.")
                    except pytesseract.TesseractNotFoundError:
                        app.logger.error("Tesseract executable not found. Ensure it's installed and in PATH.")
                        raise # Re-raise the specific error
                    except Exception as tess_error:
                        app.logger.error(f"Error during Tesseract processing for page {i+1}: {tess_error}")
                        text = f"[Tesseract processing error on page {i+1}]"
                elif ocr_engine == "easyocr" and ocr_reader:
                    # EasyOCR expects numpy array or file path
                    result = ocr_reader.readtext(temp_image_path, detail=0, paragraph=True)
                    text = '\n'.join(result)
                elif ocr_engine == "paddleocr" and ocr_reader:
                    result = ocr_reader.ocr(temp_image_path, cls=True)
                    # Extract text from PaddleOCR structure
                    page_text = []
                    for line in result[0]: # result is nested list [[line1], [line2], ...]
                         page_text.append(line[1][0]) # line[1][0] contains the text
                    text = '\n'.join(page_text)
                elif ocr_engine == "kraken" and ocr_reader:
                     # Kraken requires PIL image
                     im = Image.open(temp_image_path)
                     # Perform segmentation and prediction
                     segmentation = pageseg.segment(im) # Basic segmentation
                     if not segmentation or 'lines' not in segmentation:
                         text = "" # No lines found
                     else:
                         records = list(ocr_reader.predict_lines(im, segmentation['lines']))
                         text = '\n'.join([record.prediction for record in records])
                # elif ocr_engine == "calamari" and ocr_reader:
                #     text = "[Calamari processing not fully implemented]"
                elif ocr_engine == "pyocr" and ocr_tool:
                    # PyOCR needs language mapping for Tesseract tool
                    lang_map = {'eng': 'eng', 'fra': 'fra', 'deu': 'deu', 'spa': 'spa', 'ita': 'ita', 'por': 'por', 'chi_sim': 'chi_sim', 'chi_tra': 'chi_tra', 'jpn': 'jpn', 'kor': 'kor', 'rus': 'rus', 'ara': 'ara', 'hin': 'hin'}
                    pyocr_lang = lang_map.get(language, 'eng') # Default to eng
                    text = ocr_tool.image_to_string(
                        Image.open(temp_image_path),
                        lang=pyocr_lang,
                        builder=pyocr.builders.TextBuilder()
                    )

            except Exception as page_error:
                app.logger.error(f"Error processing page {i+1} with {ocr_engine}: {page_error}")
                text = f"[Error processing page {i+1} with {ocr_engine}]"
                # Removed progress update

            full_text += text + "\n\n" # Add page break representation

            # Add paragraph for each page with a page break
            document.add_paragraph(text)
            if i < len(images) - 1:  # Don't add page break after the last page
                document.add_page_break()

            os.remove(temp_image_path) # Clean up temp image

        # Removed progress update

        # Save DOCX to file system
        orig_filename = session.get('orig_filename', 'document.pdf') # Get original filename from session
        output_filename = os.path.splitext(orig_filename)[0] + '.docx'
        docx_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{conversion_id}_{output_filename}")
        document.save(docx_path)

        # Store path in session for download
        session['docx_path'] = docx_path
        session['output_filename'] = output_filename

        # Removed final progress update

        # Clean up original PDF after successful conversion
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        return True, docx_path, output_filename

    except ImportError as e:
        # Handle missing pdf2image or other core imports
        error_message = f"Core dependency missing: {e}. Please ensure pdf2image and its requirements (like Poppler) are installed."
        app.logger.error(error_message)
        # Removed progress update
        if 'pdf_path' in locals() and os.path.exists(pdf_path): os.remove(pdf_path)
        return False, None, error_message
    except NotImplementedError as e:
        error_message = str(e)
        app.logger.error(f"NotImplementedError during processing: {error_message}")
        # Removed progress update
        if 'pdf_path' in locals() and os.path.exists(pdf_path): os.remove(pdf_path)
        return False, None, error_message
    except Exception as e:
        import traceback
        error_message = f"An unexpected error occurred: {str(e)}"
        app.logger.error(f"Error during PDF processing: {traceback.format_exc()}")
        # Removed progress update
        # Clean up potentially created files
        if 'pdf_path' in locals() and os.path.exists(pdf_path): os.remove(pdf_path)
        # Clean up temp images if loop was interrupted
        for i in range(total_pages if 'total_pages' in locals() else 0):
             temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{conversion_id}_page_{i}.png')
             if os.path.exists(temp_image_path):
                 os.remove(temp_image_path)
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
        # Store conversion_id in session if needed for cleanup or associating files
        session['conversion_id'] = conversion_id

        # Save original filename for later use
        orig_filename = file.filename
        session['orig_filename'] = orig_filename # Keep for success page and output naming

        # Get OCR options from form
        ocr_engine = request.form.get('ocr-engine', 'tesseract')
        language = request.form.get('language', 'eng')
        quality = request.form.get('ocr-quality', 'standard')

        # Create unique filename to avoid collisions
        filename = f"{conversion_id}_{secure_filename(file.filename)}"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        # Keep pdf_path in session for potential cleanup on error or new conversion
        session['pdf_path'] = pdf_path

        # Removed progress initialization

        # Process synchronously
        try:
            success, result_path, output_or_error = process_pdf_with_progress(
                pdf_path, conversion_id, ocr_engine, language, quality)

            if success:
                # result_path is docx_path, output_or_error is output_filename
                # Session variables 'docx_path' and 'output_filename' are set within process_pdf_with_progress
                return redirect(url_for('success'))
            else:
                # output_or_error contains the error message
                flash(f"Conversion failed: {output_or_error}", 'error')
                # Clean up the uploaded PDF if it still exists
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                session.pop('pdf_path', None) # Remove from session too
                return redirect(url_for('index'))

        except Exception as e:
            # Catch unexpected errors during the synchronous call
            app.logger.error(f"Unexpected error in /upload route: {e}")
            flash(f"An unexpected error occurred during processing: {e}", 'error')
            # Clean up the uploaded PDF if it still exists
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            session.pop('pdf_path', None)
            return redirect(url_for('index'))

    else:
        flash('Invalid file type. Please upload a PDF.', 'error')
        return redirect(url_for('index'))

# Removed the /progress/<conversion_id> route
# Removed the /api/progress/<conversion_id> route

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
    # Use conversion_id from session if available for more specific cleanup,
    # but general cleanup based on session paths is okay too.
    pdf_path = session.pop('pdf_path', None)
    if pdf_path and os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except OSError as e:
            app.logger.warning(f"Could not remove PDF file {pdf_path}: {e}")

    docx_path = session.pop('docx_path', None)
    if docx_path and os.path.exists(docx_path):
         try:
            os.remove(docx_path)
         except OSError as e:
            app.logger.warning(f"Could not remove DOCX file {docx_path}: {e}")

    # Clear remaining session data relevant to a conversion
    session.pop('conversion_id', None)
    session.pop('orig_filename', None)
    session.pop('output_filename', None)

    # Redirect to index
    return redirect(url_for('index'))

# NEW: Add the guide route to render the installation guide page
@app.route('/guide')
def guide():
    return render_template('guide.html')

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