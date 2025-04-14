import os
import io
import sys
import subprocess
import threading
import time
import multiprocessing
from flask import Flask, request, render_template, send_file, flash, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from docx import Document
from PIL import Image
import uuid
import time
import json
from threading import Thread
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB limit
app.secret_key = os.environ.get('SECRET_KEY', 'MSlCVqRL7LEbkeriYRLc4jNE7LSWUaWt')  # Change this in production!

# Docker-specific configuration
DOCKER_ENV = os.environ.get('DOCKER_ENV', 'false').lower() == 'true'

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Add storage for background tasks
TASK_STATUS = {}  # Store task status updates
TASK_RESULTS = {}  # Store task results

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

def check_dependency(name):
    """Check a specific dependency and return detailed information"""
    if DOCKER_ENV:
        return True, {"installed": True, "version": "Docker Environment", "message": "Running in Docker container"}
    
    try:
        if name.lower() == 'poppler':
            try:
                output = subprocess.check_output(['pdftoppm', '-v'], stderr=subprocess.STDOUT, text=True)
                version = output.strip() if output else "Unknown version"
                return True, {"installed": True, "version": version, "message": "Poppler is installed"}
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False, {"installed": False, "message": "Poppler is not installed or not in PATH"}
        
        elif name.lower() == 'tesseract':
            try:
                version_output = subprocess.check_output(['tesseract', '--version'], stderr=subprocess.STDOUT, text=True)
                version = version_output.split('\n')[0] if version_output else "Unknown version"
                
                # Try to get available languages
                langs_output = subprocess.check_output(['tesseract', '--list-langs'], stderr=subprocess.STDOUT, text=True)
                langs = [l.strip() for l in langs_output.split('\n')[1:] if l.strip()]
                
                return True, {
                    "installed": True, 
                    "version": version, 
                    "languages": langs,
                    "message": "Tesseract is installed"
                }
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False, {"installed": False, "message": "Tesseract is not installed or not in PATH"}
        
        else:
            return False, {"installed": False, "message": f"Unknown dependency: {name}"}
    
    except Exception as e:
        return False, {"installed": False, "message": f"Error checking {name}: {str(e)}"}

@app.route('/')
def index():
    # Check if dependencies are properly installed
    deps_installed, message = check_dependencies()
    if not deps_installed:
        flash(message, 'error')
    return render_template('index.html')

def sanitize_text(text):
    import re
    # Remove control characters (except newline \n and tab \t)
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)

def process_image(i, image_path, ocr_engine, language):
    """Process a single image with OCR (to be used in parallel processing)"""
    try:
        text = ""
        if ocr_engine == "tesseract":
            config = f"--oem 1 --psm 3 -l {language}"
            img_to_process = Image.open(image_path)
            text = pytesseract.image_to_string(img_to_process, config=config)
        elif ocr_engine == "easyocr":
            import easyocr
            lang_map = {'eng': 'en', 'fra': 'fr', 'deu': 'de', 'spa': 'es', 'ita': 'it', 'por': 'pt', 
                       'chi_sim': 'ch_sim', 'chi_tra': 'ch_tra', 'jpn': 'ja', 'kor': 'ko', 'rus': 'ru', 
                       'ara': 'ar', 'hin': 'hi'}
            langs_to_load = [lang_map.get(lang, lang) for lang in language.split('+')]
            reader = easyocr.Reader(langs_to_load)
            result = reader.readtext(image_path, detail=0, paragraph=True)
            text = '\n'.join(result)
        # Add cases for other OCR engines similarly
        # ...

        # Sanitize text
        import re
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
        return i, text
    except Exception as e:
        return i, f"[Error processing page {i+1}: {str(e)}]"
    finally:
        # We don't delete the file here as it will be managed by the main process
        pass

def process_pdf_with_progress(pdf_path, conversion_id, ocr_engine="tesseract", language="eng", quality="standard", orig_filename=None):
    """Process PDF with parallel processing for speed"""
    try:
        # Import PDF conversion library
        from pdf2image import convert_from_path, pdfinfo_from_path

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
            if os.path.exists(pdf_path): os.remove(pdf_path) # Clean up uploaded file
            return False, None, init_error_msg or f"Failed to initialize OCR engine '{ocr_engine}'."

        # Create a DOCX document
        document = Document()
        
        # Save images to temporary files
        temp_image_paths = []
        for i, image in enumerate(images):
            temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{conversion_id}_page_{i}.png')
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(temp_image_path, 'PNG')
            temp_image_paths.append((i, temp_image_path))
        
        # Determine number of worker processes (use 75% of available cores)
        num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
        
        # Process images in parallel using a process pool
        results = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create tasks for each image
            futures = {
                executor.submit(
                    process_image, i, temp_path, ocr_engine, language
                ): i 
                for i, temp_path in temp_image_paths
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(futures)):
                page_idx, text = future.result()
                results[page_idx] = text
                
                # Update progress
                progress = int((i + 1) / len(images) * 100)
                if conversion_id in TASK_STATUS:
                    TASK_STATUS[conversion_id]["progress"] = progress
        
        # Sort results by page index and add to document
        for i in range(len(images)):
            if i in results:
                text = results[i]
                document.add_paragraph(text)
                if i < len(images) - 1:  # Don't add page break after the last page
                    document.add_page_break()
        
        # Clean up temporary files
        for _, temp_path in temp_image_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Save DOCX to file system
        # Instead of accessing session, use passed orig_filename parameter
        document_name = orig_filename or 'document.pdf'
        output_filename = os.path.splitext(document_name)[0] + '.docx'
        docx_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{conversion_id}_{output_filename}")
        document.save(docx_path)

        # Clean up original PDF after successful conversion
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        return True, docx_path, output_filename

    except ImportError as e:
        # Handle missing pdf2image or other core imports
        error_message = f"Core dependency missing: {e}. Please ensure pdf2image and its requirements (like Poppler) are installed."
        app.logger.error(error_message)
        if 'pdf_path' in locals() and os.path.exists(pdf_path): os.remove(pdf_path)
        return False, None, error_message
    except NotImplementedError as e:
        error_message = str(e)
        app.logger.error(f"NotImplementedError during processing: {error_message}")
        if 'pdf_path' in locals() and os.path.exists(pdf_path): os.remove(pdf_path)
        return False, None, error_message
    except Exception as e:
        import traceback
        error_message = f"An unexpected error occurred: {str(e)}"
        app.logger.error(f"Error during PDF processing: {traceback.format_exc()}")
        if 'pdf_path' in locals() and os.path.exists(pdf_path): os.remove(pdf_path)
        for i in range(total_pages if 'total_pages' in locals() else 0):
             temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{conversion_id}_page_{i}.png')
             if os.path.exists(temp_image_path):
                 os.remove(temp_image_path)
        return False, None, error_message

def run_task_in_background(func, task_id, *args, **kwargs):
    """Run a function in a background thread and track its status"""
    def task_wrapper():
        try:
            TASK_STATUS[task_id] = {"status": "processing", "progress": 0}
            result = func(*args, **kwargs)
            TASK_RESULTS[task_id] = result
            TASK_STATUS[task_id] = {"status": "completed", "progress": 100}
        except Exception as e:
            import traceback
            error_msg = str(e)
            tb = traceback.format_exc()
            app.logger.error(f"Background task error: {error_msg}\n{tb}")
            TASK_STATUS[task_id] = {"status": "failed", "error": error_msg, "progress": 0}
    
    thread = Thread(target=task_wrapper)
    thread.daemon = True
    thread.start()
    return task_id

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

        # Process asynchronously to avoid Cloudflare timeout
        # Pass the original filename as a parameter instead of accessing it from session
        task_id = run_task_in_background(
            process_pdf_with_progress,
            conversion_id,
            pdf_path, 
            conversion_id, 
            ocr_engine, 
            language, 
            quality,
            orig_filename  # Pass this as a parameter
        )
        
        # Store task_id in session
        session['task_id'] = conversion_id
        
        # Redirect to status page that will check for completion
        return redirect(url_for('status', task_id=conversion_id))

    else:
        flash('Invalid file type. Please upload a PDF.', 'error')
        return redirect(url_for('index'))

@app.route('/status/<task_id>')
def status(task_id):
    # Show status page with JavaScript to poll for completion
    return render_template('status.html', task_id=task_id)

@app.route('/api/task_status/<task_id>')
def task_status(task_id):
    """API endpoint to check task status"""
    if task_id in TASK_STATUS:
        status_data = TASK_STATUS[task_id].copy()  # Create a copy to avoid modifying the original
        
        # If task is completed, include the path to results
        if status_data.get("status") == "completed" and task_id in TASK_RESULTS:
            success, result_path, output_filename = TASK_RESULTS[task_id]
            if success:
                # Store results in session within this request context
                session['docx_path'] = result_path
                session['output_filename'] = output_filename
                status_data["redirect"] = url_for('success')
            else:
                # Store error
                status_data["error"] = output_filename
                status_data["redirect"] = url_for('index')
                flash(f"Conversion failed: {output_filename}", 'error')
        
        return jsonify(status_data)
    
    return jsonify({"status": "not_found"})

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

@app.route('/api/check-dependency')
def api_check_dependency():
    """API endpoint to check if a specific dependency is installed"""
    name = request.args.get('name', '')
    if not name:
        return jsonify({"error": "No dependency name provided"}), 400
    
    installed, data = check_dependency(name)
    return jsonify(data)

@app.route('/guide')
def guide():
    return render_template('guide.html')

if __name__ == '__main__':
    # Set the start method for multiprocessing to 'spawn' for better compatibility across platforms
    multiprocessing.set_start_method('spawn', force=True)
    
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