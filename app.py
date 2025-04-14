import os
import io
import sys
import subprocess
from flask import Flask, request, render_template, send_file, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import pytesseract
from docx import Document
from PIL import Image
import uuid
import time

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024 # 16 MB limit
app.secret_key = 'supersecretkey' # Change this in production!

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
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
        
        # Create unique filename to avoid collisions
        filename = f"{conversion_id}_{secure_filename(file.filename)}"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        session['pdf_path'] = pdf_path

        try:
            # Import here to avoid issues if not installed
            from pdf2image import convert_from_path
            
            # Convert PDF to images
            images = convert_from_path(pdf_path)

            # Create a DOCX document
            document = Document()
            full_text = ""

            # Perform OCR on each image
            for i, image in enumerate(images):
                # Save image temporarily to perform OCR
                temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{conversion_id}_page_{i}.png')
                image.save(temp_image_path, 'PNG')

                # Use pytesseract to do OCR on the image
                text = pytesseract.image_to_string(Image.open(temp_image_path))
                full_text += text + "\n\n" # Add page break representation

                # Add paragraph for each page with a page break
                document.add_paragraph(text)
                if i < len(images) - 1:  # Don't add page break after the last page
                    document.add_page_break()
                
                os.remove(temp_image_path) # Clean up temp image

            # Save DOCX to file system
            output_filename = os.path.splitext(orig_filename)[0] + '.docx'
            docx_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{conversion_id}_{output_filename}")
            document.save(docx_path)
            
            # Store path in session for download
            session['docx_path'] = docx_path
            session['output_filename'] = output_filename
            
            # Redirect to success page
            return redirect(url_for('success'))

        except ImportError:
            # Clean up if error occurs
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            flash('PDF2Image library is not installed properly. Please run "pip install pdf2image"', 'error')
            return redirect(url_for('index'))
        except Exception as e:
            # Clean up if error occurs
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                
            error_message = str(e)
            if "Unable to get page count. Is poppler installed and in PATH?" in error_message:
                flash('Poppler is not installed or not in your PATH. On macOS, install it with "brew install poppler"', 'error')
            else:
                flash(f'An error occurred during processing: {error_message}', 'error')
            return redirect(url_for('index'))
        finally:
            # Ensure temporary image files are cleaned up even if loop breaks
            if 'images' in locals():
                for i in range(len(images)):
                    temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{conversion_id}_page_{i}.png')
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

    else:
        flash('Invalid file type. Please upload a PDF.', 'error')
        return redirect(url_for('index'))

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
    
    app.run(debug=True)