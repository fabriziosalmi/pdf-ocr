import os
import sys
import time
import json
import subprocess
import logging
import logging.handlers
import secrets
from flask import Flask, request, render_template, send_file, flash, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from docx import Document
from PIL import Image
import uuid
from threading import Thread
import shutil
import tempfile
import re
from datetime import timedelta
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

# Configure logging. The log file is opt-in: a container or a read-only
# deployment should not fail to boot because the working directory is not
# writable, and stdout is what an orchestrator actually collects.
_log_handlers: list = [logging.StreamHandler()]
_log_file = os.environ.get('LOG_FILE')
if _log_file:
    try:
        # Rotate, so a long-running instance cannot fill the disk.
        _log_handlers.append(logging.handlers.RotatingFileHandler(
            _log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding='utf-8'
        ))
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        print(f"Warning: could not open LOG_FILE {_log_file!r}: {exc}", file=sys.stderr)

logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=_log_handlers,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}
SUPPORTED_ENGINES = {'tesseract', 'easyocr', 'pyocr', 'paddleocr'}
SUPPORTED_OUTPUT_FORMATS = {'docx', 'txt', 'md', 'html'}


def env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer setting, failing with a message that names it.

    These are documented, user-set knobs; a bare
    `ValueError: invalid literal for int()` at import time gives an operator
    nothing to go on when the container crash-loops.
    """
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = env_int('MAX_UPLOAD_MB', 64) * 1024 * 1024
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)  # Session timeout
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# Only send the session cookie over HTTPS when serving over TLS.
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'false').lower() == 'true'

# Docker-specific configuration
DOCKER_ENV = os.environ.get('DOCKER_ENV', 'false').lower() == 'true'


def _resolve_secret_key() -> str:
    """Return the session signing key, refusing to run unpinned in production.

    A key generated at import time differs in every gunicorn worker and changes
    on every restart, so session cookies silently fail to validate as soon as
    more than one worker serves traffic. Require SECRET_KEY unless the app is
    explicitly running in development or under test.
    """
    key = os.environ.get('SECRET_KEY')
    if key:
        return key
    if os.environ.get('FLASK_ENV') == 'development' or os.environ.get('PDF_OCR_TESTING') == '1':
        logger.warning(
            "SECRET_KEY is not set; using an ephemeral development key. Sessions "
            "will not survive a restart and will break with more than one worker."
        )
        return secrets.token_hex(32)
    raise RuntimeError(
        "SECRET_KEY environment variable is required. Generate one with:\n"
        "  python -c 'import secrets; print(secrets.token_hex(32))'\n"
        "and set it in the environment (see .env.example). For local development "
        "set FLASK_ENV=development instead."
    )


app.secret_key = _resolve_secret_key()

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Setup periodic cleanup
CLEANUP_INTERVAL = 3600  # 1 hour in seconds
TASK_TIMEOUT = 3600  # 1 hour in seconds
# A conversion refreshes its record after every page, so a "processing" task
# that has not been touched for this long is not slow, it is gone: the worker
# was recycled or the container restarted mid-conversion. Without this the
# progress page polls a task that will never finish.
STALE_TASK_TIMEOUT = env_int('STALE_TASK_TIMEOUT', 1800)
LAST_CLEANUP_TIME = time.time()


def log_safe(value: Any, limit: int = 200) -> str:
    """Flatten a value for logging so it cannot forge log structure.

    Newlines in a logged value let an attacker append what look like genuine
    log lines. Most of the values we log are already run through
    `secure_filename` or an allowlist, but that guarantee lives far from the
    log call — making it local means it cannot be lost by a later refactor.
    """
    text = str(value).replace('\r', ' ').replace('\n', ' ')
    return text[:limit] + '...' if len(text) > limit else text


class TaskStore:
    """Filesystem-backed store for background conversion tasks.

    The previous implementation kept task state in two module-level dicts.
    Under gunicorn with more than one worker the upload lands in one process
    and the status poll lands in another, so the progress page reported
    "not found" for a task that was running fine — the app only ever worked
    single-process. One JSON file per task in the shared upload volume is the
    smallest thing that survives multiple workers and a restart.
    """

    def _dir(self) -> Path:
        # Read the folder from the config on every call, so tests (and any
        # deployment overriding UPLOAD_FOLDER) point at the right place.
        path = Path(app.config['UPLOAD_FOLDER']) / '.tasks'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _path(self, task_id: str) -> Optional[Path]:
        # Task ids are server-generated UUIDs; reject anything else rather than
        # letting a crafted id escape the task directory.
        if not re.fullmatch(r'[0-9a-fA-F-]{1,64}', task_id or ''):
            return None
        return self._dir() / f"{task_id}.json"

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(task_id)
        if path is None or not path.is_file():
            return None
        try:
            with path.open(encoding='utf-8') as fh:
                return json.load(fh)
        except (OSError, ValueError) as exc:
            logger.error(f"Could not read task {log_safe(task_id)}: {exc}")
            return None

    def set(self, task_id: str, data: Dict[str, Any]) -> None:
        path = self._path(task_id)
        if path is None:
            return
        data = {**data, "timestamp": time.time()}
        # Write-then-rename, so a concurrent reader never sees a partial file.
        tmp = path.with_suffix('.tmp')
        try:
            with tmp.open('w', encoding='utf-8') as fh:
                json.dump(data, fh)
            tmp.replace(path)
        except OSError as exc:
            logger.error(f"Could not write task {log_safe(task_id)}: {exc}")

    def update(self, task_id: str, **fields: Any) -> None:
        current = self.get(task_id)
        if current is None:
            return
        current.update(fields)
        self.set(task_id, current)

    def delete(self, task_id: str) -> None:
        path = self._path(task_id)
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(f"Could not delete task {log_safe(task_id)}: {exc}")

    def items(self):
        for path in sorted(self._dir().glob('*.json')):
            record = self.get(path.stem)
            if record is not None:
                yield path.stem, record

    def clear(self) -> None:
        for path in self._dir().glob('*.json'):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


TASKS = TaskStore()

def allowed_file(filename: Optional[str]) -> bool:
    """Check if a file extension is allowed."""
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_clean_filename(filename: str) -> str:
    """Secure and sanitize the filename."""
    filename = secure_filename(filename)
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = filename.replace(' ', '_')
    return filename


def is_within_upload_folder(path: str) -> bool:
    """Return True if `path` resolves inside the configured upload folder.

    Compared with `Path.resolve()` and a parent check rather than a string
    prefix: `"uploads_evil/x".startswith("uploads")` is True, which the old
    check in the download route would have accepted.
    """
    try:
        root = Path(app.config['UPLOAD_FOLDER']).resolve()
        target = Path(path).resolve()
    except (OSError, ValueError):
        return False
    return target == root or root in target.parents


def looks_like_pdf(stream) -> bool:
    """Check the PDF magic bytes on an uploaded stream, then rewind it.

    The extension check alone says nothing about the content: anything named
    `.pdf` was previously saved to disk and handed straight to Poppler.
    """
    try:
        head = stream.read(5)
        stream.seek(0)
    except (OSError, ValueError):
        return False
    return head == b'%PDF-'

def cleanup_old_files() -> None:
    """Remove old files from the uploads directory and expired tasks."""
    global LAST_CLEANUP_TIME
    current_time = time.time()
    if current_time - LAST_CLEANUP_TIME < CLEANUP_INTERVAL:
        return
    LAST_CLEANUP_TIME = current_time
    logger.info("Running periodic cleanup")
    try:
        upload_path = Path(app.config['UPLOAD_FOLDER'])
        for file_path in upload_path.iterdir():
            if file_path.is_file() and current_time - file_path.stat().st_mtime > 86400:
                try:
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting old file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error during file cleanup: {e}")
    for task_id, record in list(TASKS.items()):
        if record.get("timestamp", 0) + TASK_TIMEOUT < current_time:
            # Drop the produced file along with the record, otherwise results
            # linger for a full day after their task has expired.
            result_path = record.get("result_path")
            if result_path and is_within_upload_folder(result_path):
                try:
                    os.remove(result_path)
                except OSError:
                    pass
            TASKS.delete(task_id)
            logger.info(f"Removed expired task: {task_id}")

_dependency_check_cache: Dict[str, Tuple[float, Tuple[bool, str]]] = {}

def check_dependencies() -> Tuple[bool, str]:
    """Check if all required dependencies are installed. Caches result for 60s."""
    cache_key = 'all'
    now = time.time()
    if cache_key in _dependency_check_cache:
        ts, result = _dependency_check_cache[cache_key]
        if now - ts < 60:
            return result
    try:
        if DOCKER_ENV:
            result = (True, "Running in Docker, dependencies assumed to be installed")
            _dependency_check_cache[cache_key] = (now, result)
            return result
        # Poppler ships the `pdftoppm` binary that pdf2image shells out to.
        # (The previous probe called `convert_from_path.get_page_count`, an
        # attribute that does not exist on that function, so it raised
        # AttributeError, failed the string match, and never reported anything.)
        try:
            subprocess.check_output(['pdftoppm', '-v'], stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, FileNotFoundError):
            hint = "brew install poppler" if sys.platform == 'darwin' else "apt-get install poppler-utils"
            result = (False, f"Poppler is not installed or not in PATH. Install it with '{hint}'")
            _dependency_check_cache[cache_key] = (now, result)
            return result
        try:
            subprocess.check_output(['tesseract', '--version'], stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, FileNotFoundError):
            result = (False, "Tesseract OCR is not installed or not in PATH. On macOS, install it with 'brew install tesseract'")
            _dependency_check_cache[cache_key] = (now, result)
            return result
        result = (True, "All dependencies are properly installed")
        _dependency_check_cache[cache_key] = (now, result)
        return result
    except Exception as e:
        logger.error(f"Error checking dependencies: {e}", exc_info=True)
        result = (False, "Error checking dependencies. See the server log for details.")
        _dependency_check_cache[cache_key] = (now, result)
        return result

def check_dependency(name: str) -> Tuple[bool, Dict[str, Any]]:
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
                langs = [lang.strip() for lang in langs_output.split('\n')[1:] if lang.strip()]
                
                return True, {
                    "installed": True, 
                    "version": version, 
                    "languages": langs,
                    "message": "Tesseract is installed"
                }
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False, {"installed": False, "message": "Tesseract is not installed or not in PATH"}

        elif name.lower() == 'paddleocr':
            # Optional engine: lazy-import test so its (heavy) absence never breaks the check.
            try:
                import paddleocr
                from paddleocr import PaddleOCR  # noqa: F401
                version = getattr(paddleocr, '__version__', 'Unknown version')
                return True, {"installed": True, "version": version, "message": "PaddleOCR is installed"}
            except Exception:
                return False, {"installed": False, "message": "PaddleOCR is not installed. Install with 'pip install -r requirements-paddleocr.txt'"}

        else:
            return False, {"installed": False, "message": f"Unknown dependency: {name}"}
    
    except Exception as e:
        # The detail goes to the log, not to an anonymous HTTP caller: this
        # endpoint needs no authentication, and exception text leaks paths and
        # library internals.
        logger.error(f"Error checking dependency {log_safe(name)}: {e}", exc_info=True)
        return False, {"installed": False, "message": f"Error checking {log_safe(name)}"}

# Endpoints that must not create a session. Touching `session.permanent`
# writes `_permanent` into the session dict, which marks it modified and makes
# Flask emit Set-Cookie — so an uptime monitor polling /healthz was being
# handed a cookie on every probe.
SESSIONLESS_ENDPOINTS = {'healthz', 'system_check', 'api_check_dependency', 'static'}


@app.before_request
def before_request():
    """Run before each request to perform housekeeping."""
    # Run cleanup periodically
    cleanup_old_files()

    if request.endpoint not in SESSIONLESS_ENDPOINTS:
        # Make session permanent but with a timeout
        session.permanent = True


# The page loads Tailwind from static/vendor and defines inline styles/handlers,
# so 'unsafe-inline' is still required; everything else is same-origin only.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "object-src 'none'"
)


@app.after_request
def set_security_headers(response):
    """Attach baseline security headers to every response."""
    response.headers.setdefault('Content-Security-Policy', CONTENT_SECURITY_POLICY)
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'no-referrer')
    response.headers.setdefault('Cross-Origin-Opener-Policy', 'same-origin')
    response.headers.setdefault('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
    if app.config['SESSION_COOKIE_SECURE']:
        response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return response


@app.route('/healthz')
def healthz():
    """Liveness/readiness probe: is the process up and the upload folder usable?"""
    upload_dir = app.config['UPLOAD_FOLDER']
    writable = os.path.isdir(upload_dir) and os.access(upload_dir, os.W_OK)
    body = {"status": "ok" if writable else "degraded", "upload_dir_writable": writable}
    return jsonify(body), (200 if writable else 503)

@app.route('/')
def index():
    """Home page route."""
    # Check if dependencies are properly installed
    deps_installed, message = check_dependencies()
    if not deps_installed:
        flash(message, 'error')
    return render_template('index.html')

def sanitize_text(text: Optional[str]) -> str:
    """Sanitize text by removing control characters."""
    if not text:
        return ""
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)

DEFAULT_PREPROCESS_OPTIONS: Dict[str, Any] = {
    "grayscale": True,
    "sharpen": True,
    "contrast": 1.5,
    "threshold": False,
}


def parse_preprocess_options(form) -> Dict[str, Any]:
    """Read the preprocessing checkboxes/slider off the upload form."""
    try:
        contrast = float(form.get('pre-contrast', DEFAULT_PREPROCESS_OPTIONS["contrast"]))
    except (TypeError, ValueError):
        contrast = DEFAULT_PREPROCESS_OPTIONS["contrast"]
    return {
        "grayscale": form.get('pre-grayscale') == '1',
        "sharpen": form.get('pre-sharpen') == '1',
        # The slider runs 0.5-2.5; clamp so a hand-crafted request cannot ask
        # for an absurd enhancement factor.
        "contrast": min(max(contrast, 0.5), 2.5),
        "threshold": form.get('pre-threshold') == '1',
    }


def otsu_threshold(image: Image.Image) -> int:
    """Pick a global binarisation cutoff with Otsu's method.

    Computed from the 8-bit histogram so that thresholding does not require
    OpenCV/NumPy, which would add ~60 MB to the image for one checkbox.
    """
    histogram = image.histogram()[:256]
    total = sum(histogram)
    if total == 0:
        return 128

    sum_all = sum(level * count for level, count in enumerate(histogram))
    sum_background = 0.0
    weight_background = 0
    best_variance = -1.0
    best_cutoff = 128

    for level, count in enumerate(histogram):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += level * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_cutoff = level

    return best_cutoff


def enhance_image(image: Image.Image, options: Optional[Dict[str, Any]] = None) -> Image.Image:
    """Enhance image quality for better OCR results.

    Every step here is backed by a control in the upload form. Options the UI
    used to offer but nothing implemented (denoise, deskew, border removal and
    the preset profiles) have been removed from the form rather than stubbed.
    """
    settings = {**DEFAULT_PREPROCESS_OPTIONS, **(options or {})}
    try:
        # Import here to avoid requiring these packages unless needed
        from PIL import ImageEnhance, ImageFilter

        # Apply a slight sharpening filter
        if settings["sharpen"]:
            image = image.filter(ImageFilter.SHARPEN)

        # Adjust contrast
        contrast = float(settings["contrast"])
        if abs(contrast - 1.0) > 0.01:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)

        # Convert to grayscale if not already (thresholding needs it too)
        if (settings["grayscale"] or settings["threshold"]) and image.mode != 'L':
            image = image.convert('L')

        # Binarise, which usually helps Tesseract on clean scans and hurts on
        # photographs, hence off by default.
        if settings["threshold"]:
            cutoff = otsu_threshold(image)
            image = image.point(lambda p: 255 if p > cutoff else 0, mode='L')

        return image
    except Exception as e:
        logger.warning(f"Image enhancement failed: {e}")
        return image  # Return original image if enhancement fails

def process_image(i: int, image_path: str, ocr_engine: str, language: str, preprocess: bool = False, preprocess_options: Optional[Dict[str, Any]] = None) -> Tuple[int, str]:
    """Run OCR on a single rendered page and return (page index, text)."""
    img_to_process = None
    preprocessed_path = None
    try:
        text = ""
        # Open image
        logger.debug(f"Attempting to open image: {image_path}")  # Debugging log
        img_to_process = Image.open(image_path)
        logger.debug(f"Image opened successfully: {image_path}")  # Debugging log

        # Preprocess image if requested
        if preprocess:
            img_to_process = enhance_image(img_to_process, preprocess_options)
            # EasyOCR and PaddleOCR read the file from disk rather than taking
            # the PIL object, so without writing the enhanced image back out
            # they silently received the untouched page.
            preprocessed_path = f"{image_path}.pre.png"
            try:
                img_to_process.save(preprocessed_path, 'PNG')
                image_path = preprocessed_path
            except Exception as e:
                logger.warning(f"Could not persist preprocessed page {i+1}: {e}")
                preprocessed_path = None

        # Log OCR engine being used for debugging
        logger.info(f"Processing page {i+1} with OCR engine: {ocr_engine}")
        
        if ocr_engine == "tesseract":
            try:
                # Tesseract configuration for better accuracy
                config = f"--oem 1 --psm 3 -l {language}"
                if 'eng' in language and '+' not in language:
                    # Add extra parameters for English for better accuracy
                    config += " --dpi 300"
                
                # Verify tesseract is available
                tesseract_version = pytesseract.get_tesseract_version()
                logger.info(f"Using Tesseract version: {tesseract_version}")
                
                # Log what language is being used
                logger.info(f"OCR language settings: {language}, config: {config}")
                
                text = pytesseract.image_to_string(img_to_process, config=config)
                if not text.strip():
                    logger.warning(f"Empty OCR result for page {i+1}. Trying alternative method.")
                    # Try with a different PSM mode if empty
                    config = f"--oem 1 --psm 6 -l {language}"
                    text = pytesseract.image_to_string(img_to_process, config=config)
            except Exception as e:
                logger.error(f"Tesseract OCR error: {str(e)}", exc_info=True)
                return i, f"[Error with Tesseract OCR: {str(e)}]"
        
        elif ocr_engine == "easyocr":
            try:
                import easyocr
                # Map common 3-letter ISO codes to 2-letter EasyOCR codes
                lang_map = {
                    'eng': 'en', 'fra': 'fr', 'deu': 'de', 'spa': 'es', 'ita': 'it', 'por': 'pt', 
                    'chi_sim': 'ch_sim', 'chi_tra': 'ch_tra', 'jpn': 'ja', 'kor': 'ko', 'rus': 'ru', 
                    'ara': 'ar', 'hin': 'hi'
                }
                
                # Parse and map languages (handling multiple languages separated by +)
                langs_to_load = []
                for lang in language.split('+'):
                    if lang in lang_map:
                        langs_to_load.append(lang_map[lang])
                    else:
                        langs_to_load.append(lang)
                
                # Initialize reader with all requested languages
                reader = easyocr.Reader(langs_to_load)
                
                # Process with EasyOCR (using the file path directly)
                result = reader.readtext(image_path, detail=0, paragraph=True)
                text = '\n'.join(result) if result else ""
            except Exception as e:
                logger.error(f"EasyOCR error: {str(e)}", exc_info=True)
                return i, f"[Error with EasyOCR: {str(e)}]"
        
        elif ocr_engine == "pyocr":
            try:
                import pyocr
                import pyocr.builders
                
                # Get available tools (should be Tesseract or Cuneiform)
                tools = pyocr.get_available_tools()
                if len(tools) == 0:
                    return i, "[Error: No OCR tool found for PyOCR. Install Tesseract or Cuneiform.]"
                
                # Use the first available tool (typically Tesseract)
                tool = tools[0]
                
                # Map Tesseract language codes to PyOCR
                # PyOCR uses the same language codes as Tesseract
                
                # Perform OCR
                text = tool.image_to_string(
                    img_to_process,
                    lang=language,
                    builder=pyocr.builders.TextBuilder()
                )
            except Exception as e:
                logger.error(f"PyOCR error: {str(e)}", exc_info=True)
                return i, f"[Error with PyOCR: {str(e)}]"

        elif ocr_engine == "paddleocr":
            try:
                import paddleocr as paddleocr_module
                from paddleocr import PaddleOCR

                # This branch is written against the 2.x API. Say so plainly
                # rather than letting 3.x fail with a bare TypeError about an
                # unexpected keyword argument, which points at nothing useful.
                installed_version = str(getattr(paddleocr_module, '__version__', ''))
                if installed_version and not installed_version.startswith('2.'):
                    return i, (
                        f"[Error: PaddleOCR {installed_version} is installed, but this app "
                        f"targets the 2.x API. Install it with "
                        f"'pip install -r requirements-paddleocr.txt', or use another engine.]"
                    )

                # Map common ISO codes (3-letter Tesseract or 2-letter) to PaddleOCR codes
                lang_map = {
                    'eng': 'en', 'en': 'en', 'ita': 'it', 'it': 'it',
                    'fra': 'fr', 'fr': 'fr', 'deu': 'german', 'de': 'german',
                    'spa': 'es', 'es': 'es', 'chi_sim': 'ch', 'ch': 'ch',
                }
                # PaddleOCR loads a single language model per reader; use the first requested language
                first_lang = language.split('+')[0]
                paddle_lang = lang_map.get(first_lang, 'en')

                # PaddleOCR 2.x API (use_angle_cls / cls / show_log were removed in 3.x)
                reader = PaddleOCR(use_angle_cls=True, lang=paddle_lang, show_log=False)
                result = reader.ocr(image_path, cls=True)

                # Defensively extract text from the PaddleOCR 2.x nested structure:
                # result = [ [ [box], (text, confidence) ], ... ]  (outer list is per-image)
                lines = []
                if result:
                    for page in result:
                        if not page:
                            continue
                        for entry in page:
                            try:
                                line = entry[1][0]
                            except (IndexError, TypeError):
                                # Skip malformed entries
                                continue
                            if isinstance(line, str) and line:
                                lines.append(line)
                text = '\n'.join(lines)
            except Exception as e:
                logger.error(f"PaddleOCR error: {str(e)}", exc_info=True)
                return i, f"[Error with PaddleOCR: {str(e)}]"

        else:
            return i, f"[Error: Unsupported OCR engine: {ocr_engine}]"

        # Sanitize text
        text = sanitize_text(text)
        
        # Attempt to detect and fix common OCR errors
        text = fix_common_ocr_errors(text)
        
        return i, text
    except FileNotFoundError as e:
        logger.error(f"File not found error processing page {i+1}: {str(e)}", exc_info=True)
        return i, f"[Error: File not found: {str(e)}. Ensure the file exists and is accessible.]"
    except Exception as e:
        logger.error(f"Error processing page {i+1} with {ocr_engine}: {str(e)}", exc_info=True)
        return i, f"[Error processing page {i+1}: {str(e)}]"
    finally:
        # Ensure PIL Image is properly closed to prevent resource leaks
        if img_to_process is not None:
            try:
                img_to_process.close()
            except Exception as e:
                logger.warning(f"Error closing image for page {i+1}: {str(e)}")
        # The rendered page itself is owned by the caller, but the preprocessed
        # copy is ours to remove.
        if preprocessed_path:
            try:
                os.remove(preprocessed_path)
            except OSError:
                pass

def fix_common_ocr_errors(text: str, reflow: bool = False) -> str:
    """Tidy up OCR output without altering the characters the engine recognised.

    This function is deliberately conservative. An earlier version applied blind
    global substitutions ('0'->'O', '1'->'I', '5'->'S', 'rn'->'m', 'cl'->'d'),
    which corrupted every digit in every document: an invoice total of
    "1,250.00 EUR" came out as "I,2SO.OO EUR". Character-level disambiguation is
    not decidable without per-glyph confidence data, which we do not have here,
    so we do not guess. Only whitespace/punctuation layout is normalised:

    - join words split by a hyphen at a line break ("exam-\\nple" -> "example"),
      which is a genuine artefact of the source layout, not of the OCR engine;
    - drop spaces inserted before closing punctuation;
    - strip trailing whitespace on each line;
    - collapse runs of 3+ blank lines into a single paragraph break.

    If `reflow` is True, single newlines inside a paragraph are turned into
    spaces. That helps continuous prose but destroys line-oriented documents
    (tables, addresses, invoices), so it is opt-in and off by default.
    """
    if not text:
        return text

    # Re-join words hyphenated across a line break.
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Remove spaces/tabs before closing punctuation (but never across a newline).
    text = re.sub(r'[ \t]+([,.;:!?])', r'\1', text)

    # Strip trailing whitespace on every line.
    text = re.sub(r'[ \t]+(?=\n|$)', '', text)

    if reflow:
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Collapse excessive blank lines into a single paragraph break.
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text

def save_as_markdown(text_results: Dict[int, str], output_path: str) -> None:
    """Save the extracted text results as a Markdown file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in sorted(text_results.keys()):
            text = text_results[i]
            # Basic Markdown: treat paragraphs separated by double newlines
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                f.write(para.strip() + '\n\n') # Add double newline after each paragraph
            # Add a horizontal rule as a page separator (optional)
            if i < max(text_results.keys()):
                f.write('---\n\n')

def escape_html(text: str) -> str:
    """Escape the characters that would otherwise close out of a text node."""
    return (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                .replace('"', '&quot;'))


def save_as_html(text_results: Dict[int, str], output_path: str, title: str = "Converted Document") -> None:
    """Save the extracted text results as a basic HTML file."""
    # The title reaches here from the uploaded filename. secure_clean_filename
    # already strips angle brackets, so this is defence in depth rather than a
    # live hole — but the paragraphs below were escaped and the title was not,
    # which is exactly the asymmetry that becomes a hole after a refactor.
    title = escape_html(title)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('<!DOCTYPE html>\n')
        f.write('<html lang="en">\n')
        f.write('<head>\n')
        f.write('    <meta charset="UTF-8">\n')
        f.write(f'    <title>{title}</title>\n')
        f.write('    <style>body { font-family: sans-serif; line-height: 1.6; } .page-break { page-break-after: always; }</style>\n')
        f.write('</head>\n')
        f.write('<body>\n')
        f.write(f'<h1>{title}</h1>\n')
        
        for i in sorted(text_results.keys()):
            text = text_results[i]
            # Basic HTML: treat paragraphs separated by double newlines
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                # Escape basic HTML characters
                escaped_para = escape_html(para)
                f.write(f'<p>{escaped_para.strip()}</p>\n')
            # Add a visual separator or page break indicator
            if i < max(text_results.keys()):
                f.write('<hr class="page-break">\n')
                
        f.write('</body>\n')
        f.write('</html>\n')

MAX_PAGES = env_int('MAX_PAGES', 200)
# Pages rendered per Poppler call. Rendering the whole document in one go
# materialises every page as a full-resolution bitmap at once: a 100-page PDF
# at 600 DPI is several GB of RSS, which the 2 GB container limit cannot hold.
RENDER_BATCH_SIZE = env_int('RENDER_BATCH_SIZE', 4)


def save_output(results: Dict[int, str], output_format: str, output_path: str, base_filename: str) -> None:
    """Write the per-page OCR text out in the requested format."""
    if output_format == "docx":
        document = Document()
        ordered = sorted(results.keys())
        for position, i in enumerate(ordered):
            document.add_paragraph(results[i])
            if position < len(ordered) - 1:
                document.add_page_break()
        document.save(output_path)
    elif output_format == "txt":
        with open(output_path, 'w', encoding='utf-8') as f:
            ordered = sorted(results.keys())
            for position, i in enumerate(ordered):
                f.write(results[i])
                # Add a separator between pages for clarity
                if position < len(ordered) - 1:
                    f.write("\n\n--- Page Break ---\n\n")
    elif output_format == "md":
        save_as_markdown(results, output_path)
    elif output_format == "html":
        save_as_html(results, output_path, title=base_filename)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def process_pdf_with_progress(pdf_path: str, conversion_id: str, ocr_engine: str = "tesseract", language: str = "eng", quality: str = "standard", preprocess: bool = False, orig_filename: Optional[str] = None, output_format: str = "docx", preprocess_options: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str], str]:
    """Render a PDF page by page, OCR each page, and write the chosen format."""
    if output_format not in {"docx", "txt", "md", "html"}:
        return False, None, f"Unsupported output format: {output_format}"

    temp_dir = None
    try:
        # Create a temporary directory for image files
        temp_dir = tempfile.mkdtemp(prefix="ocr_")
        logger.info(f"Created temporary directory: {temp_dir}")

        # Import PDF conversion library
        from pdf2image import convert_from_path, pdfinfo_from_path

        # Convert PDF to images with appropriate DPI based on quality
        dpi = 600 if quality == "high" else 300

        TASKS.update(conversion_id, status="processing", step="converting", progress=0)

        # The page count decides how the work is batched, so it is required
        # rather than best-effort: without it the whole document has to be
        # rendered at once just to find out how long it is.
        pdf_info = pdfinfo_from_path(pdf_path)
        total_pages = int(pdf_info["Pages"])
        if total_pages < 1:
            raise ValueError("The PDF reports zero pages.")
        if total_pages > MAX_PAGES:
            raise ValueError(
                f"This PDF has {total_pages} pages; the limit is {MAX_PAGES} "
                f"(raise MAX_PAGES to allow more)."
            )

        start_time = time.time()
        logger.info(
            f"Processing {total_pages} page(s) at {dpi} DPI, engine: {ocr_engine}, "
            f"lang: {language}, output: {output_format}"
        )

        results: Dict[int, str] = {}
        # Render and OCR in batches, discarding each page's bitmap as soon as
        # its text has been extracted so peak memory stays bounded.
        for batch_start in range(0, total_pages, RENDER_BATCH_SIZE):
            batch_end = min(batch_start + RENDER_BATCH_SIZE, total_pages)
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=batch_start + 1,   # Poppler page numbers are 1-based
                last_page=batch_end,
                thread_count=1,
                use_pdftocairo=True,
                fmt='png',
            )

            for offset, img in enumerate(images):
                i = batch_start + offset
                img_path = os.path.join(temp_dir, f'page_{i}.png')
                try:
                    img.save(img_path, 'PNG')
                finally:
                    img.close()

                if not (os.path.exists(img_path) and os.path.getsize(img_path) > 0):
                    logger.error(f"Failed to save page {i + 1} to {img_path}")
                    results[i] = f"[Error: page {i + 1} could not be rendered]"
                    continue

                try:
                    _, text = process_image(i, img_path, ocr_engine, language, preprocess, preprocess_options)
                    results[i] = text
                finally:
                    try:
                        os.remove(img_path)
                    except OSError:
                        pass

                # Allocate 5-95% of the progress bar to the OCR pass.
                progress = 5 + int(((i + 1) / total_pages) * 90)
                TASKS.update(conversion_id, step="ocr", progress=progress)

            del images

        if not results:
            raise FileNotFoundError("No pages could be rendered from the PDF.")

        TASKS.update(conversion_id, step="assembling", progress=95)

        # Determine output filename and path
        document_name = orig_filename or 'document.pdf'
        base_filename = os.path.splitext(document_name)[0]
        output_filename = f"{secure_clean_filename(base_filename)}.{output_format}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{conversion_id}_{output_filename}")

        save_output(results, output_format, output_path, base_filename)

        # Log performance metrics
        elapsed_time = time.time() - start_time
        pages_per_second = total_pages / elapsed_time if elapsed_time > 0 else 0
        logger.info(f"PDF processing completed. Pages: {total_pages}, Time: {elapsed_time:.2f}s, Pages/sec: {pages_per_second:.2f}, Output: {output_path}")

        # (the uploaded PDF is removed in the finally block, on every path)

        return True, output_path, output_filename # Return the actual path and filename

    except ImportError as e:
        # Specific handling for missing OCR engine imports
        error_message = f"Error: The required OCR engine '{ocr_engine}' is not properly installed. Please install it with pip: {str(e)}"
        logger.error(error_message)
        return False, None, error_message
    except Exception as e:
        logger.error(f"Error during PDF processing: {str(e)}", exc_info=True)
        error_message = f"An unexpected error occurred: {str(e)}"
        return False, None, error_message
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}") # Log temp dir cleanup
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {e}")

        # Remove the uploaded PDF whatever the outcome. It used to be deleted
        # only on the success path, so a failed conversion left the user's
        # document sitting in the upload folder until the next daily sweep.
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError as e:
                logger.warning(f"Could not remove uploaded PDF {pdf_path}: {e}")

def run_task_in_background(func: callable, task_id: str, *args: Any, **kwargs: Any) -> str:
    """Run a conversion in a background thread, recording progress in the store."""

    def task_wrapper():
        # The worker touches app.config (upload folder, task store paths), so it
        # needs an application context of its own.
        with app.app_context():
            try:
                success, result_path, output_filename = func(*args, **kwargs)
                if success:
                    TASKS.update(
                        task_id,
                        status="completed",
                        step="done",
                        progress=100,
                        result_path=result_path,
                        output_filename=output_filename,
                    )
                else:
                    # On failure the third element carries the error message.
                    TASKS.update(task_id, status="failed", progress=0, error=output_filename)
            except Exception as e:
                logger.error(f"Background task error: {str(e)}", exc_info=True)
                TASKS.update(task_id, status="failed", progress=0, error=str(e))

    TASKS.set(task_id, {"status": "processing", "step": "initializing", "progress": 0})
    thread = Thread(target=task_wrapper, daemon=True)
    thread.start()
    return task_id

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and initiate OCR processing."""
    try:
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

        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a PDF.', 'error')
            return redirect(url_for('index'))

        # The extension says nothing about the content, and the file is handed
        # straight to Poppler; require the PDF magic bytes too.
        if not looks_like_pdf(file.stream):
            flash('That file is not a PDF (missing %PDF- header).', 'error')
            return redirect(url_for('index'))

        # Validate the form options against allowlists. `output_format` in
        # particular used to be interpolated straight into the output path.
        ocr_engine = request.form.get('ocr-engine', 'tesseract')
        if ocr_engine not in SUPPORTED_ENGINES:
            flash('Unknown OCR engine selected.', 'error')
            return redirect(url_for('index'))

        output_format = request.form.get('output-format', 'docx')
        if output_format not in SUPPORTED_OUTPUT_FORMATS:
            flash('Unknown output format selected.', 'error')
            return redirect(url_for('index'))

        language = request.form.get('language', 'eng')
        if not re.fullmatch(r'[a-zA-Z_]{2,16}(\+[a-zA-Z_]{2,16})*', language):
            flash('Invalid language selection.', 'error')
            return redirect(url_for('index'))

        quality = 'high' if request.form.get('ocr-quality') == 'high' else 'standard'
        preprocess = request.form.get('preprocess', '0') == '1'
        preprocess_options = parse_preprocess_options(request.form) if preprocess else None

        # Generate unique ID for this conversion
        conversion_id = str(uuid.uuid4())
        orig_filename = secure_clean_filename(file.filename) or 'document.pdf'

        # Log processing request
        logger.info(
            f"Processing request: file={log_safe(orig_filename)}, engine={log_safe(ocr_engine)}, "
            f"lang={log_safe(language)}, quality={quality}, preprocess={preprocess}, "
            f"format={log_safe(output_format)}"
        )

        # Create a temporary filename to avoid collisions
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{conversion_id}_{orig_filename}")

        # Save the uploaded file
        try:
            file.save(pdf_path)
            logger.info(f"Saved uploaded file to {pdf_path}")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}", exc_info=True)
            flash("An error occurred while saving the uploaded file. Please try again.", 'error')
            return redirect(url_for('index'))

        # Remember which tasks this browser started, so results are not readable
        # by anyone who learns a task id.
        owned = session.get('owned_tasks', [])
        session['owned_tasks'] = (owned + [conversion_id])[-20:]

        # Process asynchronously
        try:
            run_task_in_background(
                process_pdf_with_progress,
                conversion_id,
                pdf_path,
                conversion_id,
                ocr_engine,
                language,
                quality,
                preprocess,
                orig_filename,
                output_format,
                preprocess_options,
            )
            return redirect(url_for('status', task_id=conversion_id))
        except Exception as e:
            logger.error(f"Error starting background task: {str(e)}", exc_info=True)
            flash("An error occurred while starting the OCR process. Please try again.", 'error')
            return redirect(url_for('index'))

    except Exception as e:
        logger.error(f"Unexpected error during file upload: {str(e)}", exc_info=True)
        flash("An unexpected error occurred. Please try again.", 'error')
        return redirect(url_for('index'))

def owns_task(task_id: str) -> bool:
    """Whether the current browser session started this conversion."""
    return task_id in session.get('owned_tasks', [])


def fail_if_stale(task_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Mark an abandoned conversion as failed instead of leaving it pending.

    The conversion refreshes its record after every page, so a task still
    claiming to be "processing" long after its last update is not slow — its
    worker was recycled or the container restarted. Nothing would ever have
    moved it out of that state, so the progress page polled forever.
    """
    if record.get("status") != "processing":
        return record
    if time.time() - record.get("timestamp", 0) <= STALE_TASK_TIMEOUT:
        return record

    logger.warning(f"Task {task_id} has not progressed in {STALE_TASK_TIMEOUT}s; marking failed")
    TASKS.update(
        task_id,
        status="failed",
        error="The conversion stopped unexpectedly (the server may have restarted). Please try again.",
    )
    return TASKS.get(task_id) or record


def get_owned_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a task record, or None if it is missing or not ours."""
    if not owns_task(task_id):
        return None
    record = TASKS.get(task_id)
    if record is None:
        return None
    return fail_if_stale(task_id, record)


@app.route('/status/<task_id>')
def status(task_id):
    """Display status page for an ongoing conversion."""
    if get_owned_task(task_id) is None:
        flash("The requested conversion was not found.", 'error')
        return redirect(url_for('index'))

    return render_template('status.html', task_id=task_id)

@app.route('/api/task_status/<task_id>')
def task_status(task_id):
    """API endpoint to check task status"""
    record = get_owned_task(task_id)
    if record is None:
        return jsonify({"status": "not_found"}), 404

    # Never expose the on-disk path of the result to the browser.
    status_data = {k: v for k, v in record.items() if k != "result_path"}
    if record.get("status") == "completed":
        status_data["redirect"] = url_for('success', task_id=task_id)
    elif record.get("status") == "failed":
        status_data["redirect"] = url_for('index')

    return jsonify(status_data)

@app.route('/success/<task_id>')
def success(task_id):
    """Display success page after successful conversion."""
    record = get_owned_task(task_id)
    if record is None or record.get("status") != "completed":
        flash('No conversion data found. Please upload a file first.', 'error')
        return redirect(url_for('index'))

    return render_template('success.html', filename=record.get("output_filename"), task_id=task_id)

@app.route('/download/<task_id>')
def download_file(task_id):
    """Provide the converted file for download."""
    record = get_owned_task(task_id)
    if record is None or record.get("status") != "completed":
        flash('No conversion data found. Please upload a file first.', 'error')
        return redirect(url_for('index'))

    result_path = record.get("result_path") or ""
    output_filename = record.get("output_filename") or "download"

    # The path is server-generated, but confirm it still points inside the
    # upload folder before handing the file out.
    if not is_within_upload_folder(result_path):
        logger.error(f"Refusing to serve a result outside the upload folder: {result_path}")
        flash('Access denied.', 'error')
        return redirect(url_for('index'))

    if not os.path.isfile(result_path):
        flash('The converted file is no longer available.', 'error')
        return redirect(url_for('index'))

    # Determine MIME type based on file extension
    _, ext = os.path.splitext(output_filename)
    mime_types = {
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.html': 'text/html',
    }
    mime_type = mime_types.get(ext.lower(), 'application/octet-stream') # Default MIME type

    try:
        return send_file(
            result_path,
            mimetype=mime_type, # Use determined MIME type
            as_attachment=True,
            download_name=output_filename
        )
    except Exception as e:
        logger.error(f"Error during file download: {str(e)}", exc_info=True)
        flash("Error downloading file.", 'error')
        return redirect(url_for('index'))

@app.route('/new_conversion')
@app.route('/new_conversion/<task_id>')
def new_conversion(task_id: Optional[str] = None):
    """Start a new conversion, discarding the result of a previous one."""
    if task_id:
        record = get_owned_task(task_id)
        if record is not None:
            result_path = record.get("result_path")
            if result_path and is_within_upload_folder(result_path):
                try:
                    os.remove(result_path)
                    logger.info(f"Cleaned up result file: {result_path}")
                except OSError as e:
                    logger.warning(f"Could not remove result file {result_path}: {e}")
            TASKS.delete(task_id)
            session['owned_tasks'] = [t for t in session.get('owned_tasks', []) if t != task_id]

    return redirect(url_for('index'))

@app.route('/api/check-dependency')
def api_check_dependency():
    """API endpoint to check if a specific dependency is installed"""
    name = request.args.get('name', '')
    if not name:
        return jsonify({"error": "No dependency name provided"}), 400
    
    installed, data = check_dependency(name)
    return jsonify(data)

@app.route('/system-check')
def system_check():
    """Check system status and dependencies"""
    results = {
        "status": "ok",
        "errors": [],
        "dependencies": {}
    }
    
    # Major.minor only. The full sys.version string carries the patch level,
    # build date and compiler — free fingerprinting for an endpoint that needs
    # no authentication.
    results["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Check Tesseract
    try:
        tesseract_installed, tesseract_data = check_dependency('tesseract')
        results["dependencies"]["tesseract"] = tesseract_data
        
        if not tesseract_installed:
            results["status"] = "error"
            results["errors"].append("Tesseract OCR is not installed or not found in PATH")
    except Exception as e:
        logger.error(f"Error checking Tesseract: {e}", exc_info=True)
        results["dependencies"]["tesseract"] = {"error": "check failed"}
        results["status"] = "error"
        results["errors"].append("Error checking Tesseract. See the server log for details.")
    
    # Check Poppler
    try:
        poppler_installed, poppler_data = check_dependency('poppler')
        results["dependencies"]["poppler"] = poppler_data
        
        if not poppler_installed:
            results["status"] = "error"
            results["errors"].append("Poppler is not installed or not found in PATH")
    except Exception as e:
        logger.error(f"Error checking Poppler: {e}", exc_info=True)
        results["dependencies"]["poppler"] = {"error": "check failed"}
        results["status"] = "error"
        results["errors"].append("Error checking Poppler. See the server log for details.")

    # Check PaddleOCR (optional engine — reported for info, does not gate overall status)
    try:
        _, paddleocr_data = check_dependency('paddleocr')
        results["dependencies"]["paddleocr"] = paddleocr_data
    except Exception as e:
        logger.error(f"Error checking PaddleOCR: {e}", exc_info=True)
        results["dependencies"]["paddleocr"] = {"error": "check failed"}

    # Check upload directory
    upload_dir = app.config['UPLOAD_FOLDER']
    results["upload_dir"] = {
        "path": upload_dir,
        "exists": os.path.exists(upload_dir),
        "writable": os.access(upload_dir, os.W_OK) if os.path.exists(upload_dir) else False
    }
    
    if not results["upload_dir"]["exists"]:
        results["status"] = "error"
        results["errors"].append(f"Upload directory does not exist: {upload_dir}")
    elif not results["upload_dir"]["writable"]:
        results["status"] = "error"
        results["errors"].append(f"Upload directory is not writable: {upload_dir}")
    
    return jsonify(results)

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('error.html', error="Page not found", code=404), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}", exc_info=True)
    return render_template('error.html', error="Internal server error", code=500), 500

@app.errorhandler(413)
def request_entity_too_large(e):
    """Handle file too large errors."""
    flash('The file is too large. Maximum size is 64MB.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Initial dependency check
    deps_installed, message = check_dependencies()
    if not deps_installed:
        logger.warning(f"Dependency issue: {message}")
        print(f"Warning: {message}")
        print("The application will start but may not work correctly until all dependencies are installed.")

    # Server configuration
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 8011))
    debug = os.environ.get('FLASK_ENV') == 'development'

    app.run(debug=debug, host=host, port=port)