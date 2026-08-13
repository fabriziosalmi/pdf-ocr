import os

# The app refuses to start without SECRET_KEY outside development; declare that
# these are tests before importing it.
os.environ.setdefault('PDF_OCR_TESTING', '1')

import unittest
import tempfile
import shutil
import json
import time
import threading
from unittest.mock import patch, MagicMock
from PIL import Image
import colorama
from colorama import Fore, Style

from app import (  # noqa: E402
    app, allowed_file, secure_clean_filename,
    check_dependency, sanitize_text, enhance_image,
    process_image, fix_common_ocr_errors, save_as_markdown, save_as_html,
    is_within_upload_folder, looks_like_pdf, save_output, cleanup_old_files,
    process_pdf_with_progress, parse_preprocess_options, otsu_threshold,
    run_task_in_background, env_int, log_safe, ConversionCancelled,
    TASKS, TASK_TIMEOUT, STALE_TASK_TIMEOUT
)

def _temp_path(suffix: str) -> str:
    """Reserve a temporary path without tempfile.mktemp.

    `tempfile.mktemp` only returns a name, leaving a window in which another
    process can create the file first; mkstemp creates it atomically.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


# Initialize colorama for colored terminal output
colorama.init(autoreset=True)

# Custom test result class for colored output
class ColorTextTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        self.stream.write(f"{Fore.GREEN}✓{Style.RESET_ALL} ")
        super().addSuccess(test)
        
    def addError(self, test, err):
        self.stream.write(f"{Fore.RED}✗{Style.RESET_ALL} ")
        super().addError(test, err)
        
    def addFailure(self, test, err):
        self.stream.write(f"{Fore.RED}✗{Style.RESET_ALL} ")
        super().addFailure(test, err)
        
    def addSkip(self, test, reason):
        self.stream.write(f"{Fore.YELLOW}s{Style.RESET_ALL} ")
        super().addSkip(test, reason)
        
    def printErrorList(self, flavour, errors):
        for test, err in errors:
            self.stream.writeln(self.separator1)
            self.stream.writeln(f"{Fore.RED if flavour == 'ERROR' else Fore.YELLOW}{flavour}: {self.getDescription(test)}{Style.RESET_ALL}")
            self.stream.writeln(self.separator2)
            self.stream.writeln(f"{err}")

# Custom test runner that uses our colored result class
class ColorTextTestRunner(unittest.TextTestRunner):
    def __init__(self, **kwargs):
        kwargs.setdefault('resultclass', ColorTextTestResult)
        super().__init__(**kwargs)

class TestOCRApp(unittest.TestCase):
    
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        
        # Create temp upload folder for testing
        self.test_upload_folder = tempfile.mkdtemp()
        self.original_upload_folder = app.config['UPLOAD_FOLDER']
        app.config['UPLOAD_FOLDER'] = self.test_upload_folder
    
    def tearDown(self):
        # Clean up after tests
        TASKS.clear()
        shutil.rmtree(self.test_upload_folder, ignore_errors=True)
        app.config['UPLOAD_FOLDER'] = self.original_upload_folder
        self.app_context.pop()

        TASKS.clear()
    
    def test_allowed_file(self):
        self.assertTrue(allowed_file('test.pdf'))
        self.assertFalse(allowed_file('test.docx'))
        self.assertFalse(allowed_file(''))
        self.assertFalse(allowed_file(None))
    
    def test_secure_clean_filename(self):
        self.assertEqual(secure_clean_filename('test.pdf'), 'test.pdf')
        self.assertEqual(secure_clean_filename('test file.pdf'), 'test_file.pdf')
        self.assertEqual(secure_clean_filename('../dangerous.pdf'), 'dangerous.pdf')
        self.assertEqual(secure_clean_filename('file!@#$%^&*.pdf'), 'file.pdf')
    
    def test_sanitize_text(self):
        # Test with control characters
        self.assertEqual(sanitize_text("Hello\x00World"), "HelloWorld")
        self.assertEqual(sanitize_text("Line1\x0BLine2"), "Line1Line2")
        # Test with normal text
        self.assertEqual(sanitize_text("Normal text"), "Normal text")
        # Test with empty or None
        self.assertEqual(sanitize_text(""), "")
        self.assertEqual(sanitize_text(None), "")
    
    def test_fix_common_ocr_errors(self):
        # Punctuation spacing is normalised.
        self.assertEqual(fix_common_ocr_errors("Hello , world ."), "Hello, world.")
        # Words hyphenated across a line break are re-joined.
        self.assertEqual(fix_common_ocr_errors("exam-\nple"), "example")
        # Trailing whitespace is stripped, excessive blank lines collapsed.
        self.assertEqual(fix_common_ocr_errors("Para1   \n\n\n\nPara2"), "Para1\n\nPara2")
        # Line structure is preserved by default (no reflow).
        self.assertEqual(fix_common_ocr_errors("Line1\nLine2"), "Line1\nLine2")
        # Reflow is opt-in.
        self.assertEqual(fix_common_ocr_errors("Line1\nLine2", reflow=True), "Line1 Line2")
        # Test with empty
        self.assertEqual(fix_common_ocr_errors(""), "")
        self.assertEqual(fix_common_ocr_errors(None), None)

    def test_fix_common_ocr_errors_never_corrupts_content(self):
        """Regression guard: the clean-up pass must not rewrite characters.

        A previous implementation applied blind substitutions ('0'->'O',
        '1'->'I', '5'->'S', 'rn'->'m', 'cl'->'d'), silently destroying every
        digit in every converted document. Nothing may reintroduce that.
        """
        invoice = (
            "ACME Corporation\n"
            "Invoice No. 2024-0042\n"
            "IBAN: IT60X0542811101000000123456\n"
            "Total due: 1,250.00 EUR"
        )
        self.assertEqual(fix_common_ocr_errors(invoice), invoice)
        # The specific substitutions that used to corrupt output.
        self.assertEqual(fix_common_ocr_errors("015"), "015")
        self.assertEqual(fix_common_ocr_errors("modern class vvv"), "modern class vvv")
    
    @patch('PIL.ImageEnhance.Contrast')
    @patch('PIL.Image.Image.filter')
    @patch('PIL.Image.Image.convert')
    def test_enhance_image(self, mock_convert, mock_filter, mock_contrast):
        # Setup mocks
        test_image = Image.new('RGB', (100, 100))
        mock_filter.return_value = test_image
        mock_enhancer = MagicMock()
        mock_enhancer.enhance.return_value = test_image
        mock_contrast.return_value = mock_enhancer
        mock_convert.return_value = test_image
        
        # Call the function
        result = enhance_image(test_image)
        
        # Verify mocks were called
        mock_filter.assert_called_once()
        mock_contrast.assert_called_once()
        
        # For non-L mode image, convert should be called
        mock_convert.assert_called_once_with('L')
        
        # Result should be the test image
        self.assertEqual(result, test_image)
    
    @patch('app.logger')
    def test_enhance_image_error_handling(self, mock_logger):
        # Setup mock
        test_image = Image.new('RGB', (100, 100))
        mock_logger.warning = MagicMock()
        
        # Mock an error when enhancing
        with patch('PIL.Image.Image.filter', side_effect=Exception("Test error")):
            # Call the function
            result = enhance_image(test_image)
            
            # Should return original image on error
            self.assertEqual(result, test_image)
            
            # Should log a warning
            mock_logger.warning.assert_called_once()
    
    def test_parse_preprocess_options(self):
        opts = parse_preprocess_options({
            'pre-grayscale': '1', 'pre-sharpen': '1',
            'pre-threshold': '1', 'pre-contrast': '1.8',
        })
        self.assertEqual(opts, {
            "grayscale": True, "sharpen": True, "threshold": True, "contrast": 1.8,
        })

        # Unchecked boxes are absent from the form, not sent as '0'.
        opts = parse_preprocess_options({'pre-contrast': '1.0'})
        self.assertFalse(opts["grayscale"])
        self.assertFalse(opts["threshold"])

        # A hand-crafted request cannot ask for an absurd contrast factor.
        self.assertEqual(parse_preprocess_options({'pre-contrast': '999'})["contrast"], 2.5)
        self.assertEqual(parse_preprocess_options({'pre-contrast': '-5'})["contrast"], 0.5)
        self.assertEqual(parse_preprocess_options({'pre-contrast': 'abc'})["contrast"], 1.5)

    def test_otsu_threshold_separates_two_populations(self):
        # Half the pixels black, half white: the cutoff must land between them.
        image = Image.new('L', (10, 10), color=0)
        for y in range(5):
            for x in range(10):
                image.putpixel((x, y), 220)
        cutoff = otsu_threshold(image)
        # The cutoff is the top of the dark class: `p > cutoff` must send the
        # 0-valued pixels to black and the 220-valued ones to white.
        self.assertGreaterEqual(cutoff, 0)
        self.assertLess(cutoff, 220)

        # A uniform image has no split to find; the fallback must not divide by
        # zero or blank the page.
        self.assertEqual(otsu_threshold(Image.new('L', (4, 4), color=0)), 128)
        self.assertEqual(otsu_threshold(Image.new('L', (0, 0), color=0)), 128)

    def test_enhance_image_threshold_binarises(self):
        image = Image.new('L', (20, 20), color=0)
        for x in range(20):
            for y in range(10):
                image.putpixel((x, y), 200)

        result = enhance_image(image, {
            "grayscale": True, "sharpen": False, "contrast": 1.0, "threshold": True,
        })
        self.assertEqual(set(result.convert('L').tobytes()), {0, 255})

    def test_enhance_image_honours_disabled_options(self):
        image = Image.new('RGB', (10, 10), color='white')
        result = enhance_image(image, {
            "grayscale": False, "sharpen": False, "contrast": 1.0, "threshold": False,
        })
        # Nothing was asked for, so the mode is untouched.
        self.assertEqual(result.mode, 'RGB')

    @patch('app.logger')
    def test_preprocessing_reaches_path_based_engines(self, mock_logger):
        """EasyOCR/PaddleOCR read the file from disk, not the PIL object.

        Without writing the enhanced page back out they silently received the
        untouched render, so the preprocessing checkbox did nothing for them.
        """
        img = Image.new('RGB', (40, 40), color='white')
        img_path = os.path.join(self.test_upload_folder, 'page.png')
        img.save(img_path)

        fake_reader = MagicMock()
        fake_reader.readtext.return_value = ["text"]
        fake_module = MagicMock()
        fake_module.Reader = MagicMock(return_value=fake_reader)

        import sys
        with patch.dict(sys.modules, {'easyocr': fake_module}):
            process_image(0, img_path, "easyocr", "eng", preprocess=True)

        used_path = fake_reader.readtext.call_args[0][0]
        self.assertNotEqual(used_path, img_path)
        self.assertTrue(used_path.endswith('.pre.png'))
        # The temporary preprocessed copy is cleaned up afterwards.
        self.assertFalse(os.path.exists(used_path))

    def test_save_as_markdown(self):
        test_results = {0: "Test page 1", 1: "Test page 2\n\nParagraph 2"}
        test_output = _temp_path('.md')
        
        try:
            save_as_markdown(test_results, test_output)
            
            # Check file exists
            self.assertTrue(os.path.exists(test_output))
            
            # Check file content
            with open(test_output, 'r') as f:
                content = f.read()
                self.assertIn("Test page 1", content)
                self.assertIn("Test page 2", content)
                self.assertIn("Paragraph 2", content)
                self.assertIn("---", content)  # Page separator
        finally:
            # Clean up
            if os.path.exists(test_output):
                os.remove(test_output)
    
    def test_save_as_html(self):
        test_results = {0: "Test page 1", 1: "Test page 2\n\nParagraph 2", 2: "Test with <html> & entities"}
        test_output = _temp_path('.html')
        test_title = "Test Document"
        
        try:
            save_as_html(test_results, test_output, test_title)
            
            # Check file exists
            self.assertTrue(os.path.exists(test_output))
            
            # Check file content
            with open(test_output, 'r') as f:
                content = f.read()
                self.assertIn("<!DOCTYPE html>", content)
                self.assertIn(f"<title>{test_title}</title>", content)
                self.assertIn("<p>Test page 1</p>", content)
                self.assertIn("<p>Test page 2</p>", content)
                self.assertIn("<p>Paragraph 2</p>", content)
                self.assertIn("<hr class=\"page-break\">", content)
                # Check HTML entities are escaped
                self.assertIn("&lt;html&gt; &amp; entities", content)
        finally:
            # Clean up
            if os.path.exists(test_output):
                os.remove(test_output)
    
    @patch('app.subprocess.check_output')
    def test_check_dependency_tesseract(self, mock_check_output):
        # Mock tesseract version output
        mock_check_output.side_effect = [
            "tesseract 4.1.1\n Released version", 
            "List of languages:\neng\nfra\ndeu"
        ]
        
        installed, data = check_dependency('tesseract')
        
        self.assertTrue(installed)
        self.assertEqual(data["installed"], True)
        self.assertIn("tesseract 4.1.1", data["version"])
        self.assertIn("eng", data["languages"])
        self.assertIn("fra", data["languages"])
    
    @patch('app.subprocess.check_output')
    def test_check_dependency_poppler(self, mock_check_output):
        # Mock poppler version output
        mock_check_output.return_value = "pdftoppm version 22.02.0"
        
        installed, data = check_dependency('poppler')
        
        self.assertTrue(installed)
        self.assertEqual(data["installed"], True)
        self.assertIn("pdftoppm version", data["version"])
    
    @patch('app.subprocess.check_output')
    def test_check_dependency_not_installed(self, mock_check_output):
        # Mock subprocess raising FileNotFoundError
        mock_check_output.side_effect = FileNotFoundError("No such file")
        
        installed, data = check_dependency('tesseract')
        
        self.assertFalse(installed)
        self.assertEqual(data["installed"], False)
        self.assertIn("not installed", data["message"])
    
    def test_check_dependency_unknown(self):
        installed, data = check_dependency('unknown')
        
        self.assertFalse(installed)
        self.assertEqual(data["installed"], False)
        self.assertIn("Unknown dependency", data["message"])
    
    def test_index_route(self):
        # Mock check_dependencies to return success
        with patch('app.check_dependencies', return_value=(True, "All good")):
            response = self.app.get('/')
            self.assertEqual(response.status_code, 200)
    
    def test_index_route_with_dependency_error(self):
        # Mock check_dependencies to return failure
        with patch('app.check_dependencies', return_value=(False, "Missing tesseract")):
            with self.app.session_transaction():
                pass  # Setup session if needed
            
            response = self.app.get('/', follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            # Flash message content would be in response data
            self.assertIn(b"Missing tesseract", response.data)
    
    @patch('app.logger')
    def test_process_image_tesseract(self, mock_logger):
        # Create a test image
        img = Image.new('RGB', (100, 100), color='white')
        img_path = os.path.join(self.test_upload_folder, 'test_image.png')
        img.save(img_path)
        
        # Mock pytesseract to return known text
        with patch('pytesseract.image_to_string', return_value="Test OCR result"):
            with patch('pytesseract.get_tesseract_version', return_value="4.1.1"):
                idx, text = process_image(0, img_path, "tesseract", "eng")
                
                self.assertEqual(idx, 0)
                self.assertEqual(text, "Test OCR result")
                mock_logger.info.assert_called()
    
    @patch('app.logger')
    def test_process_image_unsupported_engine(self, mock_logger):
        # Create a test image
        img = Image.new('RGB', (100, 100), color='white')
        img_path = os.path.join(self.test_upload_folder, 'test_image.png')
        img.save(img_path)
        
        idx, text = process_image(0, img_path, "unsupported", "eng")
        
        self.assertEqual(idx, 0)
        self.assertIn("Error: Unsupported OCR engine", text)
    
    @patch('app.logger')
    def test_process_image_file_not_found(self, mock_logger):
        # Non-existent image path
        img_path = os.path.join(self.test_upload_folder, 'nonexistent.png')
        
        idx, text = process_image(0, img_path, "tesseract", "eng")
        
        self.assertEqual(idx, 0)
        self.assertIn("Error: File not found", text)
        mock_logger.error.assert_called()
    
    # --- helpers -------------------------------------------------------

    TASK_ID = "11111111-2222-3333-4444-555555555555"

    def _own(self, *task_ids):
        """Mark the given tasks as started by this test client's session."""
        with self.app.session_transaction() as sess:
            sess['owned_tasks'] = list(task_ids)

    def _make_task(self, task_id=None, **fields):
        """Create a task record owned by this session and return its id."""
        task_id = task_id or self.TASK_ID
        TASKS.set(task_id, fields)
        self._own(task_id)
        return task_id

    # --- task status ---------------------------------------------------

    def test_api_task_status_not_found(self):
        response = self.app.get(f'/api/task_status/{self.TASK_ID}')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "not_found")

    def test_api_task_status_processing(self):
        task_id = self._make_task(status="processing", step="converting", progress=50)

        response = self.app.get(f'/api/task_status/{task_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "processing")
        self.assertEqual(data["progress"], 50)

    def test_api_task_status_completed(self):
        task_id = self._make_task(
            status="completed", progress=100,
            result_path="/path/to/result.docx", output_filename="result.docx",
        )

        response = self.app.get(f'/api/task_status/{task_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["progress"], 100)
        self.assertIn("redirect", data)  # Should have redirect URL
        # The on-disk location of the result must never reach the browser.
        self.assertNotIn("result_path", data)

    def test_task_status_of_another_session_is_not_readable(self):
        """A task id alone must not grant access to someone else's conversion."""
        TASKS.set(self.TASK_ID, {"status": "completed", "output_filename": "secret.docx"})
        # No session ownership recorded for this client.
        self.assertEqual(self.app.get(f'/api/task_status/{self.TASK_ID}').status_code, 404)
        self.assertEqual(
            self.app.get(f'/download/{self.TASK_ID}', follow_redirects=False).status_code, 302
        )

    def test_task_store_rejects_traversal_ids(self):
        """A crafted task id must not escape the task directory."""
        self.assertIsNone(TASKS.get('../../etc/passwd'))
        TASKS.set('../../evil', {"status": "completed"})
        self.assertFalse(os.path.exists(os.path.join(self.test_upload_folder, '..', '..', 'evil.json')))

    def test_task_store_roundtrip_and_update(self):
        TASKS.set(self.TASK_ID, {"status": "processing", "progress": 10})
        TASKS.update(self.TASK_ID, progress=55)
        record = TASKS.get(self.TASK_ID)
        self.assertEqual(record["status"], "processing")
        self.assertEqual(record["progress"], 55)
        self.assertIn("timestamp", record)
        TASKS.delete(self.TASK_ID)
        self.assertIsNone(TASKS.get(self.TASK_ID))

    def test_is_within_upload_folder(self):
        inside = os.path.join(self.test_upload_folder, 'result.docx')
        self.assertTrue(is_within_upload_folder(inside))
        self.assertFalse(is_within_upload_folder('/etc/passwd'))
        # A sibling directory sharing the prefix must not pass (the old check
        # used str.startswith and would have accepted this).
        self.assertFalse(is_within_upload_folder(self.test_upload_folder + '_evil/x'))

    def test_looks_like_pdf(self):
        import io
        self.assertTrue(looks_like_pdf(io.BytesIO(b'%PDF-1.7\nrest')))
        self.assertFalse(looks_like_pdf(io.BytesIO(b'<?php echo 1; ?>')))
        self.assertFalse(looks_like_pdf(io.BytesIO(b'')))
        # The stream must be rewound so the file can still be saved.
        stream = io.BytesIO(b'%PDF-1.7\nrest')
        looks_like_pdf(stream)
        self.assertEqual(stream.tell(), 0)

    def test_healthz(self):
        response = self.app.get('/healthz')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data)["status"], "ok")

    def test_security_headers_are_set(self):
        with patch('app.check_dependencies', return_value=(True, "All good")):
            response = self.app.get('/')
        self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
        self.assertEqual(response.headers['X-Frame-Options'], 'DENY')
        self.assertIn("default-src 'self'", response.headers['Content-Security-Policy'])

    def test_upload_rejects_non_pdf_content(self):
        """A file named .pdf but not containing a PDF must not reach Poppler."""
        import io
        with patch('app.check_dependencies', return_value=(True, "All good")):
            response = self.app.post(
                '/upload',
                data={'file': (io.BytesIO(b'<?php echo 1; ?>'), 'payload.pdf')},
                content_type='multipart/form-data',
                follow_redirects=True,
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'not a PDF', response.data)

    def test_upload_rejects_unknown_output_format(self):
        """`output-format` used to be interpolated straight into the file path."""
        import io
        with patch('app.check_dependencies', return_value=(True, "All good")):
            response = self.app.post(
                '/upload',
                data={
                    'file': (io.BytesIO(b'%PDF-1.7\n'), 'doc.pdf'),
                    'output-format': '../../../etc/cron.d/x',
                },
                content_type='multipart/form-data',
                follow_redirects=True,
            )
        self.assertIn(b'Unknown output format', response.data)

    def test_run_task_in_background_records_success(self):
        """Cover the thread wrapper itself, not just the function it calls.

        Nothing exercised this path, so an AttributeError inside it surfaced
        only as a generic "error starting the OCR process" flash at runtime.
        """
        done = threading.Event()

        def fake_conversion():
            return True, os.path.join(self.test_upload_folder, "r.txt"), "r.txt"

        TASKS.set(self.TASK_ID, {"status": "processing", "progress": 0})
        run_task_in_background(lambda: (done.set(), fake_conversion())[1], self.TASK_ID)
        self.assertTrue(done.wait(timeout=5))

        for _ in range(50):
            record = TASKS.get(self.TASK_ID)
            if record and record.get("status") == "completed":
                break
            time.sleep(0.05)

        record = TASKS.get(self.TASK_ID)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["progress"], 100)
        self.assertEqual(record["output_filename"], "r.txt")

    def test_run_task_in_background_records_failure(self):
        def boom():
            raise RuntimeError("conversion exploded")

        TASKS.set(self.TASK_ID, {"status": "processing", "progress": 0})
        with patch('app.logger'):
            run_task_in_background(boom, self.TASK_ID)
            for _ in range(50):
                record = TASKS.get(self.TASK_ID)
                if record and record.get("status") == "failed":
                    break
                time.sleep(0.05)

        record = TASKS.get(self.TASK_ID)
        self.assertEqual(record["status"], "failed")
        self.assertIn("conversion exploded", record["error"])

    def test_stale_processing_task_is_marked_failed(self):
        """A conversion whose worker died must not stay pending forever.

        The record refreshes after every page, so a "processing" task with an
        old timestamp is abandoned. Nothing used to move it out of that state
        and the progress page polled it indefinitely.
        """
        task_file = os.path.join(self.test_upload_folder, '.tasks', f'{self.TASK_ID}.json')
        os.makedirs(os.path.dirname(task_file), exist_ok=True)
        with open(task_file, 'w') as f:
            json.dump({
                "status": "processing", "progress": 40,
                "timestamp": time.time() - (STALE_TASK_TIMEOUT + 60),
            }, f)
        self._own(self.TASK_ID)

        with patch('app.logger'):
            response = self.app.get(f'/api/task_status/{self.TASK_ID}')

        data = json.loads(response.data)
        self.assertEqual(data["status"], "failed")
        self.assertIn("stopped unexpectedly", data["error"])

    def test_recent_processing_task_is_left_alone(self):
        task_id = self._make_task(status="processing", progress=40)
        response = self.app.get(f'/api/task_status/{task_id}')
        self.assertEqual(json.loads(response.data)["status"], "processing")

    def test_health_and_diagnostic_endpoints_do_not_set_a_cookie(self):
        """An uptime monitor polling /healthz was handed a session cookie."""
        for path in ('/healthz', '/system-check'):
            with patch('app.check_dependency', return_value=(True, {"installed": True})):
                response = self.app.get(path)
            self.assertIsNone(response.headers.get('Set-Cookie'), msg=path)

    def test_save_as_html_escapes_the_title(self):
        out = _temp_path('.html')
        try:
            save_as_html({0: "body"}, out, title='x"><script>alert(1)</script>')
            with open(out, encoding='utf-8') as fh:
                content = fh.read()
            self.assertNotIn("<script>", content)
            self.assertIn("&lt;script&gt;", content)
            self.assertIn("&quot;", content)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_env_int_reports_the_variable_it_could_not_parse(self):
        with patch.dict(os.environ, {'MAX_PAGES': 'abc'}):
            with self.assertRaises(RuntimeError) as ctx:
                env_int('MAX_PAGES', 200)
        self.assertIn("MAX_PAGES", str(ctx.exception))
        self.assertIn("abc", str(ctx.exception))

        with patch.dict(os.environ, {'MAX_PAGES': '0'}):
            with self.assertRaises(RuntimeError):
                env_int('MAX_PAGES', 200)

        # Unset and empty both fall back to the default.
        with patch.dict(os.environ, {'MAX_PAGES': ''}):
            self.assertEqual(env_int('MAX_PAGES', 200), 200)

    def test_log_safe_flattens_forged_log_lines(self):
        self.assertEqual(log_safe("normal.pdf"), "normal.pdf")
        # A newline in a logged value would otherwise let an attacker append
        # something that reads like a genuine log entry.
        forged = "evil.pdf\n2026-01-01 - app - INFO - Admin logged in"
        self.assertNotIn("\n", log_safe(forged))
        self.assertNotIn("\r", log_safe("a\rb"))
        # Long values are truncated rather than flooding the log.
        self.assertLessEqual(len(log_safe("x" * 5000)), 210)

    def test_system_check_does_not_leak_internals(self):
        """It needs no authentication, so it must not fingerprint the host."""
        with patch('app.check_dependency', side_effect=Exception("boom: /usr/local/secret")):
            with patch('app.logger'):
                response = self.app.get('/system-check')
        body = response.data.decode()
        self.assertNotIn("/usr/local/secret", body)
        self.assertNotIn("boom", body)

        data = json.loads(body)
        # Major.minor only — the full sys.version carries patch level, build
        # date and compiler.
        self.assertRegex(data["python_version"], r'^\d+\.\d+$')

    def test_check_dependency_error_does_not_reach_the_caller(self):
        with patch('app.subprocess.check_output', side_effect=Exception("boom: /usr/local/secret")):
            with patch('app.logger'):
                installed, data = check_dependency('tesseract')
        self.assertFalse(installed)
        self.assertNotIn("secret", json.dumps(data))
    def test_cancel_marks_the_task(self):
        task_id = self._make_task(status="processing", progress=40)
        with patch('app.logger'):
            response = self.app.post(f'/cancel/{task_id}')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(json.loads(response.data)["cancelled"])
        self.assertTrue(TASKS.is_cancel_requested(task_id))

    def test_cancel_is_a_noop_on_a_finished_task(self):
        task_id = self._make_task(status="completed", progress=100)
        response = self.app.post(f'/cancel/{task_id}')
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.data)
        self.assertFalse(body["cancelled"])
        self.assertEqual(body["status"], "completed")
        self.assertFalse(TASKS.is_cancel_requested(task_id))

    def test_cancel_requires_ownership(self):
        """A task id alone must not let a stranger stop someone's conversion."""
        TASKS.set(self.TASK_ID, {"status": "processing", "progress": 10})
        response = self.app.post(f'/cancel/{self.TASK_ID}')
        self.assertEqual(response.status_code, 404)
        self.assertFalse(TASKS.is_cancel_requested(self.TASK_ID))

    def test_cancel_flag_survives_a_concurrent_progress_write(self):
        """The flag must not be clobbered by the worker's own record writes.

        `TaskStore.update()` is a read-modify-write with no cross-process
        locking. Had the flag lived in the task record, this sequence would
        lose it: the worker reads the record, the cancel lands, the worker
        writes its stale copy back. The marker file has a single writer.
        """
        task_id = self._make_task(status="processing", progress=10)

        stale = TASKS.get(task_id)          # worker reads...
        TASKS.request_cancel(task_id)       # ...cancel arrives...
        stale["progress"] = 20
        TASKS.set(task_id, stale)           # ...worker writes its stale copy

        self.assertTrue(TASKS.is_cancel_requested(task_id))

    def test_deleting_a_task_removes_its_cancel_marker(self):
        task_id = self._make_task(status="processing")
        TASKS.request_cancel(task_id)
        TASKS.delete(task_id)
        self.assertFalse(TASKS.is_cancel_requested(task_id))
        # A later task reusing the id must not start out cancelled.
        TASKS.set(task_id, {"status": "processing"})
        self.assertFalse(TASKS.is_cancel_requested(task_id))

    def test_cancel_rejects_get(self):
        """Cancellation changes state, so it must not be reachable by GET."""
        task_id = self._make_task(status="processing")
        self.assertEqual(self.app.get(f'/cancel/{task_id}').status_code, 405)

    def test_save_output_rejects_unknown_format(self):
        with self.assertRaises(ValueError):
            save_output({0: "text"}, "exe", os.path.join(self.test_upload_folder, "x"), "x")


    def test_api_check_dependency(self):
        with patch('app.check_dependency', return_value=(True, {"installed": True, "version": "4.1.1"})):
            response = self.app.get('/api/check-dependency?name=tesseract')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data["installed"], True)
            self.assertEqual(data["version"], "4.1.1")
    
    def test_api_check_dependency_no_name(self):
        response = self.app.get('/api/check-dependency')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)
    
    def test_system_check(self):
        # Mock dependency checks (order: tesseract, poppler, paddleocr).
        # PaddleOCR is reported but optional, so its absence must NOT flip status to error.
        with patch('app.check_dependency', side_effect=[
            (True, {"installed": True, "version": "4.1.1"}),
            (True, {"installed": True, "version": "22.02.0"}),
            (False, {"installed": False, "message": "PaddleOCR is not installed"})
        ]):
            response = self.app.get('/system-check')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data["status"], "ok")
            self.assertIn("python_version", data)
            self.assertIn("dependencies", data)
            self.assertIn("tesseract", data["dependencies"])
            self.assertIn("poppler", data["dependencies"])
            self.assertIn("paddleocr", data["dependencies"])
            self.assertFalse(data["dependencies"]["paddleocr"]["installed"])
            self.assertIn("upload_dir", data)

    def test_allowed_file_uppercase(self):
        self.assertTrue(allowed_file('TEST.PDF'))
        self.assertFalse(allowed_file('TEST.DOCX'))

    def test_secure_clean_filename_unicode_and_traversal(self):
        # The secure_clean_filename function removes non-ASCII characters, so expect only ASCII output
        self.assertEqual(secure_clean_filename('üñîçødé.pdf'), 'unicde.pdf')  # Updated expected value
        # The function replaces spaces with underscores, so expect 'etc_passwd.pdf'
        self.assertEqual(secure_clean_filename('../../etc/passwd.pdf'), 'etc_passwd.pdf')  # Updated expected value

    def test_sanitize_text_only_control(self):
        self.assertEqual(sanitize_text("\x00\x01\x02"), "")

    def test_fix_common_ocr_errors_punctuation_only(self):
        text = "l1 rn cl vv , . ; : ! ? 0 1 5"
        # Only the spaces before punctuation are removed; every character the
        # OCR engine produced survives verbatim.
        expected = "l1 rn cl vv,.;:!? 0 1 5"
        self.assertEqual(fix_common_ocr_errors(text), expected)

    @patch('app.logger')
    def test_process_image_easyocr(self, mock_logger):
        """Exercise the real easyocr branch with a fake module (as for paddleocr).

        The previous version of this test patched `app.process_image`, then
        called the mock and asserted the mock had been called — it never
        touched application code at all. Same for the pyocr case below.
        """
        img = Image.new('RGB', (100, 100), color='white')
        img_path = os.path.join(self.test_upload_folder, 'easyocr.png')
        img.save(img_path)

        fake_reader = MagicMock()
        fake_reader.readtext.return_value = ["Hello", "World"]
        fake_module = MagicMock()
        fake_module.Reader = MagicMock(return_value=fake_reader)

        import sys
        with patch.dict(sys.modules, {'easyocr': fake_module}):
            idx, text = process_image(0, img_path, "easyocr", "eng+ita")

        self.assertEqual(idx, 0)
        self.assertEqual(text, "Hello\nWorld")
        # 3-letter Tesseract codes are mapped to EasyOCR's 2-letter codes.
        fake_module.Reader.assert_called_once_with(['en', 'it'])

    @patch('app.logger')
    def test_process_image_pyocr(self, mock_logger):
        img = Image.new('RGB', (100, 100), color='white')
        img_path = os.path.join(self.test_upload_folder, 'pyocr.png')
        img.save(img_path)

        fake_tool = MagicMock()
        fake_tool.image_to_string.return_value = "PyOCR result"
        fake_module = MagicMock()
        fake_module.get_available_tools.return_value = [fake_tool]

        import sys
        with patch.dict(sys.modules, {'pyocr': fake_module, 'pyocr.builders': MagicMock()}):
            idx, text = process_image(0, img_path, "pyocr", "eng")

        self.assertEqual(idx, 0)
        self.assertEqual(text, "PyOCR result")
        fake_tool.image_to_string.assert_called_once()

    @patch('app.logger')
    def test_process_image_pyocr_without_tools(self, mock_logger):
        img = Image.new('RGB', (100, 100), color='white')
        img_path = os.path.join(self.test_upload_folder, 'pyocr_empty.png')
        img.save(img_path)

        fake_module = MagicMock()
        fake_module.get_available_tools.return_value = []

        import sys
        with patch.dict(sys.modules, {'pyocr': fake_module, 'pyocr.builders': MagicMock()}):
            idx, text = process_image(0, img_path, "pyocr", "eng")

        self.assertIn("No OCR tool found", text)

    @patch('app.logger')
    def test_process_image_paddleocr(self, mock_logger):
        # Exercises the REAL paddleocr dispatch branch by injecting a fake
        # `paddleocr` module into sys.modules, so the test never needs the
        # (heavy) paddleocr package installed — keeping CI green.
        img = Image.new('RGB', (100, 100), color='white')
        img_path = os.path.join(self.test_upload_folder, 'paddleocr.png')
        img.save(img_path)

        # Fake PaddleOCR 2.x return: [ [ [box], (text, confidence) ], ... ]
        # (outer list is per-image). Include malformed entries that must be skipped.
        fake_result = [[
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("Hello", 0.99)],
            [[[0, 2], [1, 2], [1, 3], [0, 3]], ("World", 0.98)],
            None,                       # malformed -> TypeError -> skipped
            [],                         # malformed -> IndexError -> skipped
            [[[0, 4]], (123, 0.5)],     # non-str text -> skipped by isinstance guard
        ]]

        fake_reader = MagicMock()
        fake_reader.ocr.return_value = fake_result
        fake_paddle_cls = MagicMock(return_value=fake_reader)
        fake_module = MagicMock()
        fake_module.__version__ = '2.7.0'   # the API this branch targets
        fake_module.PaddleOCR = fake_paddle_cls

        import sys
        with patch.dict(sys.modules, {'paddleocr': fake_module}):
            idx, text = process_image(0, img_path, "paddleocr", "eng")

        self.assertEqual(idx, 0)
        # Only the two well-formed lines survive; line structure is preserved.
        self.assertEqual(text, "Hello\nWorld")
        # Built with the PaddleOCR 2.x API and 'eng' mapped to 'en'
        fake_paddle_cls.assert_called_once_with(use_angle_cls=True, lang="en", show_log=False)
        fake_reader.ocr.assert_called_once_with(img_path, cls=True)

    @patch('app.logger')
    def test_process_image_paddleocr_3x_reports_the_mismatch(self, mock_logger):
        """A 3.x install must say so, not fail with a bare TypeError.

        The dispatch targets the 2.x API (use_angle_cls/cls/show_log, .ocr()).
        CI cannot catch a bad bump here — paddleocr is a lazy import and is not
        installed there — so the runtime message has to be the useful one.
        """
        img = Image.new('RGB', (100, 100), color='white')
        img_path = os.path.join(self.test_upload_folder, 'paddle3.png')
        img.save(img_path)

        fake_module = MagicMock()
        fake_module.__version__ = '3.7.0'

        import sys
        with patch.dict(sys.modules, {'paddleocr': fake_module}):
            idx, text = process_image(0, img_path, "paddleocr", "eng")

        self.assertEqual(idx, 0)
        self.assertIn("3.7.0", text)
        self.assertIn("targets the 2.x API", text)
        # It must not have tried to build a reader with the removed arguments.
        fake_module.PaddleOCR.assert_not_called()

    def test_cleanup_old_files_removes_old(self):
        """Call the app's own cleanup, not a reimplementation of it.

        The previous version defined a local `force_cleanup` helper and
        asserted on that, so `cleanup_old_files` was never executed.
        """
        old_file = os.path.join(self.test_upload_folder, "old.pdf")
        recent_file = os.path.join(self.test_upload_folder, "recent.pdf")
        for path in (old_file, recent_file):
            with open(path, 'w') as f:
                f.write("test content")

        # Make one of them seven days old.
        old_mtime = time.time() - (7 * 24 * 60 * 60)
        os.utime(old_file, (old_mtime, old_mtime))

        # An expired task record, plus the result file it points at. Written
        # directly because TaskStore.set() always stamps the current time.
        expired_result = os.path.join(self.test_upload_folder, "expired.docx")
        with open(expired_result, 'w') as f:
            f.write("docx")
        task_file = os.path.join(self.test_upload_folder, '.tasks', f'{self.TASK_ID}.json')
        os.makedirs(os.path.dirname(task_file), exist_ok=True)
        with open(task_file, 'w') as f:
            json.dump({
                "status": "completed",
                "result_path": expired_result,
                "timestamp": time.time() - (2 * TASK_TIMEOUT),
            }, f)

        with patch('app.LAST_CLEANUP_TIME', 0):
            cleanup_old_files()

        self.assertFalse(os.path.exists(old_file))
        self.assertTrue(os.path.exists(recent_file))
        self.assertIsNone(TASKS.get(self.TASK_ID))
        self.assertFalse(os.path.exists(expired_result))

    def test_save_as_markdown_empty(self):
        test_results = {}
        test_output = _temp_path('.md')
        try:
            save_as_markdown(test_results, test_output)
            self.assertTrue(os.path.exists(test_output))
            with open(test_output, 'r') as f:
                content = f.read()
                self.assertEqual(content, "")
        finally:
            if os.path.exists(test_output):
                os.remove(test_output)

    def test_save_as_html_empty(self):
        test_results = {}
        test_output = _temp_path('.html')
        try:
            save_as_html(test_results, test_output, "EmptyDoc")
            self.assertTrue(os.path.exists(test_output))
            with open(test_output, 'r') as f:
                content = f.read()
                self.assertIn("<title>EmptyDoc</title>", content)
        finally:
            if os.path.exists(test_output):
                os.remove(test_output)

    def test_download_route_unknown_task(self):
        # Should redirect to index with an error when the task is unknown
        response = self.app.get(f'/download/{self.TASK_ID}', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'No conversion data found', response.data)

    def test_download_serves_the_result(self):
        result_path = os.path.join(self.test_upload_folder, "dummy.txt")
        with open(result_path, "w") as f:
            f.write("converted text")
        task_id = self._make_task(
            status="completed", progress=100,
            result_path=result_path, output_filename="dummy.txt",
        )

        response = self.app.get(f'/download/{task_id}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"converted text")
        self.assertIn('attachment', response.headers['Content-Disposition'])
        # send_file keeps the file object open on the response; close it so the
        # suite does not emit a ResourceWarning on every run.
        response.close()

    def test_download_refuses_result_outside_upload_folder(self):
        outside = _temp_path('.txt')
        with open(outside, 'w') as f:
            f.write("secret")
        try:
            task_id = self._make_task(
                status="completed", result_path=outside, output_filename="x.txt",
            )
            response = self.app.get(f'/download/{task_id}', follow_redirects=True)
            self.assertIn(b'Access denied', response.data)
        finally:
            os.remove(outside)

    def test_new_conversion_route_cleans_files(self):
        result_path = os.path.join(self.test_upload_folder, "dummy.docx")
        with open(result_path, "w") as f:
            f.write("docx")
        task_id = self._make_task(
            status="completed", result_path=result_path, output_filename="dummy.docx",
        )

        response = self.app.get(f'/new_conversion/{task_id}', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(os.path.exists(result_path))
        self.assertIsNone(TASKS.get(task_id))

def _poppler_available() -> bool:
    return shutil.which('pdftoppm') is not None


@unittest.skipUnless(_poppler_available(), "Poppler (pdftoppm) is not installed")
class TestConversionPipeline(unittest.TestCase):
    """End-to-end cover for `process_pdf_with_progress`.

    Poppler does the rendering for real; only the OCR call is stubbed, so this
    exercises page counting, batched rendering, progress reporting, output
    assembly and cleanup — none of which had any test coverage.
    """

    def setUp(self):
        app.config['TESTING'] = True
        self.app_context = app.app_context()
        self.app_context.push()
        self.upload_folder = tempfile.mkdtemp()
        self.original_upload_folder = app.config['UPLOAD_FOLDER']
        app.config['UPLOAD_FOLDER'] = self.upload_folder

    def tearDown(self):
        TASKS.clear()
        shutil.rmtree(self.upload_folder, ignore_errors=True)
        app.config['UPLOAD_FOLDER'] = self.original_upload_folder
        self.app_context.pop()

    def _make_pdf(self, pages: int) -> str:
        """Write a real multi-page PDF into the upload folder."""
        images = [Image.new('RGB', (200, 280), color='white') for _ in range(pages)]
        pdf_path = os.path.join(self.upload_folder, 'input.pdf')
        images[0].save(pdf_path, 'PDF', save_all=True, append_images=images[1:])
        return pdf_path

    def test_converts_a_multi_page_pdf_to_txt(self):
        pdf_path = self._make_pdf(3)
        task_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        TASKS.set(task_id, {"status": "processing", "progress": 0})

        with patch('app.process_image', side_effect=lambda i, *a, **k: (i, f"page {i}")):
            success, out_path, out_name = process_pdf_with_progress(
                pdf_path, task_id, output_format="txt", orig_filename="input.pdf"
            )

        self.assertTrue(success, msg=out_name)
        self.assertEqual(out_name, "input.txt")
        with open(out_path, encoding='utf-8') as f:
            content = f.read()
        for i in range(3):
            self.assertIn(f"page {i}", content)
        self.assertEqual(content.count("--- Page Break ---"), 2)

        # The uploaded PDF is removed once the conversion succeeds.
        self.assertFalse(os.path.exists(pdf_path))
        # Progress reached the assembly stage.
        self.assertGreaterEqual(TASKS.get(task_id)["progress"], 95)
        # No page images are left behind in a temp directory.
        self.assertEqual(
            [p for p in os.listdir(self.upload_folder) if p.endswith('.png')], []
        )

    def test_refuses_a_pdf_over_the_page_limit(self):
        pdf_path = self._make_pdf(3)
        task_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        TASKS.set(task_id, {"status": "processing", "progress": 0})

        with patch('app.MAX_PAGES', 2):
            success, out_path, message = process_pdf_with_progress(
                pdf_path, task_id, output_format="txt", orig_filename="input.pdf"
            )

        self.assertFalse(success)
        self.assertIn("limit is 2", message)
        # The upload is removed on the failure path too; it used to be deleted
        # only after a successful conversion, so a rejected document sat in the
        # upload folder until the next daily sweep.
        self.assertFalse(os.path.exists(pdf_path))

    def test_cancellation_stops_the_conversion_partway(self):
        """Cancel a real conversion mid-run and check it stops and leaves nothing.

        The flag is set from inside the OCR callback, i.e. while the worker is
        between pages — the same moment the /cancel route would set it.
        """
        pdf_path = self._make_pdf(6)
        task_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        TASKS.set(task_id, {"status": "processing", "progress": 0})

        pages_seen = []

        def ocr_then_cancel(i, *args, **kwargs):
            pages_seen.append(i)
            if i == 1:
                # Ask for cancellation the way the route does.
                TASKS.request_cancel(task_id)
            return (i, f"page {i}")

        with patch('app.process_image', side_effect=ocr_then_cancel):
            with self.assertRaises(ConversionCancelled):
                process_pdf_with_progress(
                    pdf_path, task_id, output_format="txt", orig_filename="input.pdf"
                )

        # It stopped early rather than running to the end.
        self.assertLess(len(pages_seen), 6)
        self.assertIn(1, pages_seen)

        # Nothing is left behind: no output file, and the upload is gone.
        self.assertFalse(os.path.exists(pdf_path))
        leftovers = [f for f in os.listdir(self.upload_folder) if not f.startswith('.')]
        self.assertEqual(leftovers, [])

    def test_cancelled_conversion_is_recorded_as_cancelled_not_failed(self):
        pdf_path = self._make_pdf(4)
        task_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        TASKS.set(task_id, {"status": "processing", "progress": 0})

        def ocr_then_cancel(i, *args, **kwargs):
            TASKS.request_cancel(task_id)
            return (i, "text")

        with patch('app.process_image', side_effect=ocr_then_cancel), patch('app.logger'):
            run_task_in_background(
                process_pdf_with_progress, task_id,
                pdf_path, task_id, "tesseract", "eng", "standard", False, "input.pdf", "txt",
            )
            for _ in range(100):
                record = TASKS.get(task_id)
                if record and record.get("status") != "processing":
                    break
                time.sleep(0.05)

        record = TASKS.get(task_id)
        self.assertEqual(record["status"], "cancelled")
        # A cancellation is not an error, so it must carry no error message.
        self.assertNotIn("error", record)

    def test_renders_in_batches_rather_than_all_at_once(self):
        """Peak memory depends on this: one Poppler call per RENDER_BATCH_SIZE."""
        pdf_path = self._make_pdf(5)
        task_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        TASKS.set(task_id, {"status": "processing", "progress": 0})

        import pdf2image
        real_convert = pdf2image.convert_from_path
        calls = []

        def counting_convert(*args, **kwargs):
            calls.append((kwargs.get('first_page'), kwargs.get('last_page')))
            return real_convert(*args, **kwargs)

        with patch('app.process_image', side_effect=lambda i, *a, **k: (i, f"page {i}")), \
                patch('pdf2image.convert_from_path', side_effect=counting_convert), \
                patch('app.RENDER_BATCH_SIZE', 2):
            success, _, message = process_pdf_with_progress(
                pdf_path, task_id, output_format="md", orig_filename="input.pdf"
            )

        self.assertTrue(success, msg=message)
        self.assertEqual(calls, [(1, 2), (3, 4), (5, 5)])


if __name__ == '__main__':
    unittest.main(testRunner=ColorTextTestRunner(verbosity=2))