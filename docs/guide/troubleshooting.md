# Troubleshooting

Start here:

```bash
curl -s localhost:8011/system-check | python -m json.tool   # if it is running
python ocr_test.py                                          # if it is not
```

## "SECRET_KEY environment variable is required"

The app will not start without it. Generate one:

```bash
python -c 'import secrets; print(secrets.token_hex(32))'
```

For local development only, `FLASK_ENV=development` starts without it.

## Tesseract not found

Confirm `tesseract --version` works in the same shell that starts the app, and that the
install directory is on `PATH`.

There is no setting for the binary's location. On Windows the fallback is editing `app.py`
to add, after the imports:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

Putting Tesseract on `PATH` is the better fix.

## PDF conversion errors

Confirm `pdftoppm -v` works and Poppler's `bin/` is on `PATH`. Restart the terminal after
changing `PATH` — an already-open shell keeps the old one.

## Empty or poor output

In rough order of what helps:

1. Switch to **high** quality (600 DPI). Small text at 300 DPI is often the whole problem.
2. Enable preprocessing. **Threshold** helps clean document scans and hurts photographs.
3. Check the **language** matches the document and that its Tesseract pack is installed.
4. Try another engine — EasyOCR often does better on noisy or handwritten-ish input.

OCR cannot recover text that is not legible in the image. If you cannot read it zoomed in,
neither can the engine.

## The progress page sits at the same percentage

Progress updates after each page, so a large page at 600 DPI can look stuck for a while.

If a conversion genuinely stops — the container restarted, or a worker was recycled — it is
reported as failed after `STALE_TASK_TIMEOUT` (default 30 minutes) rather than staying
pending forever. Raise it if you OCR very large pages on slow hardware.

## "The requested conversion was not found"

Conversions are scoped to the browser session that started them. This appears when the
session cookie is missing or different:

- A different browser, or a private window.
- `SESSION_COOKIE_SECURE=true` while serving over plain HTTP — the browser discards the
  cookie.
- More than one gunicorn worker with **no** `SECRET_KEY` pinned, so each worker signs with a
  different key. In a deployment this cannot happen: the app refuses to start.

## A large PDF is rejected

`MAX_PAGES` defaults to 200 and `MAX_UPLOAD_MB` to 64. Raise either. If it runs out of memory
instead, lower `RENDER_BATCH_SIZE` or use standard quality.

## The first EasyOCR run is very slow

It downloads language models on first use. Subsequent runs are normal.

## Cancelling does not stop it immediately

By design. Cancellation takes effect at the next page boundary, because that is the only
place a conversion can stop cleanly and leave nothing behind. A 600 DPI page can take a while
to finish.
