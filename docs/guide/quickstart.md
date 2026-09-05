# Quickstart

Docker is the fastest path, because the image already contains Tesseract and Poppler.

## With Docker Compose

```bash
git clone https://github.com/fabriziosalmi/pdf-ocr.git
cd pdf-ocr
cp .env.example .env
sed -i.bak "s|^SECRET_KEY=.*|SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')|" .env && rm .env.bak
docker compose up --build
```

Then open <http://localhost:8011>, drop a PDF on the page, pick an output format, and
download the result.

::: tip Why the SECRET_KEY step is not optional
It signs the session cookie, and conversions are scoped to the session that started them. The
app **refuses to start** without it. A key generated per process would differ in every
gunicorn worker, so a conversion started on one worker would be invisible to the next
request — which is exactly the bug this rule exists to prevent.
:::

## Without Docker

You need two system binaries first. Python packages alone are not enough:

::: code-group
```bash [macOS]
brew install tesseract poppler
```
```bash [Debian/Ubuntu]
sudo apt-get install -y tesseract-ocr poppler-utils
```
:::

Then:

```bash
pip install -r requirements.txt

# Development server (Werkzeug, single-threaded, local use only)
FLASK_ENV=development python app.py            # http://127.0.0.1:8011

# Or the way the container serves it
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))') \
  gunicorn --bind 127.0.0.1:8011 --workers 2 --threads 4 app:app
```

::: warning Never set FLASK_ENV=development in a deployment
It enables the Werkzeug debugger, which is arbitrary code execution over HTTP.
:::

## Check your toolchain

```bash
python ocr_test.py
```

It renders a small image, OCRs it, converts a generated PDF through Poppler, and prints a
verdict per check. It exits non-zero if either binary is missing, so it is safe to use in a
script.

The running app answers the same question over HTTP:

```bash
curl -s localhost:8011/system-check | python -m json.tool
```

## What the output looks like

Two OCR'd pages of an invoice, exported as Markdown, with `---` marking the page break —
this is the literal output of the Markdown writer:

```markdown
ACME Corporation

Invoice No. 2024-0042
Date: 2024-11-03

Total due: 1,250.00 EUR

---

Terms: Net 30 days.
Thank you for your business.
```

HTML wraps each paragraph in `<p>`, escapes entities, and separates pages with
`<hr class="page-break">`. DOCX writes one paragraph per block with a page break between
pages — plain text, not a reproduction of the source layout.

OCR quality depends entirely on the scan. Nothing here can recover text that is not legible
in the image.
