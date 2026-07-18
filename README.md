# pdf-ocr

[![CI](https://github.com/fabriziosalmi/pdf-ocr/actions/workflows/ci.yml/badge.svg)](https://github.com/fabriziosalmi/pdf-ocr/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small **Flask web app** that turns scanned/image PDFs into editable text formats
(**DOCX, TXT, Markdown, HTML**) using OCR. You upload a PDF in the browser, it renders
each page to an image with Poppler, runs OCR (Tesseract by default), and gives you the
extracted text back as a downloadable file.

This is a working single-file application (`app.py`, ~920 lines) with a unit-test suite
— not a stub. It is a **useful local/self-hosted tool**, not a hardened multi-tenant
service. See [What works today vs. Roadmap](#what-works-today-vs-roadmap) for an honest
feature-by-feature breakdown before you rely on any specific capability.

## Screenshots

![screenshot1](screenshot_1.png)
![screenshot2](screenshot_2.png)
![screenshot3](screenshot_3.png)
![screenshot4](screenshot_4.png)

## Quickstart (Docker, 3 commands)

Docker is the fastest path because the image already contains Tesseract and Poppler,
so you don't have to install anything else:

```bash
git clone https://github.com/fabriziosalmi/pdf-ocr.git
cd pdf-ocr
docker compose up --build
```

Then open <http://localhost:8011>, drop a PDF onto the page, pick an output format,
and download the result.

### Run locally without Docker

You need the two system binaries first (Python packages alone are not enough):

```bash
# macOS
brew install tesseract poppler
# Debian/Ubuntu
# sudo apt-get install -y tesseract-ocr poppler-utils

pip install -r requirements.txt
python app.py                     # serves on http://127.0.0.1:8011
```

To confirm the OCR toolchain is actually wired up on your machine:

```bash
python ocr_test.py                # renders a test image, OCRs it, prints PASS/FAIL
```

## Demo (input → output)

The conversion flow is: **PDF → Poppler renders each page to PNG → OCR engine reads the
image → text is written to your chosen format.** OCR quality depends entirely on the
scan quality of your input.

For example, two OCR'd pages of an invoice come back as Markdown like this (this is the
literal output of the app's Markdown writer, with `---` marking a page break):

```markdown
ACME Corporation

Invoice No. 2024-0042
Date: 2024-11-03

Total due: 1,250.00 EUR

---

Terms: Net 30 days.
Thank you for your business.
```

The same pages exported as HTML wrap each paragraph in `<p>` tags, escape HTML entities,
and insert `<hr class="page-break">` between pages. DOCX output writes one paragraph per
text block with a page break between pages (plain text — see the roadmap note on layout).

## What works today vs. Roadmap

The application genuinely does the core job. Some capabilities described in older versions
of this README were aspirational; they are listed under Roadmap below so expectations match
the code.

### Works today (verified in the code)

- **Web upload → convert → download** flow with a background worker and a live progress
  page (`/status/<id>` polling `/api/task_status/<id>`).
- **Four output formats:** DOCX, TXT, Markdown, HTML.
- **Three OCR engines:** Tesseract (default, always available), plus **EasyOCR** and
  **PyOCR** if you install their optional dependencies.
- **Tesseract language selection** (e.g. `eng`, `ita`, `fra`, `deu`, `chi_sim`, `+`-joined
  for multiple), with sensible 3-letter → EasyOCR code mapping.
- **Basic image preprocessing** (opt-in): sharpen filter + contrast boost + grayscale.
- **Quality toggle:** standard (300 DPI) or high (600 DPI) rendering.
- **Light OCR clean-up:** control-character stripping and a small substitution pass for
  common misreads, plus paragraph splitting on blank lines.
- **Self-diagnostics:** dependency checks for Tesseract/Poppler and a `/system-check` JSON
  endpoint.
- **Docker image** bundling Tesseract (with several language packs) and Poppler.
- **Automatic cleanup** of old uploads and finished tasks.
- **34 unit tests** (`test_app.py`) covering the pure logic and Flask routes.

### Roadmap / not implemented yet

These are referenced in the UI or were previously advertised, but are **not** in the code
today. Contributions welcome.

- **Advanced preprocessing** — denoising, deskewing, thresholding, border removal, and the
  named preset profiles (text-heavy / scanned / low-quality / handwriting). Only
  sharpen+contrast+grayscale exist right now.
- **DOCX layout/formatting preservation** — output is currently plain paragraphs, not a
  faithful reproduction of the source layout.
- **Heading / structure detection** in the output.
- **Real task cancellation** — the "Cancel" link on the status page only navigates home;
  the background OCR job keeps running to completion.
- **Parallel page processing** — OCR currently runs one page at a time (single worker) on
  purpose, to avoid multiprocessing races. Throughput work is pending.
- **Batch / folder processing** — there is no batch mode; each conversion is one uploaded
  PDF via the web UI.
- **PaddleOCR engine** — `requirements-paddleocr.txt` ships, but PaddleOCR is not wired
  into the engine selector yet.
- **Env-configurable `UPLOAD_FOLDER`, `MAX_CONTENT_LENGTH`, `HOST`, and cleanup intervals**
  — these are currently constants in `app.py` (see [Configuration](#configuration) for
  what is actually read from the environment).

## Installation

### Prerequisites

- **Python 3.9+** and `pip`.
- **Tesseract OCR** — `brew install tesseract tesseract-lang` (macOS),
  `apt-get install tesseract-ocr` + language packs (Debian/Ubuntu), or the
  [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) on Windows
  (ensure it is on your `PATH`).
- **Poppler** — `brew install poppler` (macOS), `apt-get install poppler-utils`
  (Debian/Ubuntu), or the
  [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) binaries
  on Windows (add `bin/` to `PATH`).

Docker users can skip both binaries — they are baked into the image.

### Optional OCR engines

Tesseract works out of the box. The other two engines are optional:

```bash
# EasyOCR (pulls in PyTorch — see https://pytorch.org for the right build)
pip install -r requirements-easyocr.txt

# PyOCR (a thin wrapper over the Tesseract/Cuneiform binaries)
pip install pyocr
```

### Helper script

`python install_dependencies.py` installs the core Python packages and checks whether
Tesseract and Poppler are reachable on your `PATH`. Use `--engine easyocr|all` to pull
in optional engines.

## Configuration

Only the following environment variables are actually read by `app.py` today:

| Variable      | Effect                                                        | Default            |
|---------------|---------------------------------------------------------------|--------------------|
| `SECRET_KEY`  | Flask session secret. A random one is generated if unset.     | random per start   |
| `PORT`        | Port to listen on.                                            | `8011`             |
| `FLASK_ENV`   | `development` enables Flask debug mode.                       | production         |
| `DOCKER_ENV`  | `true` skips local dependency checks (set in the image).      | `false`            |

Other settings (upload folder, 64 MB max upload size, `0.0.0.0` bind host, cleanup timings)
are constants in `app.py`. Making them env-configurable is on the roadmap.

## Running the tests

The unit tests mock the Tesseract/Poppler binaries, so they run anywhere with just the
Python dependencies — no OCR engines required:

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m unittest test_app -v          # 34 tests
ruff check .                            # lint (same gate as CI)
```

CI runs both of these on every push and pull request — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Troubleshooting

**Tesseract not found** — confirm `tesseract --version` works in your shell and that the
install dir is on `PATH`. On Windows you may need to set it explicitly in `app.py`:
`pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`.

**Poppler / PDF conversion errors** — confirm `pdftoppm -v` works and Poppler's `bin/` is
on `PATH`; restart your terminal after changing `PATH`.

**Empty or poor OCR output** — try High quality (600 DPI), enable preprocessing, or switch
engines. OCR is only as good as the source scan.

**First EasyOCR run is slow** — it downloads language models on first use.

## Contributing

Issues and pull requests are welcome. Please run `ruff check .` and
`python -m unittest test_app` before opening a PR.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built on [Tesseract OCR](https://github.com/tesseract-ocr/tesseract),
[EasyOCR](https://github.com/JaidedAI/EasyOCR),
[PyOCR](https://gitlab.gnome.org/World/OpenPaperwork/pyocr),
[Flask](https://flask.palletsprojects.com/),
[pdf2image](https://github.com/Belval/pdf2image),
[python-docx](https://python-docx.readthedocs.io/),
[Pillow](https://python-pillow.org/), and
[Tailwind CSS](https://tailwindcss.com/).
