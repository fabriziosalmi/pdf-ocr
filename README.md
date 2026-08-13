# pdf-ocr

[![CI](https://github.com/fabriziosalmi/pdf-ocr/actions/workflows/ci.yml/badge.svg)](https://github.com/fabriziosalmi/pdf-ocr/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small **Flask web app** that turns scanned/image PDFs into editable text formats
(**DOCX, TXT, Markdown, HTML**) using OCR. You upload a PDF in the browser, it renders
each page to an image with Poppler, runs OCR (Tesseract by default), and gives you the
extracted text back as a downloadable file.

It is a single-module Flask application (`app.py`) with a real test suite — not a stub —
and a **local or self-hosted tool**, not a hardened multi-tenant service. See
[What works today](#what-works-today) for a feature-by-feature breakdown, and
[SECURITY.md](SECURITY.md) before putting it anywhere other people can reach.

## Screenshots

![screenshot1](screenshot_1.png)
![screenshot2](screenshot_2.png)
![screenshot3](screenshot_3.png)
![screenshot4](screenshot_4.png)

## Quickstart (Docker)

Docker is the fastest path because the image already contains Tesseract and Poppler,
so you don't have to install anything else:

```bash
git clone https://github.com/fabriziosalmi/pdf-ocr.git
cd pdf-ocr
cp .env.example .env
sed -i.bak "s|^SECRET_KEY=.*|SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')|" .env && rm .env.bak
docker compose up --build
```

`SECRET_KEY` signs the session cookie. The app refuses to start without it unless
`FLASK_ENV=development` is set — a key generated per process would differ in every
gunicorn worker, silently invalidating sessions.

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

# Development server (Werkzeug, single-threaded — local use only)
FLASK_ENV=development python app.py             # http://127.0.0.1:8011

# Or the way the container serves it
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))') \
  gunicorn --bind 127.0.0.1:8011 --workers 2 --threads 4 app:app
```

To confirm the OCR toolchain is actually wired up on your machine:

```bash
python ocr_test.py    # checks Tesseract and Poppler, exits non-zero if either fails
```

It renders a small image, runs OCR on it, converts a generated PDF through Poppler, and
prints a verdict per check. It writes `test_pdf_conversion.png` and `test_pdf.pdf` into the
working directory; both are gitignored.

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
text block with a page break between pages — plain text, not a reproduction of the source
layout (see [What it does not do](#what-it-does-not-do)).

## What works today

- **Web upload → convert → download** flow with a background worker and a live progress
  page (`/status/<id>` polling `/api/task_status/<id>`).
- **Four output formats:** DOCX, TXT, Markdown, HTML.
- **Four OCR engines:** Tesseract (the default, and the only one whose Python dependency is
  installed by default), plus **EasyOCR**, **PyOCR** and **PaddleOCR** once you install their
  optional dependencies. All four still need their engine present — Tesseract and PyOCR need
  the `tesseract` binary.
- **Language selection** — the form offers `eng`, `fra`, `deu`, `spa`, `ita`, `por`,
  `chi_sim`, `chi_tra`, `jpn`, `kor`, `rus`, `ara`, `hin`, and one `+`-joined combination.
  Any `+`-joined set of Tesseract codes is accepted by the server. EasyOCR and PaddleOCR get
  these 3-letter codes mapped to their own; PaddleOCR loads one language per reader, so it
  uses the first of a `+`-joined set.
- **Image preprocessing** (opt-in): grayscale, sharpen, a contrast slider, and Otsu
  binarisation — each wired to its own control in the form.
- **Quality toggle:** standard (300 DPI) or high (600 DPI) rendering.
- **Conservative OCR clean-up:** control-character stripping, re-joining words hyphenated
  across a line break, punctuation spacing, and blank-line collapsing. It never rewrites a
  character the engine recognised.
- **Self-diagnostics:** dependency checks for Tesseract/Poppler (plus an optional PaddleOCR
  lazy-import probe), a `/system-check` JSON endpoint, and a `/healthz` probe used by the
  container HEALTHCHECK.
- **Bounded resource use:** pages are rendered in small batches rather than all at once, with
  caps on upload size (`MAX_UPLOAD_MB`) and page count (`MAX_PAGES`).
- **Multi-worker safe:** task state lives on the filesystem, so gunicorn can run more than
  one worker.
- **Docker image** bundling Tesseract (with several language packs) and Poppler. It runs as
  an unprivileged user; the read-only root filesystem and dropped capabilities come from how
  you run it — `docker-compose.yml` sets them, and the [Deployment](#deployment) command
  below passes the equivalent flags.
- **Cancellation:** a running conversion can be stopped from the progress page. It stops at
  the next page boundary and leaves nothing behind — no output file, no uploaded PDF. Simply
  navigating away does *not* cancel it; the conversion continues in the background.
- **Automatic cleanup**, swept at most once an hour: files in the upload folder older than
  24 hours are deleted, and task records — with the file each points at — are dropped an hour
  after their last update.
- **79 tests** (`test_app.py`). Most mock the OCR call; `TestConversionPipeline` renders real
  PDFs through Poppler, and `TestRealOCR` runs text through actual Tesseract and requires the
  words and digits back.

## What it does not do

Two separate lists, because "we have not got to it" and "we decided against it" are
different promises.

### Decided against

- **Advanced preprocessing.** Denoising, deskewing, border removal and the named preset
  profiles were once checkboxes the server never read. They were removed from the UI rather
  than stubbed, and they are not coming back: doing them properly needs OpenCV, roughly 60 MB
  on an image that already carries Tesseract with a dozen language packs, for three
  checkboxes. Deskew is the only one that would meaningfully help crooked scans, and alone it
  does not justify the weight. Preprocess such scans before uploading.
- **Authentication and rate limiting.** Deliberately absent rather than half-built. Run the
  app on a private network or behind an authenticating proxy — see [SECURITY.md](SECURITY.md).

### Not built yet

- **DOCX layout preservation** — output is plain paragraphs, not a reproduction of the source
  layout.
- **Heading and structure detection** in the output.
- **Parallel page processing** — one page at a time within a conversion. Concurrent
  conversions are handled by separate gunicorn workers.
- **Batch or folder processing** — one uploaded PDF per conversion, via the web UI.

## Endpoints

| Method | Path                        | Purpose                                                        |
|--------|-----------------------------|----------------------------------------------------------------|
| GET    | `/`                         | Upload form.                                                    |
| POST   | `/upload`                   | Accepts the PDF, starts a conversion, redirects to its status.   |
| GET    | `/status/<task_id>`         | Progress page.                                                  |
| GET    | `/api/task_status/<task_id>`| Progress as JSON; the page polls this. 404 if unknown or not yours. |
| POST   | `/cancel/<task_id>`         | Stops a running conversion at the next page boundary.            |
| GET    | `/success/<task_id>`        | Result page for a finished conversion.                          |
| GET    | `/download/<task_id>`       | Downloads the converted file.                                   |
| GET    | `/new_conversion/<task_id>` | Discards a result and returns to the form.                      |
| GET    | `/healthz`                  | Liveness probe; used by the image HEALTHCHECK.                  |
| GET    | `/system-check`             | Dependency diagnostics as JSON.                                 |
| GET    | `/api/check-dependency`     | Checks one dependency by `?name=`.                              |

Conversions are **scoped to the browser session that started them**: a task id on its own
gets a 404 from the status, cancel, success and download routes.

## Installation

### Prerequisites

- **Python 3.11+** and `pip`. CI runs the tests on 3.11, 3.12, 3.13 and 3.14; the Docker
  image uses 3.14. 3.9 and 3.10 are out: the pinned dependencies do not resolve on them.
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

Tesseract works out of the box. The other three engines are optional:

```bash
# EasyOCR (pulls in PyTorch — see https://pytorch.org for the right build)
pip install -r requirements-easyocr.txt

# PyOCR (a thin wrapper over the Tesseract/Cuneiform binaries)
pip install pyocr

# PaddleOCR (heavy — pulls in paddlepaddle; the 2.x line is required)
pip install -r requirements-paddleocr.txt
```

PaddleOCR is wired against the **PaddleOCR 2.x** API (`paddleocr>=2.6,<3.0` with
`paddlepaddle>=2.5,<3.0`). The 3.x release removed `use_angle_cls`/`cls`/`show_log` and
renamed `.ocr()` to `.predict()`, so it is intentionally pinned below 3.0.

### Helper script

`python install_dependencies.py` installs the pinned core requirements and reports whether
Tesseract and Poppler are reachable on your `PATH`. `--engine` accepts `tesseract`,
`easyocr`, `pyocr`, `paddleocr` or `all`. It cannot install the Tesseract or Poppler
binaries themselves — those are system packages.

## Configuration

Copy [`.env.example`](.env.example) and adjust. These are all read by `app.py` or
`entrypoint.sh`:

| Variable                | Effect                                                             | Default    |
|-------------------------|--------------------------------------------------------------------|------------|
| `SECRET_KEY`            | **Required.** Signs the session cookie; the app will not start without it (unless `FLASK_ENV=development`). | —          |
| `PORT`                  | Port to listen on.                                                 | `8011`     |
| `UPLOAD_FOLDER`         | Where uploads, results and task records are stored.                | `uploads`  |
| `MAX_UPLOAD_MB`         | Upload size limit; larger requests get a 413.                      | `64`       |
| `MAX_PAGES`             | Maximum pages per PDF.                                             | `200`      |
| `RENDER_BATCH_SIZE`     | Pages rendered per Poppler call — this is what bounds peak memory. | `4`        |
| `STALE_TASK_TIMEOUT`    | Seconds without progress before a conversion is reported as failed.| `1800`     |
| `SESSION_COOKIE_SECURE` | `true` when serving over HTTPS: marks the cookie Secure, adds HSTS.| `false`    |
| `LOG_LEVEL`             | Python logging level.                                              | `INFO`     |
| `LOG_FILE`              | If set, also log to this file (rotating, 10 MB x 3).               | stdout only|
| `WEB_CONCURRENCY`       | gunicorn worker processes (`entrypoint.sh`).                       | `2`        |
| `WEB_THREADS`           | gunicorn threads per worker (`entrypoint.sh`).                     | `4`        |
| `WEB_TIMEOUT`           | gunicorn request timeout, seconds (`entrypoint.sh`).               | `120`      |
| `FLASK_ENV`             | `development` enables the Werkzeug debugger. **Never set this in a deployment** — it is remote code execution. | production |
| `DOCKER_ENV`            | `true` skips local dependency checks (set in the image).           | `false`    |

## Deployment

Tagging `vX.Y.Z` publishes a multi-arch image to
`ghcr.io/fabriziosalmi/pdf-ocr` (see [`.github/workflows/release.yml`](.github/workflows/release.yml)):

```bash
docker run -d -p 8011:8011 \
  -e SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  -v "$PWD/uploads:/app/uploads" \
  --read-only --tmpfs /tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  ghcr.io/fabriziosalmi/pdf-ocr:latest
```

The hardening flags are not decoration: the image runs as an unprivileged user on its own,
but the read-only root filesystem and dropped capabilities come from the runtime, and
[SECURITY.md](SECURITY.md) assumes you pass them. `docker-compose.yml` sets the same options.
The app only ever writes to `/app/uploads` and `/tmp`.

Prefer a version tag over `:latest` for anything you care about — `:latest` follows the
default branch, and v0.4.0 was a breaking release.

Read [SECURITY.md](SECURITY.md) before exposing it: there is no authentication and no rate
limiting, and Poppler and Tesseract parse untrusted input.

## Running the tests

Most tests mock the OCR engine and run anywhere. Two groups need real binaries and skip
cleanly without them: `TestConversionPipeline` needs **Poppler**, and `TestRealOCR` needs
**Tesseract** — the latter renders text to an image, OCRs it for real, and requires the words
and digits back. CI installs both, so those groups always run there.

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m unittest test_app -v          # 79 tests
ruff check .                            # lint (same gate as CI)
```

CI ([`ci.yml`](.github/workflows/ci.yml)) runs four things on every push and pull request:
ruff, the tests on Python 3.11 to 3.14 with Poppler and Tesseract installed, and a Docker job
that builds the image, waits for its HEALTHCHECK to report healthy and asserts the process is
not running as root. [`codeql.yml`](.github/workflows/codeql.yml) adds static analysis on push,
pull request and weekly. The first three are required for a pull request to merge.

## Troubleshooting

**Tesseract not found** — confirm `tesseract --version` works in your shell and that the
install directory is on `PATH`. The app has no setting for the binary's location, so on
Windows the fallback is editing `app.py` to add
`pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`
after the imports. Putting Tesseract on `PATH` is the better fix.

**Poppler / PDF conversion errors** — confirm `pdftoppm -v` works and Poppler's `bin/` is
on `PATH`; restart your terminal after changing `PATH`.

**Empty or poor OCR output** — try High quality (600 DPI), enable preprocessing (thresholding
helps on clean scans and hurts on photographs), or switch engines. OCR is only as good as the
source scan.

**"SECRET_KEY environment variable is required" on startup** — set one (see
[Configuration](#configuration)), or `FLASK_ENV=development` for local work.

**A large PDF fails with a page limit error** — raise `MAX_PAGES`. If it runs out of memory
instead, lower `RENDER_BATCH_SIZE` or use standard rather than high quality.

**First EasyOCR run is slow** — it downloads language models on first use.

## Contributing

Issues and pull requests are welcome. Please run `ruff check .` and
`python -m unittest test_app` before opening a PR — CI gates on both, plus a Docker build.
By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built on [Tesseract OCR](https://github.com/tesseract-ocr/tesseract),
[EasyOCR](https://github.com/JaidedAI/EasyOCR),
[PyOCR](https://gitlab.gnome.org/World/OpenPaperwork/pyocr),
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),
[Flask](https://flask.palletsprojects.com/),
[pdf2image](https://github.com/Belval/pdf2image),
[python-docx](https://python-docx.readthedocs.io/),
[Pillow](https://python-pillow.org/), and
[Tailwind CSS](https://tailwindcss.com/).
