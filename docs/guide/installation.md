# Installation

## Requirements

- **Python 3.11+**. CI runs the tests on 3.11, 3.12, 3.13 and 3.14; the Docker image uses
  3.14. Python 3.9 and 3.10 are out — the pinned dependencies do not resolve on them.
- **Tesseract OCR** — the binary, not just the Python wrapper.
- **Poppler** — provides `pdftoppm`, which does the PDF rendering.

Docker users can skip both binaries; they are in the image.

## Installing the system binaries

::: code-group
```bash [macOS]
brew install tesseract tesseract-lang poppler
```
```bash [Debian/Ubuntu]
sudo apt-get install -y tesseract-ocr poppler-utils
# plus language packs you need, e.g.
sudo apt-get install -y tesseract-ocr-ita tesseract-ocr-fra
```
```powershell [Windows]
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# Poppler:   https://github.com/oschwartz10612/poppler-windows/releases
# Add both to PATH, then restart your terminal.
```
:::

Verify before going further — if either of these fails, the app will not work:

```bash
tesseract --version
pdftoppm -v
```

## Installing the Python side

```bash
pip install -r requirements.txt
```

Versions are pinned. That is deliberate: floating bounds meant no two installs produced the
same application, and a regression upstream was indistinguishable from one here. Dependabot
proposes updates, and CI has to pass before they land.

### The helper script

```bash
python install_dependencies.py
python install_dependencies.py --engine all
```

It installs the pinned requirements and reports whether Tesseract and Poppler are reachable.
`--engine` accepts `tesseract`, `easyocr`, `pyocr`, `paddleocr` or `all`. It **cannot** install
the Tesseract or Poppler binaries — those are system packages.

## Running the tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
python -m unittest test_app -v
```

::: warning A green run can prove less than it looks
Two groups need real binaries and **skip silently** without them: `TestConversionPipeline`
needs Poppler, and `TestRealOCR` needs Tesseract. Install both, or you are running a smaller
suite than the count suggests. CI installs both.
:::
