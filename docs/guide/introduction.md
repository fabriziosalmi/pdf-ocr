# Introduction

`pdf-ocr` takes a scanned or image-only PDF and gives you back text you can edit.

The flow is deliberately simple, and worth knowing because almost every problem people hit
maps onto one of these three steps:

1. **Poppler** renders each page of the PDF to a PNG, at 300 or 600 DPI.
2. An **OCR engine** reads that image and returns text.
3. The text is written to **DOCX, TXT, Markdown or HTML**.

If the output is empty, step 2 found nothing legible — usually a low-quality scan. If the
conversion fails before it starts, step 1 could not read the PDF.

## What it is for

One person, running it themselves, converting their own documents. That is the whole design
target. It is a good fit for a laptop, a home server, or a container on a private network.

## What it is not

It is **not** a hosted service, and it is not hardened to be one. There is no authentication
and no rate limiting, both deliberately absent rather than half-built — see
[Security](/security) for the full threat model. Anyone who can reach the port can convert
documents and consume the CPU.

Poppler and Tesseract are large C and C++ codebases that parse untrusted input. Putting this
on the public internet exposes that parsing surface to everyone.

## What works today

- Upload → convert → download, with a live progress page.
- Four output formats: DOCX, TXT, Markdown, HTML.
- Four OCR engines: Tesseract (default), EasyOCR, PyOCR, PaddleOCR.
- Language selection, including `+`-joined combinations for Tesseract.
- Opt-in preprocessing: grayscale, sharpen, a contrast slider, Otsu binarisation.
- Standard (300 DPI) or high (600 DPI) rendering.
- Cancelling a running conversion, which stops at the next page boundary and leaves nothing
  behind.
- Conversions scoped to the browser session that started them.
- A Docker image with Tesseract and Poppler already inside.

## What it does not do

Two lists, because "not got to it" and "decided against" are different promises.

**Decided against**

- **Advanced preprocessing** — denoise, deskew, border removal, preset profiles. These need
  OpenCV, roughly 60 MB on an image that already carries Tesseract with a dozen language
  packs, for three checkboxes. Deskew is the only one that would meaningfully help crooked
  scans, and alone it does not justify the weight. Preprocess such scans before uploading.
- **Authentication and rate limiting** — absent on purpose. Use a proxy.

**Not built yet**

- DOCX layout preservation; output is plain paragraphs.
- Heading and structure detection.
- Parallel page processing within one conversion.
- Batch or folder processing.

## A note on trust

Two things in this application were once untrue, and both are worth knowing about because
the fixes are what the tests now defend:

- The OCR clean-up pass applied blind substitutions — `0`→`O`, `1`→`I`, `5`→`S` — to text the
  engine had already read correctly, corrupting every number in every document. It now only
  adjusts whitespace and punctuation, and never rewrites a character.
- The options panel offered ten controls the server never read. They were removed rather than
  stubbed.

If you find something the documentation claims but the code does not do, that is a bug worth
reporting.
