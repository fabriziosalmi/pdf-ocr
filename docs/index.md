---
layout: home
title: Scanned PDFs to editable text
titleTemplate: pdf-ocr

hero:
  name: pdf-ocr
  text: Scanned PDFs to editable text
  tagline: A self-hosted Flask app that runs OCR over a PDF and gives you back DOCX, TXT, Markdown or HTML. Runs on your own machine; nothing is sent anywhere.
  actions:
    - theme: brand
      text: Quickstart
      link: /guide/quickstart
    - theme: alt
      text: Read the threat model first
      link: /security
    - theme: alt
      text: GitHub
      link: https://github.com/fabriziosalmi/pdf-ocr

features:
  - title: Four output formats
    details: DOCX, plain text, Markdown and HTML. Page breaks are preserved as page breaks in each format.
  - title: Four OCR engines
    details: Tesseract by default. EasyOCR, PyOCR and PaddleOCR work once you install their optional dependencies.
  - title: Bounded by design
    details: Pages render in small batches, so peak memory does not grow with document length. Upload size and page count are both capped.
  - title: It tells you the truth
    details: The clean-up pass never rewrites a character the engine produced, and every control in the form is read by the server. Both of those were once false, and there are tests that keep them true.
---

## What this is, in one paragraph

You upload a PDF in a browser. Poppler renders each page to an image, an OCR engine reads
the image, and the recognised text is written to the format you picked. It is a **single-user,
self-hosted tool** — there is no authentication and no rate limiting, deliberately, so it
belongs on a private network or behind an authenticating proxy. The
[threat model](/security) says exactly what it does and does not protect you from.

## Try it in three commands

```bash
git clone https://github.com/fabriziosalmi/pdf-ocr.git
cd pdf-ocr
cp .env.example .env && docker compose up --build
```

Compose refuses to start until `SECRET_KEY` is set, so
[the quickstart](/guide/quickstart) shows how to generate one first. Then open
<http://localhost:8011>.

## Or run the published image

```bash
docker run -d -p 8011:8011 \
  -e SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  -v "$PWD/uploads:/app/uploads" \
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  ghcr.io/fabriziosalmi/pdf-ocr:v0.5.1
```

Those hardening flags are not decoration — see [Deployment](/guide/deployment) for what each
one is doing and why a bare `docker run` gives you less than you might assume.
