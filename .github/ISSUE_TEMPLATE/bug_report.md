---
name: Bug report
about: Something behaves differently from what the documentation says
title: ''
labels: bug
assignees: ''
---

**What happened, and what you expected instead**


**How you are running it**

- Version or image tag (e.g. `v0.5.1`, `ghcr.io/fabriziosalmi/pdf-ocr:latest`):
- Docker, Docker Compose, or a local Python install:
- If local: OS and `python --version`:

**Which conversion**

- OCR engine (tesseract / easyocr / pyocr / paddleocr):
- Output format (docx / txt / md / html):
- Language code(s):
- Preprocessing on or off:
- Roughly how many pages, and is the PDF scanned or digital:

**Diagnostics**

Please paste the output of these — they usually identify the problem on their own:

```
curl -s localhost:8011/system-check
python ocr_test.py
```

**Server log**

The relevant lines from the container or terminal output. Redact filenames if they
are sensitive.

```

```

**Anything else**

<!-- If the output was wrong rather than missing, a small sample of what you got
     versus what the page contained is worth more than a description of it. -->
