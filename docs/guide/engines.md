# OCR engines

Tesseract is the default and the only one whose Python dependency is installed by default.
The other three are optional and lazily imported, so they cost nothing until selected.

| Engine | Install | Notes |
|---|---|---|
| **Tesseract** | included | Needs the `tesseract` binary. Fast, good on clean scans. |
| **EasyOCR** | `pip install -r requirements-easyocr.txt` | Pulls in PyTorch. First run downloads language models, so it is slow once. |
| **PyOCR** | `pip install pyocr` | A thin wrapper over the `tesseract` binary; needs it too. |
| **PaddleOCR** | `pip install -r requirements-paddleocr.txt` | Heavy. **2.x only** — see below. |

Each optional engine's requirements file layers on top of the pinned core, so installing one
cannot quietly downgrade Flask or Pillow.

## Languages

The form offers `eng`, `fra`, `deu`, `spa`, `ita`, `por`, `chi_sim`, `chi_tra`, `jpn`, `kor`,
`rus`, `ara`, `hin`, and one `+`-joined combination. The server accepts any `+`-joined set of
Tesseract codes.

The engines do not agree on language codes, so the app maps them:

- **Tesseract and PyOCR** take the codes above as-is. The matching language pack must be
  installed.
- **EasyOCR** gets three-letter codes mapped to its two-letter ones (`ita` → `it`).
- **PaddleOCR** loads one language per reader, so with a `+`-joined set it uses **the first
  one**.

## The PaddleOCR version pin

The app is written against the **PaddleOCR 2.x** API: it builds the reader with
`use_angle_cls` and `show_log`, calls `.ocr(path, cls=True)`, and walks the nested result
shape. PaddleOCR 3.x removed all of that and renamed the method to `.predict()`.

So `requirements-paddleocr.txt` pins `paddleocr>=2.6,<3.0`, and Dependabot is configured to
ignore major updates for it.

::: danger Why this is pinned rather than left to Dependabot
`paddleocr` is a lazy import and is **not installed in CI**. A bump to 3.x would leave every
check green while the engine was broken for anyone who selected it. Moving to 3.x is a code
change plus the bump, made together.
:::

If a 3.x install is present at runtime, the app says so by name rather than failing with a
`TypeError` about an unexpected keyword argument.

## Preprocessing

Opt-in, and applied before the image reaches the engine. Each control is read by the server:

| Control | What it does |
|---|---|
| Grayscale | Converts to single-channel. |
| Sharpen | A sharpening filter, helps soft scans. |
| Contrast | Slider from 0.5 to 2.5, clamped server-side. |
| Threshold | Binarises using an Otsu cutoff computed from the histogram. |

Thresholding helps on clean document scans and hurts on photographs.

Preprocessing reaches every engine, including the two that read the page from disk rather
than taking the image object — they used to silently receive the untouched render.
