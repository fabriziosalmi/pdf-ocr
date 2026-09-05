# HTTP endpoints

::: tip Checked against the code
`scripts/check_docs_match_code.py` fails CI if the app registers a route this page does not
document.
:::

## Conversion flow

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Upload form. |
| `POST` | `/upload` | Accepts the PDF, starts a conversion, redirects to its status page. |
| `GET` | `/status/<task_id>` | Progress page. Polls the JSON endpoint below. |
| `GET` | `/api/task_status/<task_id>` | Progress as JSON. |
| `POST` | `/cancel/<task_id>` | Asks a running conversion to stop at the next page boundary. |
| `GET` | `/success/<task_id>` | Result page for a finished conversion. |
| `GET` | `/download/<task_id>` | Downloads the converted file. |
| `GET` | `/new_conversion/<task_id>` | Discards a result and its record, then returns to the form. |
| `GET` | `/new_conversion` | Returns to the form without discarding anything. |

## Diagnostics

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | 200 if the upload folder is writable, 503 if not. Used by the image `HEALTHCHECK`. |
| `GET` | `/system-check` | Dependency diagnostics as JSON. |
| `GET` | `/api/check-dependency` | Checks one dependency, by `?name=tesseract`, `poppler` or `paddleocr`. |

The three diagnostic endpoints never create a session or set a cookie, so an uptime monitor
can poll them without collecting one on every probe.

## Access control

**A conversion is readable only by the browser session that started it.** The status, cancel,
success and download routes return 404 (or redirect) for a task the session does not own, so
a task id on its own grants nothing. Task ids are server-generated UUIDs.

There is **no authentication** on any of this. Anyone who can reach the port can start a
conversion. See [Security](/security).

## Upload parameters

`POST /upload` takes a multipart form:

| Field | Values | Notes |
|---|---|---|
| `file` | a PDF | Must have a `.pdf` extension **and** the `%PDF-` magic bytes. |
| `ocr-engine` | `tesseract`, `easyocr`, `pyocr`, `paddleocr` | Allowlisted. |
| `output-format` | `docx`, `txt`, `md`, `html` | Allowlisted. |
| `language` | e.g. `eng`, `ita`, `eng+fra` | Letters and underscores, `+`-joined. |
| `ocr-quality` | `standard` (300 DPI) or `high` (600 DPI) | Anything else is treated as standard. |
| `preprocess` | `1` to enable | The master switch for the four options below. |
| `pre-grayscale`, `pre-sharpen`, `pre-threshold` | `1` to enable | Only read when `preprocess=1`. |
| `pre-contrast` | `0.5` to `2.5` | Clamped server-side. |

Every field the form submits is read by the server. That was once not true — ten controls
were submitted and discarded — and is now a rule.

## Status JSON

```json
{
  "status": "processing",
  "step": "ocr",
  "progress": 45,
  "timestamp": 1786606731.9
}
```

`status` is one of `processing`, `completed`, `failed` or `cancelled`. When it reaches a
terminal state the response also carries a `redirect`. A `failed` record carries an `error`;
a `cancelled` one deliberately does not, because the user asking to stop is not an error.

The result's path on disk is never included.
