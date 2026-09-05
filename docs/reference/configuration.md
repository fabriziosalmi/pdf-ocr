# Configuration

Every setting is an environment variable. Copy
[`.env.example`](https://github.com/fabriziosalmi/pdf-ocr/blob/main/.env.example) and adjust.

::: tip This table is checked against the code
`scripts/check_docs_match_code.py` fails CI if the code reads a variable this table does not
list, or if the table lists one the code no longer reads. It exists because this
documentation drifted badly once before.
:::

## Required

| Variable | Effect | Default |
|---|---|---|
| `SECRET_KEY` | Signs the session cookie. **The app refuses to start without it.** Generate with `python -c 'import secrets; print(secrets.token_hex(32))'`. | — |

A key generated per process would differ in every gunicorn worker, so a conversion started
on one worker would be invisible to the next request, and every restart would invalidate
every session. Two escape hatches exist for local work only: `FLASK_ENV=development` and
`PDF_OCR_TESTING=1`. Never use either in a deployment.

## Serving

| Variable | Effect | Default |
|---|---|---|
| `PORT` | Port to listen on. | `8011` |
| `WEB_CONCURRENCY` | gunicorn worker processes. A conversion occupies one for its duration. | `2` |
| `WEB_THREADS` | gunicorn threads per worker. | `4` |
| `WEB_TIMEOUT` | gunicorn request timeout in seconds. Must cover a full upload, not a full conversion — conversions run in the background. | `120` |
| `SESSION_COOKIE_SECURE` | `true` when serving over HTTPS: marks the cookie `Secure` and enables HSTS. Setting it while serving plain HTTP makes the browser discard the cookie. | `false` |

The three `WEB_*` variables are read by `entrypoint.sh`, not by the application, so they
apply to the container and to any gunicorn invocation that uses that script.

## Limits

| Variable | Effect | Default |
|---|---|---|
| `MAX_UPLOAD_MB` | Upload size limit. Larger requests get a 413. | `64` |
| `MAX_PAGES` | Maximum pages per PDF. A longer document is rejected before rendering. | `200` |
| `RENDER_BATCH_SIZE` | Pages rendered per Poppler call. **This is what bounds peak memory** — the whole document is never in memory at once. | `4` |
| `STALE_TASK_TIMEOUT` | Seconds without progress before a conversion is reported as failed rather than left pending. | `1800` |

All four are positive integers; a malformed value fails at startup with a message naming the
variable rather than a bare `ValueError`.

## Storage and logging

| Variable | Effect | Default |
|---|---|---|
| `UPLOAD_FOLDER` | Where uploads, results and task records live. Mount your volume here. | `uploads` |
| `LOG_LEVEL` | Python logging level. | `INFO` |
| `LOG_FILE` | If set, also log to this file, rotating at 10 MB with 3 backups. Unset means stdout only. | unset |

Logging to stdout is the default because that is what an orchestrator collects, and because a
read-only or non-writable working directory should not stop the app from booting.

## Environment

| Variable | Effect | Default |
|---|---|---|
| `DOCKER_ENV` | `true` skips the local Tesseract and Poppler probes. Set inside the image, where both are known to be present. | `false` |
| `FLASK_ENV` | `development` runs without `SECRET_KEY` and enables the Werkzeug debugger. | unset |

::: danger FLASK_ENV=development is remote code execution
The Werkzeug debugger executes arbitrary Python from the browser. It exists for local
development only.
:::
