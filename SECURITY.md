# Security Policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub Security Advisories](https://github.com/fabriziosalmi/pdf-ocr/security/advisories/new),
or by email to fabrizio.salmi@gmail.com. Do not open a public issue for a
vulnerability.

Expect an acknowledgement within a few days. This is a spare-time project, so
there is no formal SLA and no bug bounty.

## Supported versions

Only the latest commit on `main` (and the image built from it) is supported.

## Threat model — read this before exposing the app

`pdf-ocr` is designed as a **single-tenant, self-hosted tool**. It accepts an
uploaded PDF and hands it to Poppler and an OCR engine, both of which are large
C/C++ codebases that parse untrusted input. If you put it on the public internet
you are exposing that parsing surface to anyone.

What the app does provide:

- `SECRET_KEY` is required outside development, so session cookies are signed
  with a stable key.
- Uploads must have a `.pdf` extension **and** the `%PDF-` magic bytes.
- Upload size (`MAX_UPLOAD_MB`, default 64 MB) and page count (`MAX_PAGES`,
  default 200) are capped; pages are rendered in small batches so peak memory
  does not scale with document length.
- Engine, output format and language are validated against allowlists.
- A conversion is only readable by the browser session that started it, and the
  result's path on disk is never sent to the client.
- Baseline security headers, including a same-origin CSP; no third-party assets
  are loaded (Tailwind is vendored under `static/vendor/`).
- The image runs as an unprivileged user (uid 10001). The read-only root
  filesystem, dropped capabilities and `no-new-privileges` are **runtime**
  options, not properties of the image: `docker-compose.yml` sets them, and the
  `docker run` command in the README passes the equivalent flags. A plain
  `docker run` without them still runs unprivileged, but with a writable root
  filesystem and the default capability set.
- Uploads, results and finished tasks are deleted automatically. A cancelled
  conversion leaves nothing behind at all.
- A conversion whose worker dies is reported as failed after
  `STALE_TASK_TIMEOUT` rather than pending forever.
- Static analysis (CodeQL, `security-extended`) runs on every push and pull
  request, plus weekly, because its query pack updates independently of this
  code.

What it does **not** provide, and what you must add yourself before exposing it:

- **No authentication.** Anyone who can reach the port can convert documents.
- **No rate limiting.** OCR is CPU- and memory-intensive; a handful of
  concurrent uploads will saturate a small host. Put it behind a reverse proxy
  that enforces limits, or behind an authenticating proxy.
- **No sandboxing of Poppler or Tesseract** beyond the container boundary.
- **No encryption at rest** for uploaded or converted files.
- **No CSRF tokens.** The session cookie is `SameSite=Lax`, which keeps
  mainstream browsers from attaching it to a cross-site POST, but that is a
  mitigation rather than a guarantee. State-changing routes rely on the
  session-ownership check, not on a token.
- **No audit log.** Conversions are logged, but nothing is retained about who
  did what — there is no identity to retain.

The intended deployment is on a private network, or behind an authenticating
reverse proxy, with TLS terminated in front of it (set
`SESSION_COOKIE_SECURE=true` in that case).

## Never set these in a deployment

- **`FLASK_ENV=development`** turns on the Werkzeug debugger, which allows
  arbitrary code execution over HTTP. It exists for local development only.
- **`PDF_OCR_TESTING=1`** makes the app start without `SECRET_KEY`, using a key
  generated per process. The test suite sets it so importing the app does not
  require configuration; in a deployment it means every worker signs cookies
  with a different key and every restart invalidates all sessions.

Neither is read from `.env.example`, and neither is set by the image.
