# Security

The authoritative copy is
[SECURITY.md](https://github.com/fabriziosalmi/pdf-ocr/blob/main/SECURITY.md) in the
repository. This page says the same things.

## Reporting a vulnerability

Report privately through
[GitHub Security Advisories](https://github.com/fabriziosalmi/pdf-ocr/security/advisories/new),
or by email to `fabrizio.salmi@gmail.com`. **Please do not open a public issue.**

Expect an acknowledgement within a few days. This is a spare-time project: no formal SLA, no
bug bounty. Only the latest commit on `main`, and the image built from it, is supported.

## Threat model

`pdf-ocr` is a **single-tenant, self-hosted tool**. It accepts an uploaded PDF and hands it to
Poppler and an OCR engine — large C and C++ codebases that parse untrusted input. Putting it
on the public internet exposes that parsing surface to anyone who finds it.

### What it does provide

- `SECRET_KEY` is required, so session cookies are signed with a stable key.
- Uploads need a `.pdf` extension **and** the `%PDF-` magic bytes.
- Upload size and page count are capped; pages render in small batches, so peak memory does
  not scale with document length.
- Engine, output format and language are validated against allowlists.
- A conversion is readable only by the session that started it, and the result's path on disk
  is never sent to the client.
- Baseline security headers including a same-origin CSP. No third-party assets are loaded —
  Tailwind is vendored.
- The image runs as an unprivileged user (uid 10001).
- Uploads, results and finished tasks are deleted automatically; a cancelled conversion
  leaves nothing behind at all.
- A conversion whose worker dies is reported as failed rather than left pending.
- CodeQL (`security-extended`) runs on every push and pull request, and weekly.

### What it does not provide

- **No authentication.** Anyone who can reach the port can convert documents.
- **No rate limiting.** OCR is CPU- and memory-intensive; a handful of concurrent uploads
  will saturate a small host.
- **No sandboxing** of Poppler or Tesseract beyond the container boundary.
- **No encryption at rest** for uploaded or converted files.
- **No CSRF tokens.** The session cookie is `SameSite=Lax`, which keeps mainstream browsers
  from attaching it to a cross-site POST, but that is a mitigation rather than a guarantee.
  State-changing routes rely on the session-ownership check.
- **No audit log.** There is no identity to record.

::: warning The container hardening is a runtime choice
The image runs unprivileged on its own. The read-only root filesystem, dropped capabilities
and `no-new-privileges` come from **how you run it** — `docker-compose.yml` sets them and the
[deployment command](/guide/deployment) passes the equivalent flags. A bare `docker run`
gives you neither.
:::

## Intended deployment

A private network, or behind an authenticating reverse proxy, with TLS terminated in front
and `SESSION_COOKIE_SECURE=true`.

## Never set these in a deployment

- **`FLASK_ENV=development`** enables the Werkzeug debugger: arbitrary code execution over
  HTTP.
- **`PDF_OCR_TESTING=1`** starts the app without `SECRET_KEY`, using a per-process key. The
  test suite sets it so importing the app needs no configuration.

Neither is set by the image, and neither is enabled in `.env.example`.
