# Deployment

Tagging `vX.Y.Z` publishes a multi-arch image (`linux/amd64` and `linux/arm64`) to
`ghcr.io/fabriziosalmi/pdf-ocr`.

## Docker run

```bash
docker run -d -p 8011:8011 \
  -e SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  -v "$PWD/uploads:/app/uploads" \
  --read-only --tmpfs /tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  ghcr.io/fabriziosalmi/pdf-ocr:v0.5.1
```

::: warning The hardening flags are not decoration
The image runs as an unprivileged user (uid 10001) on its own. **The read-only root
filesystem and the dropped capabilities come from the runtime**, not from the image — a bare
`docker run` gives you neither. The app only ever writes to `/app/uploads` and `/tmp`, so
these cost nothing.
:::

Prefer a version tag over `:latest` for anything you care about. `:latest` follows the
default branch, and v0.4.0 was a breaking release.

## Docker Compose

`docker-compose.yml` sets the same options, and refuses to start without `SECRET_KEY`:

```bash
cp .env.example .env
# set SECRET_KEY in .env
docker compose up -d
```

## Behind a reverse proxy

This is the intended deployment. The app has no authentication and no rate limiting, so the
proxy is where both belong. Terminate TLS in front of it and set:

```
SESSION_COOKIE_SECURE=true
```

which marks the session cookie `Secure` and enables HSTS. Leave it `false` when serving over
plain HTTP, or the browser will discard the cookie and no conversion will be visible to the
session that started it.

## Sizing

Memory is dominated by page rendering, not by OCR. `RENDER_BATCH_SIZE` (default 4) is the
number of pages Poppler renders in one call, and is what bounds peak usage — the whole
document is never in memory at once. At 600 DPI a page is a large bitmap, so lower the batch
size before lowering anything else if you are tight on RAM.

`WEB_CONCURRENCY` (default 2) sets gunicorn worker processes. A conversion occupies one
worker thread for its whole duration, so concurrent conversions need concurrent workers.

## Health checks

`/healthz` returns 200 when the upload folder is writable, and 503 when it is not. The image
declares a `HEALTHCHECK` against it. It does not create a session or set a cookie, so an
uptime monitor can poll it freely.

```bash
curl -fsS localhost:8011/healthz
```

`/system-check` gives a fuller diagnostic as JSON — dependency versions, upload directory
state. It deliberately does **not** include exception text or the full Python version; those
go to the log instead, because the endpoint needs no authentication.

## Storage and cleanup

Everything lives under `UPLOAD_FOLDER` (default `uploads`), including a `.tasks` directory of
one JSON file per conversion. That is what lets more than one gunicorn worker serve the same
conversion, and what lets progress survive a restart.

Cleanup runs at most once an hour: files older than 24 hours are deleted, and task records —
with the file each points at — are dropped an hour after their last update. A cancelled
conversion leaves nothing behind at all.

If you mount the volume, mount it at `/app/uploads`.
