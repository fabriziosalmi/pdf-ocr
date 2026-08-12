#!/bin/sh
set -eu

UPLOAD_FOLDER="${UPLOAD_FOLDER:-/app/uploads}"
mkdir -p "$UPLOAD_FOLDER"

# The Flask development server is single-threaded, has no request limits and
# ships a debugger that is remote code execution if FLASK_ENV leaks into a
# deployment. Serve with gunicorn instead.
#
# Conversions run in a background thread inside the worker, so requests stay
# short; the timeout only needs to cover an upload of MAX_UPLOAD_MB.
exec gunicorn \
    --bind "0.0.0.0:${PORT:-8011}" \
    --workers "${WEB_CONCURRENCY:-2}" \
    --threads "${WEB_THREADS:-4}" \
    --timeout "${WEB_TIMEOUT:-120}" \
    --graceful-timeout 30 \
    --access-logfile - \
    --error-logfile - \
    app:app
