# Python 3.9 reached end of life in October 2025 and stopped receiving security
# fixes; 3.12 is the version the CI test matrix covers.
FROM python:3.14-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    tesseract-ocr-ita \
    tesseract-ocr-por \
    tesseract-ocr-jpn \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-kor \
    tesseract-ocr-rus \
    tesseract-ocr-ara \
    tesseract-ocr-hin \
    libgl1 \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install Python dependencies first, so the layer is cached across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (see .dockerignore: .git, screenshots and tests stay out)
COPY . .

# Run as an unprivileged user. The uploads directory is owned by that user
# rather than made world-writable with chmod 777.
RUN useradd --system --create-home --uid 10001 appuser \
    && mkdir -p /app/uploads \
    && chown -R appuser:appuser /app \
    && chmod +x /app/entrypoint.sh
USER appuser

# Set environment variables
ENV DOCKER_ENV=true \
    PORT=8011 \
    UPLOAD_FOLDER=/app/uploads \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8011

# Fail the container if the app stops answering or loses its upload volume
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1

# Run application
ENTRYPOINT ["/app/entrypoint.sh"]
