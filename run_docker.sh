#!/bin/bash

# Build and run the Docker container
docker-compose up --build -d

echo "OCR PDF to DOCX app is running at http://localhost:8011"
echo "To stop the application, run: docker-compose down"
