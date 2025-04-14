#!/bin/bash

# Create uploads directory if it doesn't exist
mkdir -p /app/uploads
chmod 777 /app/uploads

# Start the Flask application
exec python /app/app.py
