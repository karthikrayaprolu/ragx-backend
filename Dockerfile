# Use specific Python version
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
# build-essential is often needed for compiling Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# Using CPU-only PyTorch to reduce image size significantly
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . .

# Cloud Run injects the PORT environment variable
# We use shell form to allow variable expansion
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
