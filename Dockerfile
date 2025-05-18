# Use minimal Alpine-based image to reduce CVE surface area
FROM python:3.9-slim-bullseye

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache build-base

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir ollama

# Copy project files
COPY . ./

# Default entrypoint: run full pipeline
ENTRYPOINT ["make", "all"]
