# Backend Dockerfile - FastAPI
# Minimal image for running the FastAPI app in back-end/app

FROM python:3.11-slim

# Avoid python writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (if required) and clean apt lists
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the app source code
COPY ./app /app/app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

# Default command
CMD ["/app/start.sh"]
