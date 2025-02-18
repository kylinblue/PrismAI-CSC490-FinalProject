FROM python:3.11-slim

WORKDIR /app

# Install curl for healthchecks and basic build tools
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DOCKER_CONTAINER=1

CMD ["python", "src/core.py"]
