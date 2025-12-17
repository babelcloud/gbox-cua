FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY . /app/

# Install the package
RUN pip install --no-cache-dir -e .

# Install GBox SDK (if not available on PyPI, install from GitHub)
RUN pip install --no-cache-dir git+https://github.com/babelcloud/gbox-sdk-py.git || \
    pip install --no-cache-dir gbox-sdk || \
    echo "Warning: gbox-sdk installation failed. Please install manually."

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["gbox-cua-agent", "--help"]

