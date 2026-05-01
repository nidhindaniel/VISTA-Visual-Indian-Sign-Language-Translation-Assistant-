FROM python:3.9-slim

# Install system dependencies (ffmpeg is required for moviepy)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security (Hugging Face recommendation)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download Spacy English model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY --chown=user . .

# Expose the default port for Hugging Face Spaces
EXPOSE 7860

# Command to run the application
CMD ["python", "sign_to_text/web/app.py"]
