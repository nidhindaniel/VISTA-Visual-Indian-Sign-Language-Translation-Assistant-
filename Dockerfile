FROM python:3.10-slim

# Install system dependencies for OpenCV, MediaPipe, and Video Processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
# (Hugging Face Spaces run as a non-root user by default)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create the working directory
WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the required spaCy model for text-to-gloss
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application files
COPY --chown=user . .

# Expose port 7860 (Hugging Face Docker Space default port)
EXPOSE 7860

# --- CHOOSE WHICH APP TO RUN ---
# If you want to host the main VISTA app (with Sign Recognition + Text-to-Sign), use this:
CMD ["python", "sign to text/IndianSignDetection-master/web/app.py", "--port", "7860"]

# If you only want to host the simple Text-to-Sign application, uncomment below and comment the line above:
# CMD ["python", "app.py"]
