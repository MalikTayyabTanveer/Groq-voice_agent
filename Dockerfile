# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to avoid prompts during installations
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install Python 3.10, ffmpeg, and required tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    gcc \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3-dev \
    python3-pip \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create a working directory called VoiceAgent
WORKDIR /VoiceAgent

# Copy the required files into the container
COPY QuickAgent.py audio.py audiotext.py .env system_prompt.txt requirements.txt ./

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Command to run audiotext.py
CMD ["python3", "audiotext.py"]
