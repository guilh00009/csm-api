#!/bin/bash

# Update package lists
apt-get update

# Install system dependencies
apt-get install -y ffmpeg

# Install Python dependencies
pip install -r requirements.txt

echo "All dependencies installed successfully!" 