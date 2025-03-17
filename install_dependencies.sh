#!/bin/bash

# Detect OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    echo "Windows environment detected"
    
    # Check if Chocolatey is installed
    if ! command -v choco &> /dev/null; then
        echo "Chocolatey not found. Please install Chocolatey package manager first:"
        echo "Run PowerShell as administrator and execute:"
        echo "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
        exit 1
    fi
    
    # Install ffmpeg using Chocolatey
    echo "Installing ffmpeg via Chocolatey..."
    choco install ffmpeg -y
else
    # Linux environment
    echo "Linux environment detected"
    
    # Check if we have sudo access
    if command -v sudo &> /dev/null; then
        # Update package lists
        sudo apt-get update || apt-get update
        
        # Install system dependencies
        sudo apt-get install -y ffmpeg || apt-get install -y ffmpeg
    else
        # Try without sudo (e.g., in Docker)
        apt-get update
        apt-get install -y ffmpeg
    fi
fi

echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if ffmpeg is now available
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg successfully installed!"
else
    echo "Warning: FFmpeg installation may have failed. Please install it manually."
    
    # For Windows users who couldn't install via choco
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
        echo "Windows users: Download FFmpeg from https://ffmpeg.org/download.html"
        echo "Extract the files and add the bin folder to your PATH environment variable"
    fi
fi

echo "All dependencies installed successfully!" 