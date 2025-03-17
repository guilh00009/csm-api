#!/bin/bash

# Set environment variables for better error diagnosis
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1  # Improved CUDA debugging

# Set up error handling
set -e  # Exit immediately if a command exits with a non-zero status
trap 'echo "Error occurred at line $LINENO. Command: $BASH_COMMAND"' ERR

# Install dependencies
echo "Installing dependencies..."
bash install_dependencies.sh || {
    echo "Failed to install dependencies. You may need to install ffmpeg manually."
    echo "Windows: Download from https://ffmpeg.org/download.html"
    echo "Linux: sudo apt-get install ffmpeg"
    echo "macOS: brew install ffmpeg"
    echo "Continuing with the script..."
}

# Make sure Python environment variables are set correctly
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Check if fix_and_run.py exists and run it if available
if [ -f fix_and_run.py ]; then
    echo "Running fix_and_run.py to apply RoPE patches and start training..."
    python fix_and_run.py
    exit $?
fi

# If fix_and_run.py is not available, run fix_rope.py if it exists
if [ -f fix_rope.py ]; then
    echo "Applying RoPE fixes from fix_rope.py..."
    python fix_rope.py
fi

# Check if patches.py exists
if [ ! -f patches.py ]; then
    echo "Warning: patches.py not found. This file is required to fix library compatibility issues."
    echo "Continuing without patches..."
fi

# Download and preprocess the dataset
echo "Downloading and preprocessing the Switchboard dataset..."
python preprocess_switchboard.py --download

# Print some debug information
echo "Python version:"
python --version
echo "Torch version:"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}')"
echo "Available CUDA devices:"
python -c "import torch; print(f'Device count: {torch.cuda.device_count()}, Current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "N/A"})')"
echo "Memory info:"
python -c "import torch; print(f'Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB') if torch.cuda.is_available() else print('CUDA not available')"

# Run the training with memory-efficient and safe settings
echo "Starting training with memory-efficient settings..."
python train.py --config config.json --checkpoint_activations --cpu_offload --num_workers 0

# If you want to resume training from a checkpoint, uncomment the line below
# python train.py --config config.json --checkpoint_activations --cpu_offload --num_workers 0 --resume checkpoints/checkpoint_latest.pt 