#!/bin/bash

# Download and preprocess the dataset
echo "Downloading and preprocessing the Switchboard dataset..."
python preprocess_switchboard.py --download

# Run the training with memory-efficient settings
echo "Starting training with memory-efficient settings..."
python train.py --config config.json --checkpoint_activations --cpu_offload

# If you want to resume training from a checkpoint, uncomment the line below
# python train.py --config config.json --checkpoint_activations --cpu_offload --resume checkpoints/checkpoint_latest.pt 