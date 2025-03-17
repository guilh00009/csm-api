#!/usr/bin/env python
"""
Utility script to convert model checkpoints to a consistent dtype.
This can be helpful when loading checkpoints trained with different precision settings.

Usage:
    python convert_checkpoint.py --input checkpoint.pt --output converted_checkpoint.pt --dtype bfloat16
"""

import argparse
import os
import torch
from models import Model, ModelArgs

def get_best_dtype():
    """Determine the best dtype to use based on CUDA capabilities."""
    if not torch.cuda.is_available():
        return torch.float32
    
    # Check for bfloat16 support first
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # Fall back to float16 if bfloat16 not supported
    else:
        return torch.float16

def convert_checkpoint(input_path, output_path, dtype_name):
    """Convert a checkpoint to the specified dtype."""
    print(f"Loading checkpoint from {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")
    
    # Determine the dtype
    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "float32":
        dtype = torch.float32
    elif dtype_name == "auto":
        dtype = get_best_dtype()
        print(f"Auto-detected best dtype: {dtype}")
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    
    # Convert model weights
    if "model" in checkpoint:
        print("Converting model weights...")
        model_state = checkpoint["model"]
        for k, v in model_state.items():
            # Skip non-tensor values or tensors that shouldn't be converted
            if not isinstance(v, torch.Tensor) or v.dtype in [torch.long, torch.bool, torch.int]:
                continue
            
            # Convert tensors to the specified dtype
            model_state[k] = v.to(dtype)
    else:
        # Assume the checkpoint is just the model state dict
        print("Converting model weights (state dict only)...")
        for k, v in checkpoint.items():
            # Skip non-tensor values or tensors that shouldn't be converted
            if not isinstance(v, torch.Tensor) or v.dtype in [torch.long, torch.bool, torch.int]:
                continue
            
            # Convert tensors to the specified dtype
            checkpoint[k] = v.to(dtype)
    
    # Save the converted checkpoint
    print(f"Saving converted checkpoint to {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(checkpoint, output_path)
    print("Conversion completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Convert model checkpoints to a consistent dtype")
    parser.add_argument("--input", type=str, required=True, help="Path to the input checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save the converted checkpoint")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"],
                       help="Target dtype for the model weights")
    
    args = parser.parse_args()
    convert_checkpoint(args.input, args.output, args.dtype)

if __name__ == "__main__":
    main() 