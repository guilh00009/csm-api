#!/usr/bin/env python
"""
Utility script to convert checkpoint tensor dtypes for consistency.
This script can take a checkpoint file and convert all tensors to a specific dtype.
"""

import torch
import argparse
import os
from pathlib import Path

def convert_tensor_dtypes(tensor, target_dtype):
    """Convert tensor to target dtype if compatible."""
    if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
        return tensor.to(dtype=target_dtype)
    return tensor

def get_best_dtype():
    """Get the best dtype for the current system."""
    if not torch.cuda.is_available():
        return torch.float32
    
    # Check for bfloat16 support first
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # Fall back to float16 if bfloat16 not supported
    else:
        return torch.float16

def convert_checkpoint(checkpoint_path, output_path=None, target_dtype=None):
    """
    Convert all float tensors in checkpoint to target dtype.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Path to save the converted checkpoint (defaults to original with _converted suffix)
        target_dtype: Target dtype (defaults to bfloat16 on supported hardware, otherwise float16)
    
    Returns:
        Path to the converted checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Determine target dtype if not specified
    if target_dtype is None:
        target_dtype = get_best_dtype()
        print(f"Using {target_dtype} as target dtype based on hardware capabilities")
    elif isinstance(target_dtype, str):
        # Convert string to torch.dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float": torch.float32,
            "half": torch.float16,
            "bfloat": torch.bfloat16,
        }
        target_dtype = dtype_map.get(target_dtype.lower(), torch.bfloat16)
        print(f"Using {target_dtype} as target dtype from string specification")
    
    # Determine output path if not specified
    if output_path is None:
        original_path = Path(checkpoint_path)
        output_path = original_path.with_name(f"{original_path.stem}_converted{original_path.suffix}")
    
    # Convert tensors in state_dict
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        print(f"Converting model state_dict tensors to {target_dtype}")
        converted_count = 0
        total_count = 0
        
        for key, value in checkpoint["model"].items():
            total_count += 1
            if isinstance(value, torch.Tensor) and value.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                original_dtype = value.dtype
                checkpoint["model"][key] = convert_tensor_dtypes(value, target_dtype)
                converted_count += 1
                
                # Print dtype conversion statistics
                if converted_count <= 5 or converted_count % 100 == 0:
                    print(f"Converted {key}: {original_dtype} -> {checkpoint['model'][key].dtype}")
        
        print(f"Converted {converted_count} out of {total_count} tensors to {target_dtype}")
    
    # Convert optimizer state if present
    if "optimizer" in checkpoint and isinstance(checkpoint["optimizer"], dict):
        print(f"Converting optimizer state tensors to {target_dtype}")
        optimizer_converted = 0
        optimizer_total = 0
        
        # Handle 'state' part of optimizer state_dict
        if "state" in checkpoint["optimizer"]:
            for param_id, param_state in checkpoint["optimizer"]["state"].items():
                for state_key, state_value in param_state.items():
                    optimizer_total += 1
                    if isinstance(state_value, torch.Tensor) and state_value.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        checkpoint["optimizer"]["state"][param_id][state_key] = convert_tensor_dtypes(state_value, target_dtype)
                        optimizer_converted += 1
        
        print(f"Converted {optimizer_converted} out of {optimizer_total} optimizer tensors to {target_dtype}")
    
    # Save converted checkpoint
    print(f"Saving converted checkpoint to {output_path}")
    torch.save(checkpoint, output_path)
    
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint tensor dtypes for consistency.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file")
    parser.add_argument("--output", type=str, default=None, help="Path to save the converted checkpoint")
    parser.add_argument("--dtype", type=str, default=None, 
                        choices=["float32", "float16", "bfloat16"], 
                        help="Target dtype (default: best available)")
    
    args = parser.parse_args()
    
    output_path = convert_checkpoint(args.checkpoint_path, args.output, args.dtype)
    
    print(f"Checkpoint conversion complete. Saved to {output_path}")

if __name__ == "__main__":
    main() 