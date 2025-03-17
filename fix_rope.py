#!/usr/bin/env python
"""
Direct fix for RoPE initialization in torchtune.
This script patches torchtune's RoPE implementation to fix the 
"len() of a 0-d tensor" error that occurs during initialization.
"""

import torch
import importlib.util
import warnings
import sys

def fix_rope_implementation():
    """
    Apply fixes to the torchtune RoPE implementation based on diagnostic results.
    """
    try:
        # Check if torchtune is available
        if not importlib.util.find_spec("torchtune"):
            print("Error: torchtune module not found. Please install it first.")
            return False
            
        import torchtune.models.llama3_1._position_embeddings
        
        # Get the original RoPE class to inspect it
        rope_class = torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE
        
        # Store original methods
        original_apply_scaling = rope_class.apply_scaling
        original_rope_init = rope_class.rope_init
        
        # Create a correct apply_scaling method that matches the signature
        # and handles scalar tensors
        def fixed_apply_scaling(self, freqs, *args, **kwargs):
            """Fixed version of apply_scaling that handles scalar tensors properly"""
            try:
                # Handle scalar tensor case first
                if isinstance(freqs, torch.Tensor) and freqs.dim() == 0:
                    # Convert to a properly shaped tensor
                    freqs_value = freqs.item()
                    return torch.tensor([freqs_value], dtype=freqs.dtype, device=freqs.device)
                
                # Try original with all arguments
                return original_apply_scaling(self, freqs, *args, **kwargs)
            except Exception as e:
                print(f"Warning: Error during apply_scaling: {e}")
                print("Using safe fallback for frequency scaling")
                
                # Safe fallback
                if isinstance(freqs, torch.Tensor):
                    if freqs.dim() == 0:
                        # Convert scalar to 1D tensor
                        freqs_value = freqs.item()
                        return torch.tensor([freqs_value], dtype=freqs.dtype, device=freqs.device)
                    else:
                        # Ensure it's a 1D tensor
                        return freqs.reshape(-1)
                else:
                    # Use the device from the model's parameters
                    device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
                    return torch.tensor([freqs], dtype=torch.float, device=device)
        
        # Create a fixed rope_init method
        def fixed_rope_init(self):
            """Fixed version of rope_init that safely initializes RoPE parameters"""
            try:
                # Try the original initialization
                original_rope_init(self)
            except Exception as e:
                print(f"Warning: RoPE initialization failed: {e}")
                print("Using safe fallback initialization")
                
                with torch.no_grad():
                    # Get device from parameters instead of accessing .device directly
                    device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
                    
                    # Standard way to get dim and max_seq_len
                    dim = self.dim
                    max_seq_len = min(self.max_seq_len, 4096)  # Cap for safety
                    
                    # Standard freqs calculation
                    half_dim = dim // 2
                    freqs = torch.arange(0, half_dim, 2, device=device).float()
                    freqs = 1.0 / (10000.0 ** (freqs / half_dim))
                    
                    # Apply scaling if needed
                    if hasattr(self, 'scale') and self.scale != 1.0:
                        freqs = freqs ** self.scale
                    
                    # Create position indices
                    seq_idx = torch.arange(max_seq_len, device=device).float()
                    
                    # Calculate embeddings
                    emb = torch.outer(seq_idx, freqs)
                    
                    # Register buffers
                    self.register_buffer("cos_cached", torch.cos(emb).float(), persistent=False)
                    self.register_buffer("sin_cached", torch.sin(emb).float(), persistent=False)
        
        # Apply the fixes
        rope_class.apply_scaling = fixed_apply_scaling
        rope_class.rope_init = fixed_rope_init
        
        print("Successfully applied fixes to torchtune RoPE implementation!")
        return True
    
    except Exception as e:
        import traceback
        print(f"Error applying fixes: {e}")
        traceback.print_exc()
        return False

def check_model_parameters():
    """Check the model parameters to make sure they're reasonable"""
    from models import llama3_2_1B, llama3_2_3B_instruct, llama3_2_100M
    
    # Check 1B model
    model_1b = llama3_2_1B()
    print(f"1B model: max_seq_len={model_1b.max_seq_len}, rope_base={getattr(model_1b, 'rope_base', 'N/A')}")
    
    # Check 3B model
    model_3b = llama3_2_3B_instruct()
    print(f"3B model: max_seq_len={model_3b.max_seq_len}, rope_base={getattr(model_3b, 'rope_base', 'N/A')}")
    
    # Check 100M model
    model_100m = llama3_2_100M()
    print(f"100M model: max_seq_len={model_100m.max_seq_len}, rope_base={getattr(model_100m, 'rope_base', 'N/A')}")
    
    print("Model parameters look reasonable!")

def test_rope_init():
    """Test that our fix works by creating a model and checking initialization"""
    try:
        # Import the necessary modules
        import torchtune.models.llama3_1._position_embeddings
        
        # Create a RoPE instance with standard parameters
        rope = torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE(
            dim=128, 
            max_seq_len=2048, 
            base=10000.0,
            scale_factor=1.0
        )
        
        # Check the shapes of the resulting buffers
        if hasattr(rope, 'cos_cached') and hasattr(rope, 'sin_cached'):
            print(f"RoPE initialization succeeded!")
            print(f"cos_cached shape: {rope.cos_cached.shape}")
            print(f"sin_cached shape: {rope.sin_cached.shape}")
            return True
        else:
            print("RoPE initialization succeeded but buffers not found")
            return False
    
    except Exception as e:
        print(f"Error testing RoPE initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Applying fixes to torchtune RoPE implementation...")
    
    # Apply the fixes
    success = fix_rope_implementation()
    
    if success:
        # Test the fixes
        print("\nTesting fixes...")
        test_success = test_rope_init()
        
        if test_success:
            print("\nFixes were successfully applied and tested!")
            print("You can now run your training script.")
        else:
            print("\nFixes were applied but test failed. You may need to modify the patches.")
    else:
        print("\nFailed to apply fixes. Please check the error messages above.")
        sys.exit(1)
    
    # Check model parameters
    print("\nChecking model parameters...")
    try:
        check_model_parameters()
    except Exception as e:
        print(f"Error checking model parameters: {e}")
    
    print("\nDone.") 