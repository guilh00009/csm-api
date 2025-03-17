#!/usr/bin/env python
"""
Diagnostic script to test RoPE initialization with different parameters
to identify what causes the "len() of a 0-d tensor" error.
"""

import torch
import os
import sys
import traceback

# Set environment variables for better error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def test_rope_init(dim, max_seq_len, base, scale_factor):
    """Test RoPE initialization with specific parameters"""
    print(f"\nTesting RoPE with: dim={dim}, max_seq_len={max_seq_len}, base={base}, scale_factor={scale_factor}")
    
    try:
        import torchtune.models.llama3_1._position_embeddings
        
        # Create instance
        rope = torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE(
            dim=dim, 
            max_seq_len=max_seq_len, 
            base=base, 
            scale_factor=scale_factor
        )
        print("✓ Success! RoPE initialization worked!")
        return True, None
    except Exception as e:
        print(f"✗ Failed with error: {e}")
        return False, e

def print_traceback_for_rope_init(dim, max_seq_len, base, scale_factor):
    """Print full traceback for RoPE initialization with specific parameters"""
    print(f"\n--- DETAILED ERROR FOR: dim={dim}, max_seq_len={max_seq_len}, base={base}, scale_factor={scale_factor} ---")
    
    try:
        import torchtune.models.llama3_1._position_embeddings
        
        # Create instance
        rope = torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE(
            dim=dim, 
            max_seq_len=max_seq_len, 
            base=base, 
            scale_factor=scale_factor
        )
        print("Success! No error to show.")
    except Exception:
        traceback.print_exc()
    
    print("---" * 20)

def check_module_source():
    """Check the source code of the RoPE implementation"""
    try:
        import torchtune.models.llama3_1._position_embeddings
        import inspect
        
        # Get the source code
        source = inspect.getsource(torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.rope_init)
        source_apply_scaling = inspect.getsource(torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.apply_scaling)
        
        print("\n--- RoPE INIT SOURCE CODE ---")
        print(source)
        print("\n--- APPLY SCALING SOURCE CODE ---")
        print(source_apply_scaling)
    except Exception as e:
        print(f"Failed to get source code: {e}")

def manual_rope_init(dim, max_seq_len, base=10000.0, scale_factor=1.0):
    """Try to implement RoPE initialization manually to see if it works"""
    print(f"\nTrying manual RoPE init: dim={dim}, max_seq_len={max_seq_len}, base={base}")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate half dimension
        half_dim = dim // 2
        
        # Calculate frequencies
        # Original
        freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))
        
        # Apply scaling if needed
        if scale_factor != 1.0:
            freqs = freqs ** scale_factor
        
        # Generate position indices
        seq_idx = torch.arange(max_seq_len, device=device).float()
        
        # Calculate embeddings
        emb = torch.outer(seq_idx, freqs)
        
        # Calculate cos/sin values
        cos_cached = torch.cos(emb).float()
        sin_cached = torch.sin(emb).float()
        
        print("✓ Manual implementation worked!")
        print(f"  Shape of cos_cached: {cos_cached.shape}")
        return True
    except Exception as e:
        print(f"✗ Manual implementation failed: {e}")
        traceback.print_exc()
        return False

def patch_and_test():
    """Apply patches and test if they fix the issue"""
    print("\n--- TESTING WITH PATCHED VERSION ---")
    
    try:
        import torchtune.models.llama3_1._position_embeddings
        
        # Store original method
        original_rope_init = torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.rope_init
        original_apply_scaling = torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.apply_scaling
        
        # Create patched methods
        def patched_rope_init(self):
            """Patched version of rope_init"""
            try:
                # Try original first
                original_rope_init(self)
            except Exception as e:
                print(f"Original init failed: {e}")
                print("Using fallback implementation")
                
                with torch.no_grad():
                    # Safe implementation
                    half_dim = self.dim // 2
                    freqs = torch.arange(0, half_dim, 2, device=self.device).float()
                    freqs = 1.0 / (10000.0 ** (freqs / half_dim))
                    
                    if hasattr(self, 'scale') and self.scale != 1.0:
                        try:
                            freqs = freqs ** self.scale
                        except Exception as scale_e:
                            print(f"Error applying scale: {scale_e}")
                    
                    # Create position indices
                    seq_idx = torch.arange(min(self.max_seq_len, 4096), device=self.device).float()
                    
                    # Calculate embeddings
                    emb = torch.outer(seq_idx, freqs)
                    
                    # Register buffers
                    self.register_buffer("cos_cached", torch.cos(emb).float(), persistent=False)
                    self.register_buffer("sin_cached", torch.sin(emb).float(), persistent=False)
        
        def patched_apply_scaling(self, freqs):
            """Patched version of apply_scaling"""
            try:
                # First check if freqs is a scalar
                if isinstance(freqs, torch.Tensor) and freqs.dim() == 0:
                    freqs_value = freqs.item()
                    return torch.tensor([freqs_value], dtype=freqs.dtype, device=freqs.device)
                
                # Try original implementation
                return original_apply_scaling(self, freqs)
            except Exception as e:
                print(f"Original apply_scaling failed: {e}")
                
                # Safe fallback
                if isinstance(freqs, torch.Tensor):
                    if freqs.dim() == 0:
                        freqs_value = freqs.item()
                        return torch.tensor([freqs_value], dtype=freqs.dtype, device=freqs.device)
                    else:
                        return freqs.reshape(-1)  # Ensure it's a 1D tensor
                else:
                    return torch.tensor([freqs], dtype=torch.float, device=self.device)
        
        # Apply patches
        torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.rope_init = patched_rope_init
        torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.apply_scaling = patched_apply_scaling
        
        # Test challenging cases
        challenging_params = [
            (128, 32768, 500000, 32),  # Most challenging
            (128, 4096, 10000, 1),     # Standard
        ]
        
        for dim, max_seq_len, base, scale_factor in challenging_params:
            success, _ = test_rope_init(dim, max_seq_len, base, scale_factor)
            if success:
                print(f"✓ Patched version fixed the issue for dim={dim}, max_seq_len={max_seq_len}, base={base}, scale_factor={scale_factor}")
            else:
                print(f"✗ Patched version still fails for dim={dim}, max_seq_len={max_seq_len}, base={base}, scale_factor={scale_factor}")
    
    except Exception as e:
        print(f"Error during patching: {e}")
        traceback.print_exc()
    finally:
        # Restore original methods
        try:
            torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.rope_init = original_rope_init
            torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.apply_scaling = original_apply_scaling
        except:
            pass

def main():
    """Main diagnostic function"""
    print("=== RoPE INITIALIZATION DIAGNOSTIC TOOL ===")
    
    # Check if torchtune is installed
    try:
        import torchtune
        print("✓ torchtune is installed")
    except ImportError:
        print("✗ torchtune is not installed!")
        sys.exit(1)
    
    # Try to view source code
    check_module_source()
    
    # Test various parameters
    test_cases = [
        # (dimension, max_seq_len, base, scale_factor)
        (128, 2048, 10000.0, 1.0),      # Standard parameters
        (128, 32768, 10000.0, 1.0),     # Large context
        (128, 2048, 500000.0, 1.0),     # Large base
        (128, 2048, 10000.0, 32.0),     # Large scale
        (128, 32768, 500000.0, 32.0),   # All large (likely fails)
    ]
    
    # Run tests
    results = []
    for dim, max_seq_len, base, scale_factor in test_cases:
        success, error = test_rope_init(dim, max_seq_len, base, scale_factor)
        results.append((dim, max_seq_len, base, scale_factor, success, error))
    
    # Print summary
    print("\n=== TEST RESULTS SUMMARY ===")
    for dim, max_seq_len, base, scale_factor, success, error in results:
        status = "✓ Success" if success else "✗ Failed"
        error_msg = f" - Error: {error}" if error else ""
        print(f"{status}: dim={dim}, max_seq_len={max_seq_len}, base={base}, scale_factor={scale_factor}{error_msg}")
    
    # Print detailed error for the first failure
    failures = [(dim, max_seq_len, base, scale_factor) for dim, max_seq_len, base, scale_factor, success, _ in results if not success]
    if failures:
        print_traceback_for_rope_init(*failures[0])
    
    # Try manual implementation
    manual_rope_init(128, 2048)
    manual_rope_init(128, 32768, 500000.0, 32.0)
    
    # Test patched version
    patch_and_test()
    
    print("\n=== DIAGNOSTIC COMPLETE ===")

if __name__ == "__main__":
    main() 