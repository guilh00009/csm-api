"""
Model wrapper that provides safer initialization and fallback options.
"""

import torch
import os
import warnings
from models import Model, ModelArgs, FLAVORS

def patch_llama_before_import():
    """
    Apply critical patches to torchtune before importing Llama models.
    This helps prevent initialization errors.
    """
    try:
        import torchtune.models.llama3_1._position_embeddings
        
        # Get the original rope_init method to patch
        original_rope_init = torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.rope_init
        
        # Define a safer rope_init method
        def safe_rope_init(self):
            """A safer implementation of rope_init that handles edge cases"""
            try:
                # Try the original implementation first
                original_rope_init(self)
            except Exception as e:
                print(f"RoPE initialization failed with error: {e}")
                print("Using direct fallback implementation...")
                
                # Direct fallback without relying on any existing code
                with torch.no_grad():
                    # Calculate freqs directly
                    half_dim = self.dim // 2
                    freqs = 1.0 / (10000.0 ** (torch.arange(0, half_dim, 2, device=self.device).float() / half_dim))
                    
                    # Generate position indices
                    seq_idx = torch.arange(self.max_seq_len, device=self.device).float()
                    
                    # Calculate emb using broadcasting
                    emb = torch.outer(seq_idx, freqs)
                    
                    # Register the sine and cosine buffers
                    self.register_buffer("cos_cached", torch.cos(emb).float(), persistent=False)
                    self.register_buffer("sin_cached", torch.sin(emb).float(), persistent=False)
        
        # Apply the patch
        torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.rope_init = safe_rope_init
        print("Applied direct pre-import patching to Llama3ScaledRoPE.rope_init")
        
        return True
    except Exception as e:
        print(f"Failed to apply pre-import patch: {e}")
        return False

def create_model_safely(
    model_args: ModelArgs,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype=None
) -> Model:
    """
    Create a model with safer initialization and fallback options.
    
    Args:
        model_args: ModelArgs instance with model configuration
        device: Device to place the model on
        dtype: Data type for model parameters
        
    Returns:
        Initialized Model instance
    """
    # Apply critical patches before model creation
    patch_applied = patch_llama_before_import()
    
    # Determine best dtype if not provided
    if dtype is None:
        if not torch.cuda.is_available():
            dtype = torch.float32
        elif torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    
    # Try to initialize the model with standard parameters
    try:
        print(f"Attempting to initialize model with backbone: {model_args.backbone_flavor}")
        model = Model(model_args)
        
        # Move model to target device and dtype with better error handling
        try:
            model = model.to(device=device, dtype=dtype)
        except Exception as dtype_e:
            print(f"Warning: Could not convert model to {dtype}: {dtype_e}")
            print("Falling back to float32 for model parameters")
            model = model.to(device=device, dtype=torch.float32)
        
        print("Model initialized successfully!")
        return model
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        
        # First fallback: try with completely safe values
        if "len() of a 0-d tensor" in str(e) or "max_seq_len" in str(e):
            print("Trying fallback with completely safe parameters...")
            try:
                # Create a new instance of ModelArgs with reduced context
                backup_args = ModelArgs(
                    backbone_flavor=model_args.backbone_flavor,
                    decoder_flavor=model_args.decoder_flavor,
                    text_vocab_size=model_args.text_vocab_size,
                    audio_vocab_size=model_args.audio_vocab_size,
                    audio_num_codebooks=model_args.audio_num_codebooks,
                )
                
                # Manually patch the model creation functions for all flavors
                from models import llama3_2
                
                def safe_llama_3B():
                    """Completely safe version of llama 3B that should initialize"""
                    return llama3_2.llama3_2(
                        vocab_size=128_256,
                        num_layers=28,
                        num_heads=24,
                        num_kv_heads=8,
                        embed_dim=3072,
                        max_seq_len=1024,  # Very conservative value
                        intermediate_dim=8192,
                        attn_dropout=0.0,
                        norm_eps=1e-5,
                        rope_base=10000.0,  # Standard value
                        scale_factor=1.0,   # No scaling
                    )
                
                def safe_llama_1B():
                    """Completely safe version of llama 1B that should initialize"""
                    return llama3_2.llama3_2(
                        vocab_size=128_256,
                        num_layers=16,
                        num_heads=32,
                        num_kv_heads=8,
                        embed_dim=2048,
                        max_seq_len=1024,  # Very conservative value
                        intermediate_dim=8192,
                        attn_dropout=0.0,
                        norm_eps=1e-5,
                        rope_base=10000.0,  # Standard value
                        scale_factor=1.0,   # No scaling
                    )
                
                def safe_llama_100M():
                    """Completely safe version of llama 100M that should initialize"""
                    return llama3_2.llama3_2(
                        vocab_size=128_256,
                        num_layers=4,
                        num_heads=8,
                        num_kv_heads=2,
                        embed_dim=1024,
                        max_seq_len=1024,  # Very conservative value
                        intermediate_dim=4096,
                        attn_dropout=0.0,
                        norm_eps=1e-5,
                        rope_base=10000.0,  # Standard value
                        scale_factor=1.0,   # No scaling
                    )
                
                # Store all original flavors
                original_flavors = {}
                for flavor_name in FLAVORS:
                    original_flavors[flavor_name] = FLAVORS[flavor_name]
                
                # Apply safe versions
                FLAVORS["llama-3B-instruct"] = safe_llama_3B
                FLAVORS["llama-1B"] = safe_llama_1B
                FLAVORS["llama-100M"] = safe_llama_100M
                
                try:
                    model = Model(backup_args)
                    model = model.to(device=device, dtype=torch.float32)  # Use float32 for safety
                    print("Model initialized successfully with ultra-conservative fallback parameters!")
                    return model
                except Exception as fallback_e:
                    print(f"Conservative fallback also failed: {fallback_e}")
                finally:
                    # Restore original flavors
                    for flavor_name, flavor_fn in original_flavors.items():
                        FLAVORS[flavor_name] = flavor_fn
            except Exception as fallback_e:
                print(f"Setup for fallback initialization failed: {fallback_e}")
        
        # If we get here, all initialization attempts failed
        raise RuntimeError(f"Could not initialize model: {e}") 