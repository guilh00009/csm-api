"""
Model wrapper that provides safer initialization and fallback options.
"""

import torch
import os
import warnings
from models import Model, ModelArgs, FLAVORS
import torch.nn as nn

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
                    # Use CUDA if available, otherwise CPU
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
                    # Calculate freqs directly
                    half_dim = self.dim // 2
                    freqs = 1.0 / (10000.0 ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))
                    
                    # Generate position indices
                    seq_idx = torch.arange(min(self.max_seq_len, 4096), device=device).float()
                    
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
    Create a model with safer initialization, handling potential errors.
    
    Args:
        model_args: ModelArgs instance with model configuration
        device: Device to place model on
        dtype: Data type for model parameters
        
    Returns:
        Model instance
    """
    try:
        if dtype is None:
            # Determine best dtype automatically
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif torch.cuda.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32
        
        print(f"Creating model with {model_args.backbone_flavor} backbone in {dtype} precision")
        model = Model(model_args).to(device=device, dtype=dtype)
        
        # Apply safety patches to the model
        _apply_safety_patches(model)
        
        return model
    except Exception as e:
        print(f"Error during model creation: {e}")
        # Fallback to smaller model if the main one fails
        if model_args.backbone_flavor != "llama-1B":
            print("Falling back to llama-1B model...")
            model_args.backbone_flavor = "llama-1B"
            return create_model_safely(model_args, device, dtype)
        else:
            # If even the smallest model fails, re-raise the error
            raise

def _apply_safety_patches(model):
    """Apply safety patches to the model components to handle edge cases"""
    
    # Apply attention mask dimension fix to all attention modules
    for module in model.modules():
        if hasattr(module, '_attention_call'):
            original_attention_call = module._attention_call
            
            def safe_attention_call(self, *args, **kwargs):
                """Make sure attention mask dimensions are compatible with scaled_dot_product_attention"""
                try:
                    # Extract mask from kwargs
                    mask = kwargs.get('mask')
                    
                    # Check if we have a mask that needs reshaping
                    if mask is not None and not hasattr(mask, '_safe_mask_processed'):
                        # Get mask shape
                        mask_shape = mask.shape
                        
                        # Check if mask needs to be reshaped for compatibility
                        if len(mask_shape) == 3:  # [batch, seq_len, max_seq_len]
                            # Reshape for scaled_dot_product_attention which expects [batch, heads, seq_len, max_seq_len]
                            kwargs['mask'] = mask.unsqueeze(1)
                            kwargs['mask']._safe_mask_processed = True
                    
                    # Call the original attention method with the modified kwargs
                    return original_attention_call(self, *args, **kwargs)
                except RuntimeError as e:
                    if "expanded size" in str(e) and 'mask' in kwargs:
                        # Handle dimension mismatch in attention mask
                        print(f"Fixing attention mask dimensions: {e}")
                        
                        # Try to adapt the mask
                        mask = kwargs.pop('mask', None)  # Remove mask from kwargs to avoid duplicate
                        
                        # If mask is 3D, reshape it to 4D
                        if mask is not None and len(mask.shape) == 3:
                            new_mask = mask.unsqueeze(1)
                            # Call with modified mask
                            return original_attention_call(self, *args, mask=new_mask, **kwargs)
                        
                        # If mask is 4D but has wrong sequence dimension, try to fix it
                        if mask is not None and len(mask.shape) == 4:
                            # Get q input to determine expected sequence length
                            q = args[0] if len(args) > 0 else kwargs.get('q')
                            if q is not None:
                                seq_len = q.shape[1]
                                # Create new causal mask with right dimensions
                                device = mask.device
                                causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
                                # Reshape for compatibility
                                new_mask = causal_mask.unsqueeze(0).unsqueeze(1)
                                # Call with fixed mask
                                return original_attention_call(self, *args, mask=new_mask, **kwargs)
                        
                        # If we couldn't fix the mask, try without it
                        print("Couldn't fix mask, trying without it...")
                        return original_attention_call(self, *args, **kwargs)
                    elif "multiple values for argument 'mask'" in str(e):
                        # Handle duplicate mask parameter
                        print("Fixing duplicate mask parameter")
                        
                        # Remove mask from kwargs to avoid duplicate
                        mask = kwargs.pop('mask', None)
                        
                        # Try to extract the mask from args if it's there
                        if len(args) >= 3 and isinstance(args[2], torch.Tensor):
                            # Already have mask in args, just pass args as is
                            return original_attention_call(self, *args, **kwargs)
                        else:
                            # Put mask back as a kwarg
                            return original_attention_call(self, *args, mask=mask, **kwargs)
                    else:
                        # For other errors, just propagate them up
                        raise
            
            # Bind the method to the module
            module._attention_call = safe_attention_call.__get__(module, module.__class__)
    
    return model

def create_model_safely_old(
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