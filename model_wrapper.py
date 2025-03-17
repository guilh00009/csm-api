"""
Model wrapper that provides safer initialization and fallback options.
"""

import torch
import os
import warnings
from models import Model, ModelArgs, FLAVORS

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
        model = Model(model_args).to(device=device, dtype=dtype)
        print("Model initialized successfully!")
        return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        
        # First fallback: try with smaller context window
        if "max_seq_len" in str(e) or "len() of a 0-d tensor" in str(e):
            print("Trying fallback with smaller context window...")
            try:
                # Create a new instance of ModelArgs with reduced context
                backup_args = ModelArgs(
                    backbone_flavor=model_args.backbone_flavor,
                    decoder_flavor=model_args.decoder_flavor,
                    text_vocab_size=model_args.text_vocab_size,
                    audio_vocab_size=model_args.audio_vocab_size,
                    audio_num_codebooks=model_args.audio_num_codebooks,
                )
                
                # Override FLAVORS to use more conservative parameters
                from models import llama3_2
                
                def safer_llama_3B():
                    return llama3_2.llama3_2(
                        vocab_size=128_256,
                        num_layers=28,
                        num_heads=24,
                        num_kv_heads=8,
                        embed_dim=3072,
                        max_seq_len=4096,  # Very conservative value
                        intermediate_dim=8192,
                        attn_dropout=0.0,
                        norm_eps=1e-5,
                        rope_base=10000.0,  # Standard value
                        scale_factor=1.0,   # No scaling
                    )
                
                # Temporarily replace the flavor
                original_flavor = FLAVORS.get(model_args.backbone_flavor)
                FLAVORS[model_args.backbone_flavor] = safer_llama_3B
                
                try:
                    model = Model(backup_args).to(device=device, dtype=dtype)
                    print("Model initialized successfully with fallback parameters!")
                    return model
                finally:
                    # Restore original flavor
                    if original_flavor:
                        FLAVORS[model_args.backbone_flavor] = original_flavor
            except Exception as fallback_e:
                print(f"Fallback initialization also failed: {fallback_e}")
        
        # If we get here, all initialization attempts failed
        raise RuntimeError(f"Could not initialize model: {e}") 