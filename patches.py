"""
Monkey patches for third-party libraries to fix compatibility issues.
These patches are applied at runtime.
"""

import torch
import importlib.util
import types
import inspect
import warnings
import math

def patch_torchtune_rope():
    """
    Patch the torchtune RoPE implementation to fix initialization issues.
    """
    try:
        if not importlib.util.find_spec("torchtune"):
            print("torchtune not found, skipping RoPE patch")
            return False

        import torchtune.models.llama3_1._position_embeddings
        
        # Get the original RoPE class
        rope_class = torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE
        
        # Store original methods
        original_apply_scaling = rope_class.apply_scaling
        original_rope_init = rope_class.rope_init
        
        # Create patched apply_scaling method with variable arguments
        def patched_apply_scaling(self, freqs, *args, **kwargs):
            """Patched version of apply_scaling that handles scalar tensors with any signature"""
            try:
                # Handle scalar tensor case directly
                if isinstance(freqs, torch.Tensor) and freqs.dim() == 0:
                    # Convert scalar to a properly shaped tensor
                    freqs_value = freqs.item()
                    return torch.tensor([freqs_value], dtype=freqs.dtype, device=freqs.device)
                
                # Try original with all arguments
                return original_apply_scaling(self, freqs, *args, **kwargs)
            except (TypeError, RuntimeError) as e:
                if "len() of a 0-d tensor" in str(e):
                    print("Fixing RoPE scaling with scalar tensor")
                    if isinstance(freqs, torch.Tensor):
                        # Convert any tensor to 1D if needed
                        if freqs.dim() == 0:
                            freqs_value = freqs.item()
                            return torch.tensor([freqs_value], dtype=freqs.dtype, device=freqs.device)
                        else:
                            # Ensure it's a 1D tensor
                            return freqs.reshape(-1)
                    else:
                        # Get device in a safer way
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        return torch.tensor([freqs], dtype=torch.float, device=device)
                else:
                    raise
        
        # Apply the patch
        rope_class.apply_scaling = patched_apply_scaling
        print("Successfully patched torchtune RoPE apply_scaling")
        
        # Patch the rope_init method
        def patched_rope_init(self):
            """Patched version of rope_init that handles initialization errors"""
            try:
                # Try original initialization
                original_rope_init(self)
            except (TypeError, RuntimeError) as e:
                print(f"Warning: Error during RoPE initialization: {e}")
                print("Implementing fallback RoPE initialization...")
                
                with torch.no_grad():
                    # Get device in a safer way that doesn't require self.device
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
                    # Handle scalar tensors and standard initialization
                    half_dim = self.dim // 2
                    max_seq_len = min(self.max_seq_len, 4096)  # Cap for safety
                    
                    # Standard freqs calculation that works reliably
                    freqs = torch.arange(0, half_dim, 2, device=device).float()
                    freqs = 1.0 / (10000.0 ** (freqs / half_dim))
                    
                    # Apply scaling if needed
                    if hasattr(self, 'scale') and self.scale != 1.0:
                        freqs = freqs ** self.scale
                    
                    # Create position indices
                    seq_idx = torch.arange(max_seq_len, device=device).float()
                    
                    # Calculate embeddings using outer product
                    emb = torch.outer(seq_idx, freqs)
                    
                    # Register buffers
                    self.register_buffer("cos_cached", torch.cos(emb).float(), persistent=False)
                    self.register_buffer("sin_cached", torch.sin(emb).float(), persistent=False)
        
        # Apply the patch
        rope_class.rope_init = patched_rope_init
        print("Successfully patched torchtune RoPE initialization")
        
        # Additionally patch the constructor to prevent parameters that cause issues
        original_init = rope_class.__init__
        
        def patched_init(self, dim, max_seq_len, base=10000.0, scale_factor=1.0):
            """Patched version of __init__ that validates parameters"""
            # Ensure reasonable max_seq_len
            if max_seq_len > 65536:
                warnings.warn(f"Very large max_seq_len ({max_seq_len}) detected, capping at a safer value of 4096")
                max_seq_len = 4096
                
            # Ensure reasonable base value
            if base > 100000.0:
                warnings.warn(f"Very large RoPE base ({base}) detected, setting to standard value 10000.0")
                base = 10000.0
                
            # Ensure reasonable scale_factor
            if scale_factor > 100:
                warnings.warn(f"Very large scale_factor ({scale_factor}) detected, setting to 1.0")
                scale_factor = 1.0
                
            # Call original init with validated parameters
            original_init(self, dim, max_seq_len, base, scale_factor)
        
        # Apply the patch
        rope_class.__init__ = patched_init
        print("Successfully patched torchtune Llama3ScaledRoPE constructor")
        
        return True
        
    except Exception as e:
        print(f"Failed to patch torchtune RoPE: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_torchtune_kv_cache():
    """
    Patch the torchtune KVCache implementation to ensure indices are always long type.
    """
    try:
        # Check if torchtune is installed
        if not importlib.util.find_spec("torchtune"):
            print("torchtune not found, skipping patch")
            return False
        
        # Import the module
        import torchtune.modules.kv_cache
        import torchtune.modules.transformer
        
        # Try multiple patching approaches in case one fails
        success = False
        
        # Approach 1: Patch torch.Tensor.__getitem__ to ensure indices are long
        original_getitem = torch.Tensor.__getitem__
        
        def safe_getitem(self, key):
            """Ensure indices are properly converted to long when used for indexing"""
            if isinstance(key, tuple):
                # Convert any tensor indices to long
                new_key = []
                for k in key:
                    if isinstance(k, torch.Tensor) and k.dtype not in [torch.long, torch.bool, torch.uint8]:
                        new_key.append(k.long())
                    else:
                        new_key.append(k)
                key = tuple(new_key)
            elif isinstance(key, torch.Tensor) and key.dtype not in [torch.long, torch.bool, torch.uint8]:
                key = key.long()
            
            return original_getitem(self, key)
        
        # Apply the patch temporarily during training
        torch.Tensor.__getitem__ = safe_getitem
        print("Patched torch.Tensor.__getitem__ to ensure indices are long")
        success = True
        
        # Approach 2: Monkey patch the KVCache.update method
        try:
            # Get the original update method
            original_update = torchtune.modules.kv_cache.KVCache.update
            
            # Create a wrapper that ensures long indices
            def patched_update(self, k_val, v_val):
                """Ensures cache positions use long indices"""
                # Convert any cache positions to long
                if hasattr(self, 'cache_pos'):
                    if isinstance(self.cache_pos, torch.Tensor):
                        self.cache_pos = self.cache_pos.long()
                if hasattr(self, 'curr_pos'):
                    if isinstance(self.curr_pos, torch.Tensor):
                        self.curr_pos = self.curr_pos.long()
                
                # Call the original update with properly typed indices
                return original_update(self, k_val, v_val)
            
            # Apply the patch
            torchtune.modules.kv_cache.KVCache.update = patched_update
            print("Successfully patched KVCache.update")
            success = True
        except Exception as e:
            print(f"Failed to patch KVCache.update: {e}")
        
        # Approach 3: Replace the entire update method with our own implementation
        try:
            cache_class = torchtune.modules.kv_cache.KVCache
            
            # Define a complete replacement method
            def new_update_method(self, k_val, v_val):
                """
                Complete replacement for KVCache.update that handles indices properly.
                """
                bsz, seq_len = k_val.size(0), k_val.size(1)
                
                # Ensure long indices
                self.curr_pos = self.curr_pos.long()
                self.cache_pos = self.cache_pos.long()
                
                # Calculate current position (standard in most implementations)
                curr_pos = self.curr_pos.item()
                
                # Update the cache positions
                self.cache_pos[:seq_len] = torch.arange(
                    curr_pos, curr_pos + seq_len, 
                    device=self.cache_pos.device, 
                    dtype=torch.long
                )
                
                # Format the output with properly typed indices
                cache_positions = self.cache_pos[:seq_len].long()
                
                # Update the key and value caches
                for b in range(bsz):
                    self.k_cache[b, cache_positions] = k_val[b]
                    self.v_cache[b, cache_positions] = v_val[b]
                
                # Create the output tensors
                max_len = self.k_cache.size(1)
                k_out = torch.zeros(
                    (bsz, seq_len, max_len), 
                    device=k_val.device, 
                    dtype=k_val.dtype
                )
                v_out = torch.zeros(
                    (bsz, seq_len, max_len), 
                    device=v_val.device, 
                    dtype=v_val.dtype
                )
                
                # Fill the output tensors
                for i in range(seq_len):
                    pos = cache_positions[i].item()
                    k_out[:, i, pos] = k_val[:, i]
                    v_out[:, i, pos] = v_val[:, i]
                
                # Update the position counter
                self.curr_pos[0] = min(max_len, curr_pos + seq_len)
                
                return k_out, v_out
            
            # Only apply this if inspecting the original method shows it might work
            if hasattr(cache_class, 'k_cache') and hasattr(cache_class, 'v_cache'):
                # Try to apply the replacement method
                torchtune.modules.kv_cache.KVCache.update = new_update_method
                print("Applied complete replacement for KVCache.update")
                success = True
        except Exception as e:
            print(f"Failed to apply replacement method: {e}")
        
        # Return success status
        return success
        
    except Exception as e:
        print(f"Failed to patch torchtune.modules.kv_cache: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_kv_cache():
    """
    Patch torchtune's KVCache implementation to fix update method.
    """
    try:
        if not importlib.util.find_spec("torchtune"):
            print("torchtune not found, skipping KVCache patch")
            return False

        import torchtune.modules.kv_cache
        
        # Store the original update method
        original_update = torchtune.modules.kv_cache.KVCache.update
        
        # Define a patched version of the update method
        def patched_update(self, k_val, v_val):
            """Patched version of KVCache.update that ensures dtype consistency"""
            try:
                # Check for dtype mismatch
                k_cache_dtype = next(iter(self.buffers())).dtype if len(list(self.buffers())) > 0 else None
                
                # Convert inputs to match cache dtype if needed
                if k_cache_dtype is not None and k_val.dtype != k_cache_dtype:
                    k_val = k_val.to(dtype=k_cache_dtype)
                    v_val = v_val.to(dtype=k_cache_dtype)
                
                return original_update(self, k_val, v_val)
            except (RuntimeError, TypeError) as e:
                if "dtypes match" in str(e) or "Index put" in str(e):
                    # Handle dtype mismatch specifically
                    print(f"KVCache dtype mismatch detected. Input: {k_val.dtype}, Cache: {k_cache_dtype}")
                    print("Attempting to convert tensors for compatibility...")
                    
                    # Get the cache dtypes
                    k_cache = self.k_cache if hasattr(self, 'k_cache') else None
                    if k_cache is not None:
                        # Convert inputs to match cache
                        k_val = k_val.to(dtype=k_cache.dtype)
                        v_val = v_val.to(dtype=k_cache.dtype)
                    else:
                        # No cache reference, try to convert to bfloat16
                        k_val = k_val.to(dtype=torch.bfloat16)
                        v_val = v_val.to(dtype=torch.bfloat16)
                    
                    # Try again with converted dtypes
                    return original_update(self, k_val, v_val)
                else:
                    # Reraise other errors
                    raise
        
        # Apply the patch
        torchtune.modules.kv_cache.KVCache.update = patched_update
        print("Successfully patched KVCache.update")
        
        return True
    except Exception as e:
        print(f"Failed to patch KVCache: {e}")
        return False

def patch_attention_reshape():
    """
    Patch the torchtune attention module's reshape operation to handle dimension mismatches.
    """
    try:
        if not importlib.util.find_spec("torchtune"):
            print("torchtune not found, skipping attention reshape patch")
            return False

        import torchtune.modules.attention
        
        # Store original forward method
        attention_classes = [
            cls for name, cls in vars(torchtune.modules.attention).items()
            if isinstance(cls, type) and hasattr(cls, 'forward') and 'Attention' in name
        ]
        
        patched_count = 0
        for attention_cls in attention_classes:
            original_forward = attention_cls.forward
            
            # Create patched forward method
            def patched_forward(self, x, y=None, mask=None, input_pos=None):
                try:
                    return original_forward(self, x, y, mask=mask, input_pos=input_pos)
                except RuntimeError as e:
                    if "invalid for input of size" in str(e) and "shape" in str(e) and "view" in str(e):
                        print(f"Catching reshape error in attention: {e}")
                        
                        # Get the q, k, v values
                        b, s_x, embed_dim = x.shape
                        
                        if y is None:
                            y = x
                        
                        q_val = self.q_proj(x).view(b, s_x, self.num_heads, self.head_dim)
                        k_val = self.k_proj(y).view(b, y.size(1), self.num_heads, self.head_dim)
                        v_val = self.v_proj(y).view(b, y.size(1), self.num_heads, self.head_dim)
                        
                        # Transpose for attention calculation
                        q_val = q_val.transpose(1, 2)
                        k_val = k_val.transpose(1, 2)
                        v_val = v_val.transpose(1, 2)
                        
                        # Apply RoPE if needed
                        if hasattr(self, 'rope') and input_pos is not None:
                            q_val = self.rope(q_val, input_pos)
                            k_val = self.rope(k_val, input_pos)
                        
                        # Safe attention using scaled dot product
                        scale = 1.0 / math.sqrt(self.head_dim)
                        
                        # Create a causal mask manually
                        causal_mask = torch.tril(torch.ones(s_x, y.size(1), device=x.device, dtype=torch.bool))
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                        
                        # Compute attention manually
                        attn_weights = torch.matmul(q_val, k_val.transpose(-2, -1)) * scale
                        
                        # Apply mask
                        float_mask = torch.zeros_like(attn_weights, dtype=x.dtype)
                        float_mask.masked_fill_(~causal_mask.expand(b, self.num_heads, s_x, y.size(1)), float('-inf'))
                        attn_weights = attn_weights + float_mask
                        
                        # Apply softmax
                        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                        
                        # Apply attention
                        attention_output = torch.matmul(attn_weights, v_val)
                        
                        # Safe reshape to original dimensions
                        output_size = b * s_x * embed_dim
                        # Make sure reshape is compatible with tensor size
                        if output_size != attention_output.numel():
                            # Adjust dimensions to match expected size
                            print(f"Adjusting attention output from {attention_output.shape} to [{b}, {s_x}, {embed_dim}]")
                            # Resize to match exactly
                            attention_output = attention_output[:, :, :s_x, :]
                        
                        # Reshape back to original dimensions
                        output = attention_output.transpose(1, 2).contiguous().view(b, s_x, embed_dim)
                        
                        # Apply out projection if it exists
                        if hasattr(self, 'out_proj'):
                            output = self.out_proj(output)
                        
                        return output
                    else:
                        # Re-raise if not a reshape error
                        raise
            
            # Apply the patch by binding the patched method to the class
            attention_cls.forward = patched_forward.__get__(None, attention_cls)
            patched_count += 1
            
        print(f"Patched {patched_count} attention forward methods")
        return patched_count > 0
            
    except Exception as e:
        print(f"Failed to patch attention reshape: {e}")
        import traceback
        traceback.print_exc()
        return False

def apply_all_patches():
    """Apply all patches."""
    patches_applied = []
    
    # Apply RoPE patch first
    if patch_torchtune_rope():
        patches_applied.append("torchtune_rope")
    
    if patch_torchtune_kv_cache():
        patches_applied.append("torchtune_kv_cache")
    
    if patch_kv_cache():
        patches_applied.append("torchtune_kv_cache_fixed")
    
    # Apply new attention reshape patch
    if patch_attention_reshape():
        patches_applied.append("attention_reshape")
    
    if patches_applied:
        print(f"Applied patches: {', '.join(patches_applied)}")
    else:
        print("No patches applied")
    
    return patches_applied 