"""
Monkey patches for third-party libraries to fix compatibility issues.
These patches are applied at runtime.
"""

import torch
import importlib.util
import types
import inspect

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

def apply_all_patches():
    """Apply all patches."""
    patches_applied = []
    
    if patch_torchtune_kv_cache():
        patches_applied.append("torchtune_kv_cache")
    
    if patches_applied:
        print(f"Applied patches: {', '.join(patches_applied)}")
    else:
        print("No patches applied")
    
    return patches_applied 