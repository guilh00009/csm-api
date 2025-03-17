"""
Monkey patches for third-party libraries to fix compatibility issues.
These patches are applied at runtime.
"""

import torch
import importlib.util

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

        # Store original update method
        original_update = torchtune.modules.kv_cache.KVCache.update
        
        # Create patched update method
        def patched_update(self, k_val, v_val):
            """
            Patched version of KVCache.update that ensures indices are always Long tensors.
            """
            bsz = k_val.size(0)
            seq_len = k_val.size(1)
            
            # Create output tensors
            k_out = torch.zeros(
                (bsz, seq_len, self.max_seq_len),
                device=k_val.device,
                dtype=k_val.dtype,
            )
            v_out = torch.zeros(
                (bsz, seq_len, self.max_seq_len),
                device=v_val.device,
                dtype=v_val.dtype,
            )
            
            # Ensure cache positions are long type
            cache_pos = self.cache_pos[:seq_len].long()
            
            # Update cache
            k_out[:, :, cache_pos] = k_val
            v_out[:, :, cache_pos] = v_val
            
            # Update current position
            old_pos = self.curr_pos.item()
            self.curr_pos[0] = min(self.max_seq_len, old_pos + seq_len)
            
            return k_out, v_out
        
        # Apply the patch
        torchtune.modules.kv_cache.KVCache.update = patched_update
        print("Successfully patched torchtune.modules.kv_cache.KVCache.update")
        return True
    
    except Exception as e:
        print(f"Failed to patch torchtune.modules.kv_cache: {e}")
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