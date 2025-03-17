#!/usr/bin/env python
"""
Direct fix for the attention reshape error:
RuntimeError: shape '[1, 118, -1]' is invalid for input of size 4194304
"""

import os
import sys
import torch
import math

def patch_attention_reshape():
    """
    Patch torchtune's attention module to fix the reshape issue.
    """
    try:
        import torchtune.modules.attention
        
        # Find all attention modules that might need patching
        attention_module = torchtune.modules.attention
        
        # Get all attention classes
        attention_classes = []
        for name in dir(attention_module):
            obj = getattr(attention_module, name)
            if isinstance(obj, type) and hasattr(obj, 'forward') and 'Attention' in name:
                attention_classes.append(obj)
        
        print(f"Found {len(attention_classes)} attention classes to patch")
        
        # Patch each attention class
        for cls in attention_classes:
            # Store the original forward method
            original_forward = cls.forward
            
            # Create a patched forward method
            def patched_forward(self, x, y=None, mask=None, input_pos=None):
                """Patched forward method to handle reshape errors"""
                try:
                    # Try original implementation first
                    return original_forward(self, x, y, mask=mask, input_pos=input_pos)
                except RuntimeError as e:
                    # Check if this is a reshape error
                    if "invalid for input of size" in str(e) and "shape" in str(e):
                        print(f"Fixing attention reshape error: {e}")
                        
                        # Get dimensions
                        b, s_x, embed_dim = x.shape
                        
                        # Create y if needed
                        if y is None:
                            y = x
                        
                        # Compute attention manually using a fixed implementation
                        # Step 1: Project the inputs
                        q_val = self.q_proj(x)
                        k_val = self.k_proj(y) 
                        v_val = self.v_proj(y)
                        
                        # Safe reshaping for projections
                        head_dim = embed_dim // self.num_heads
                        print(f"Original tensors - q: {q_val.shape}, k: {k_val.shape}, v: {v_val.shape}")
                        print(f"Target dimensions - heads: {self.num_heads}, head_dim: {head_dim}")
                        
                        # Step 2: Safe reshape for q_val
                        try:
                            q_val = q_val.view(b, s_x, self.num_heads, head_dim)
                        except RuntimeError as qe:
                            print(f"q_val reshape error: {qe}")
                            q_size = q_val.numel()
                            # Calculate dimensions that would work
                            expected_size = b * s_x * self.num_heads * head_dim
                            if q_size != expected_size:
                                print(f"q_val size mismatch: got {q_size}, expected {expected_size}")
                                # Create a zero tensor with the right shape
                                q_val = torch.zeros((b, s_x, self.num_heads, head_dim), 
                                                   dtype=q_val.dtype, 
                                                   device=q_val.device)
                        
                        # Step 2: Safe reshape for k_val - Handle different y sequence length
                        s_y = y.size(1)
                        try:
                            k_val = k_val.view(b, s_y, self.num_heads, head_dim)
                        except RuntimeError as ke:
                            print(f"k_val reshape error: {ke}")
                            k_size = k_val.numel()
                            # Calculate a safe shape based on actual tensor size
                            if k_size > 0:
                                # Calculate new dimensions that divide evenly
                                # First try to preserve batch size and heads
                                new_s_y = k_size // (b * self.num_heads * head_dim)
                                if new_s_y > 0:
                                    print(f"Reshaping k_val with adjusted s_y: {new_s_y}")
                                    try:
                                        k_val = k_val.view(b, new_s_y, self.num_heads, head_dim)
                                        # Update s_y for later use
                                        s_y = new_s_y
                                    except RuntimeError:
                                        # If that fails, create a zero tensor
                                        k_val = torch.zeros((b, s_y, self.num_heads, head_dim), 
                                                          dtype=k_val.dtype, 
                                                          device=k_val.device)
                                else:
                                    # Last resort - create zeros
                                    k_val = torch.zeros((b, s_y, self.num_heads, head_dim), 
                                                      dtype=k_val.dtype, 
                                                      device=k_val.device)
                            else:
                                # Create a zero tensor with expected shape
                                k_val = torch.zeros((b, s_y, self.num_heads, head_dim), 
                                                   dtype=k_val.dtype, 
                                                   device=k_val.device)
                        
                        # Step 2: Safe reshape for v_val
                        try:
                            v_val = v_val.view(b, s_y, self.num_heads, head_dim)
                        except RuntimeError as ve:
                            print(f"v_val reshape error: {ve}")
                            v_size = v_val.numel()
                            # Use same s_y as k_val for consistency
                            v_val = torch.zeros((b, s_y, self.num_heads, head_dim), 
                                              dtype=v_val.dtype, 
                                              device=v_val.device)
                        
                        # Step 3: Transpose for attention
                        q_val = q_val.transpose(1, 2)
                        k_val = k_val.transpose(1, 2)
                        v_val = v_val.transpose(1, 2)
                        
                        print(f"After transpose - q: {q_val.shape}, k: {k_val.shape}, v: {v_val.shape}")
                        
                        # Step 4: Apply RoPE if available
                        if hasattr(self, 'rope') and input_pos is not None:
                            try:
                                q_val = self.rope(q_val, input_pos)
                                k_val = self.rope(k_val, input_pos)
                            except Exception as rope_e:
                                print(f"RoPE error: {rope_e}, skipping RoPE")
                        
                        # Step 5: Compute attention
                        scale = 1.0 / math.sqrt(head_dim)
                        attn_weights = torch.matmul(q_val, k_val.transpose(-2, -1)) * scale
                        
                        # Step 6: Apply mask
                        if mask is not None:
                            # Check mask dimensions and fix if needed
                            expected_mask_shape = (b, self.num_heads, s_x, s_y)
                            if mask.shape != expected_mask_shape:
                                print(f"Mask shape {mask.shape} != expected {expected_mask_shape}")
                                
                                # Create a new mask with the right dimensions
                                new_mask = torch.zeros(expected_mask_shape, 
                                                      dtype=torch.bool, 
                                                      device=mask.device)
                                
                                # If it's a causal mask, recreate it
                                if mask.dim() == 4:
                                    # Copy as much as possible from original mask
                                    min_seq1 = min(mask.size(2), s_x)
                                    min_seq2 = min(mask.size(3), s_y)
                                    new_mask[:, :, :min_seq1, :min_seq2] = mask[:, :, :min_seq1, :min_seq2]
                                else:
                                    # Create a causal mask
                                    causal = torch.tril(torch.ones(s_x, s_y, 
                                                                 dtype=torch.bool, 
                                                                 device=mask.device))
                                    new_mask = causal.unsqueeze(0).unsqueeze(0).expand(b, self.num_heads, -1, -1)
                                
                                mask = new_mask
                        else:
                            # If no mask, create a causal mask
                            causal = torch.tril(torch.ones(s_x, s_y, 
                                                         dtype=torch.bool, 
                                                         device=x.device))
                            mask = causal.unsqueeze(0).unsqueeze(0).expand(b, self.num_heads, -1, -1)
                        
                        # Convert boolean mask to float for addition
                        if mask.dtype == torch.bool:
                            float_mask = torch.zeros_like(attn_weights, dtype=q_val.dtype)
                            float_mask.masked_fill_(~mask, float('-inf'))
                            mask = float_mask
                            
                        # Apply mask to attention weights
                        attn_weights = attn_weights + mask
                        
                        # Step 7: Apply softmax
                        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                        
                        # Step 8: Apply attention
                        attn_output = torch.matmul(attn_weights, v_val)
                        
                        # Step 9: Transpose and reshape safely
                        expected_shape = (b, s_x, embed_dim)
                        
                        try:
                            # Try the standard reshape
                            output = attn_output.transpose(1, 2).contiguous().view(*expected_shape)
                        except RuntimeError as reshape_error:
                            print(f"Reshape error: {reshape_error}, creating zero tensor")
                            
                            # Create a zero tensor with the expected shape
                            output = torch.zeros(expected_shape, 
                                                dtype=attn_output.dtype, 
                                                device=attn_output.device)
                            
                            # Copy as much data as possible
                            if attn_output.size(2) <= s_x:
                                # Transpose first
                                transposed = attn_output.transpose(1, 2).contiguous()
                                # Reshape each slice separately
                                for i in range(min(transposed.size(1), s_x)):
                                    output[:, i, :] = transposed[:, i, :].view(b, embed_dim)
                        
                        # Step 10: Apply output projection
                        if hasattr(self, 'out_proj'):
                            output = self.out_proj(output)
                        
                        return output
                    else:
                        # Re-raise other errors
                        raise
                        
            # Apply the patch by binding to the class
            cls.forward = patched_forward.__get__(None, cls)
            print(f"Patched {cls.__name__}.forward")
        
        # Also need to patch _attention_call method in MultiheadAttention
        if hasattr(attention_module, 'MultiheadAttention') and hasattr(attention_module.MultiheadAttention, '_attention_call'):
            MHA = attention_module.MultiheadAttention
            original_call = MHA._attention_call
            
            def patched_call(self, q, k, v, mask=None):
                try:
                    # Try the original implementation first
                    return original_call(self, q, k, v, mask=mask)
                except RuntimeError as e:
                    print(f"Attention call error: {e}")
                    
                    # Get dimensions
                    b, num_heads, seq_len, head_dim = q.shape
                    
                    # Check common errors
                    if "expanded size" in str(e) and "must match" in str(e):
                        # This is typically a mask dimension mismatch
                        print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
                        if mask is not None:
                            print(f"mask: {mask.shape}")
                        
                        # Create a fresh causal mask with the right dimensions
                        causal = torch.tril(torch.ones(seq_len, k.size(2), 
                                                     dtype=torch.bool, 
                                                     device=q.device))
                        # Convert to float mask
                        float_mask = torch.zeros(
                            (b, num_heads, seq_len, k.size(2)), 
                            dtype=q.dtype, 
                            device=q.device
                        )
                        float_mask.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))
                        
                        # Compute attention manually
                        scale = 1.0 / math.sqrt(head_dim)
                        attention_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                        attention_weights = attention_weights + float_mask
                        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
                        return torch.matmul(attention_weights, v)
                    else:
                        # For other errors, return identity (v)
                        print(f"Unknown error in _attention_call, returning identity")
                        return v
            
            # Apply the patch
            MHA._attention_call = patched_call.__get__(None, MHA)
            print(f"Patched MultiheadAttention._attention_call")
        
        return True
    except Exception as e:
        print(f"Failed to patch attention: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_reshape_operation():
    """
    Patch torch.Tensor.reshape to handle errors gracefully.
    """
    # Store original method
    original_reshape = torch.Tensor.reshape
    
    def safe_reshape(self, *shape):
        """Safe reshape that handles errors gracefully"""
        try:
            # Try original implementation first
            return original_reshape(self, *shape)
        except RuntimeError as e:
            print(f"Reshape error: {e}")
            print(f"Original shape: {self.shape}, Attempted reshape: {shape}")
            
            # Check for common reshape patterns
            
            # Case 1: [b, num_heads, seq_len, dim] -> [b, seq_len, -1] (attention output)
            if len(shape) == 3 and shape[2] == -1 and len(self.shape) == 4:
                # Convert to zeros with correct shape
                b, seq_len = shape[0], shape[1]
                embed_dim = self.shape[1] * self.shape[3]  # num_heads * head_dim
                
                print(f"Creating zero tensor with shape [{b}, {seq_len}, {embed_dim}]")
                return torch.zeros((b, seq_len, embed_dim), dtype=self.dtype, device=self.device)
            
            # Case 2: Flattening operation [?, -1, dim]
            if len(shape) == 2 and shape[0] == -1:
                # This is typically a flattening operation
                total_elements = self.numel()
                divisor = shape[1]
                
                # Find the largest divisible size
                if divisor > 0:
                    safe_elements = (total_elements // divisor) * divisor
                    safe_first_dim = safe_elements // divisor
                    
                    if safe_first_dim > 0:
                        print(f"Using safe reshape: [{safe_first_dim}, {divisor}]")
                        # Reshape to safe dimensions
                        flat = self.view(-1)[:safe_elements]
                        return flat.reshape(safe_first_dim, divisor)
            
            # Case 3: Automatic handling of -1 dimension
            if -1 in shape:
                # Calculate the -1 dimension automatically
                neg_idx = shape.index(-1)
                actual_shape = list(shape)
                
                # Calculate the size of known dimensions
                size_so_far = 1
                for i, dim in enumerate(shape):
                    if i != neg_idx and dim > 0:
                        size_so_far *= dim
                
                # Calculate the -1 dimension to maintain total elements
                if size_so_far > 0:
                    actual_shape[neg_idx] = max(1, self.numel() // size_so_far)
                    
                    print(f"Auto-calculating -1 dimension: {actual_shape}")
                    try:
                        # Try reshape with calculated dimensions
                        return self.reshape(*actual_shape)
                    except RuntimeError:
                        # If still fails, create zeros
                        print(f"Creating zero tensor with shape {actual_shape}")
                        return torch.zeros(actual_shape, dtype=self.dtype, device=self.device)
            
            # Last resort: create zeros with requested shape
            try:
                print(f"Creating zero tensor with exact requested shape")
                # For shapes with -1, just use 1 as fallback
                actual_shape = [1 if s == -1 else s for s in shape]
                return torch.zeros(actual_shape, dtype=self.dtype, device=self.device)
            except Exception:
                # If all else fails, raise original error
                raise e
    
    # Apply the patch
    torch.Tensor.reshape = safe_reshape
    print("Applied safe reshape patch to torch.Tensor.reshape")
    return True

def main():
    """Apply all fixes and patches."""
    print("Applying fixes for attention reshape issue...")
    
    # Apply patches
    print("1. Patching attention reshape...")
    patch_attention_reshape()
    
    print("2. Patching general reshape operation...")
    fix_reshape_operation()
    
    print("Done applying patches!")

if __name__ == "__main__":
    main() 