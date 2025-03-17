import torch
import logging

logger = logging.getLogger(__name__)

def create_causal_mask(seq_len, device, format_type="4d"):
    """
    Create a causal mask with the appropriate format for transformer attention.
    
    Args:
        seq_len: Sequence length
        device: Device to create the mask on
        format_type: Format for the mask, either "2d", "3d", or "4d"
        
    Returns:
        A properly formatted causal mask for transformer attention
    """
    if format_type == "2d":
        # Simple 2D causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    elif format_type == "3d":
        # 3D mask for batched inputs
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        mask = mask.unsqueeze(0)  # [1, seq_len, seq_len]
    elif format_type == "4d":
        # 4D mask for multi-head attention
        mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device)
        mask = torch.tril(mask)  # [1, 1, seq_len, seq_len]
    else:
        raise ValueError(f"Unknown format_type: {format_type}")
    
    return mask

def ensure_tensor_dtype(tensor, target_dtype, name="tensor"):
    """
    Ensure a tensor has the expected dtype, converting if necessary.
    
    Args:
        tensor: Input tensor
        target_dtype: Target dtype
        name: Name for logging
        
    Returns:
        Tensor with the target dtype
    """
    if tensor.dtype != target_dtype and tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
        logger.debug(f"Converting {name} from {tensor.dtype} to {target_dtype}")
        return tensor.to(dtype=target_dtype)
    return tensor

def make_compatible_mask(tensor_shape, device, format_required="auto"):
    """
    Create a compatible causal mask based on the tensor shape.
    
    Args:
        tensor_shape: Shape of the tensor that needs a mask
        device: Device to create the mask on
        format_required: Format required by the model ("auto", "2d", "3d", or "4d")
        
    Returns:
        A compatible causal mask
    """
    if len(tensor_shape) < 2:
        raise ValueError(f"Tensor shape {tensor_shape} is not compatible with a causal mask")
    
    seq_len = tensor_shape[1]  # Assuming batch is first dimension
    
    if format_required == "auto":
        # Try to infer the format based on the model architecture
        if len(tensor_shape) <= 2:
            # Probably a simple RNN
            format_required = "2d"
        elif len(tensor_shape) == 3:
            # Probably a transformer without multihead
            format_required = "3d"
        else:
            # Probably a transformer with multihead
            format_required = "4d"
    
    return create_causal_mask(seq_len, device, format_required)

def try_compatible_shapes(func, *args, **kwargs):
    """
    Try to run a function with compatible shapes, attempting different mask formats if needed.
    
    Args:
        func: Function to call
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        The result of the function call
    """
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        error_msg = str(e)
        if "expanded size" in error_msg and "must match" in error_msg:
            # Handle shape mismatch
            if "mask" in kwargs:
                # Try different mask formats
                original_mask = kwargs["mask"]
                device = original_mask.device
                
                # Extract sequence length from arguments if available
                seq_len = None
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        if len(arg.shape) >= 2:
                            seq_len = arg.shape[1]
                            break
                
                if seq_len is None:
                    # Try to infer from kwargs
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor) and k != "mask":
                            if len(v.shape) >= 2:
                                seq_len = v.shape[1]
                                break
                
                if seq_len is None:
                    # Fall back to mask size
                    if len(original_mask.shape) >= 2:
                        seq_len = original_mask.shape[-1]
                    else:
                        raise ValueError("Could not determine sequence length for mask")
                
                # Try different formats
                for format_type in ["4d", "3d", "2d"]:
                    try:
                        kwargs["mask"] = create_causal_mask(seq_len, device, format_type)
                        return func(*args, **kwargs)
                    except RuntimeError as format_error:
                        if "expanded size" not in str(format_error):
                            # This is a different error, not related to mask format
                            raise
                
                # If we get here, none of the formats worked
                raise ValueError(f"Could not find a compatible mask format for {func.__name__}")
            else:
                # Shape mismatch but not related to mask
                raise
        else:
            # Other type of error
            raise 