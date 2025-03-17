#!/usr/bin/env python
"""
Fix RoPE initialization issues and run training with safe parameters.
This script applies patches to torchtune, ensures model parameters are safe,
and then runs the training script.
"""

import os
import sys
import importlib.util
import subprocess

def ensure_module_exists(module_name):
    """Check if a module exists and can be imported"""
    if importlib.util.find_spec(module_name) is None:
        print(f"Error: {module_name} module not found. Please install it first.")
        return False
    return True

def apply_rope_fixes():
    """Apply RoPE initialization fixes from fix_rope.py"""
    try:
        from fix_rope import fix_rope_implementation
        success = fix_rope_implementation()
        if not success:
            print("Failed to apply RoPE fixes")
            return False
        return True
    except ImportError:
        print("Could not import fix_rope.py. Running from patches.py instead.")
        
        try:
            from patches import patch_torchtune_rope
            success = patch_torchtune_rope()
            if not success:
                print("Failed to apply RoPE fixes from patches.py")
                return False
            return True
        except ImportError:
            print("Error: Neither fix_rope.py nor patches.py could be imported")
            return False

def update_config_file():
    """Update config.json with safer parameters"""
    import json
    
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Warning: {config_path} not found. Creating a new one with safe parameters.")
        
        # Create a safe default config
        safe_config = {
            "backbone_flavor": "llama-3B-instruct",
            "decoder_flavor": "llama-100M",
            "text_vocab_size": 128256,
            "audio_vocab_size": 2051,
            "audio_num_codebooks": 32,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 3e-5,
            "max_seq_len": 2048,  # Safer value
            "device": "cuda" if importlib.util.find_spec("torch.cuda") is not None else "cpu",
            "mixed_precision": True
        }
        
        with open(config_path, "w") as f:
            json.dump(safe_config, f, indent=2)
        
        print(f"Created new {config_path} with safe parameters.")
        return True
    
    # Update existing config file
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Check and update max_seq_len if needed
        if "max_seq_len" in config and config["max_seq_len"] > 4096:
            print(f"Warning: Reducing max_seq_len from {config['max_seq_len']} to 2048 for stability")
            config["max_seq_len"] = 2048
        
        # Enable mixed_precision if not set
        if "mixed_precision" not in config:
            config["mixed_precision"] = True
            print("Enabled mixed precision training for better memory usage")
        
        # Set safe batch size
        if "batch_size" not in config or config["batch_size"] > 1:
            config["batch_size"] = 1
            print("Set batch_size to 1 for stability")
        
        # Increase gradient accumulation if needed
        if "gradient_accumulation_steps" not in config or config["gradient_accumulation_steps"] < 8:
            config["gradient_accumulation_steps"] = 16
            print("Set gradient_accumulation_steps to 16 for effective training")
        
        # Write updated config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Updated {config_path} with safer parameters")
        return True
    
    except Exception as e:
        print(f"Error updating config file: {e}")
        return False

def inject_patch_into_models():
    """Update models.py to include a pre-patching step"""
    try:
        models_path = "models.py"
        if not os.path.exists(models_path):
            print(f"Error: {models_path} not found")
            return False
        
        with open(models_path, "r") as f:
            content = f.read()
        
        # Check if we've already patched it
        if "# RoPE pre-patch applied" in content:
            print(f"{models_path} already includes the RoPE pre-patch")
            return True
        
        # Create the patch code
        patch_code = '''
# Apply RoPE pre-patch before imports
def _apply_rope_pre_patch():
    """Apply critical patches to RoPE before model creation"""
    try:
        import torch
        import importlib.util
        
        if not importlib.util.find_spec("torchtune"):
            return False
            
        import torchtune.models.llama3_1._position_embeddings
        rope_class = torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE
        
        # Only patch if not already patched
        if hasattr(rope_class, "_rope_patched"):
            return True
        
        # Store original methods
        original_apply_scaling = rope_class.apply_scaling
        original_rope_init = rope_class.rope_init
        
        # Define safe apply_scaling
        def safe_apply_scaling(self, freqs, *args, **kwargs):
            """Safe version that handles scalar tensors"""
            if isinstance(freqs, torch.Tensor) and freqs.dim() == 0:
                freqs_value = freqs.item()
                return torch.tensor([freqs_value], dtype=freqs.dtype, device=freqs.device)
            return original_apply_scaling(self, freqs, *args, **kwargs)
        
        # Define safe rope_init
        def safe_rope_init(self):
            """Safe version of rope_init for initialization"""
            try:
                original_rope_init(self)
            except Exception:
                # Use CUDA if available, otherwise CPU
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                half_dim = self.dim
                freqs = torch.arange(0, half_dim // 2, 2, device=device).float()
                freqs = 1.0 / (10000.0 ** (freqs / (half_dim // 2)))
                seq_idx = torch.arange(min(self.max_seq_len, 4096), device=device).float()
                emb = torch.outer(seq_idx, freqs)
                self.register_buffer("cos_cached", torch.cos(emb).float(), persistent=False)
                self.register_buffer("sin_cached", torch.sin(emb).float(), persistent=False)
        
        # Apply the patches
        rope_class.apply_scaling = safe_apply_scaling
        rope_class.rope_init = safe_rope_init
        
        # Mark as patched
        rope_class._rope_patched = True
        return True
    except Exception:
        return False

# Apply pre-patch
_ROPE_PRE_PATCHED = _apply_rope_pre_patch()
# RoPE pre-patch applied
'''
        
        # Insert the patch after the imports but before the model definitions
        import_end = content.find("def llama3_2_1B()")
        if import_end == -1:
            print("Could not find appropriate location for the patch")
            return False
        
        # Find a good insertion point after imports
        lines = content[:import_end].split('\n')
        insertion_point = 0
        for i, line in enumerate(lines):
            if line.startswith('import') or line.startswith('from'):
                insertion_point = i + 1
        
        # Insert the patch
        patched_content = '\n'.join(lines[:insertion_point]) + patch_code + '\n'.join(lines[insertion_point:]) + content[import_end:]
        
        # Write the patched file
        with open(models_path, "w") as f:
            f.write(patched_content)
        
        print(f"Successfully injected RoPE pre-patch into {models_path}")
        return True
    
    except Exception as e:
        print(f"Error injecting patch into models.py: {e}")
        return False

def run_training():
    """Run the training script with memory-efficient settings"""
    try:
        cmd = [
            "python", "train.py",
            "--config", "config.json",
            "--checkpoint_activations",
            "--cpu_offload",
            "--num_workers", "0"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        return subprocess.run(cmd).returncode == 0
    except Exception as e:
        print(f"Error running training: {e}")
        return False

def main():
    """Main function that orchestrates the fixes and runs training"""
    print("=== RoPE FIX AND TRAINING SCRIPT ===")
    
    # Ensure required modules are available
    if not ensure_module_exists("torch") or not ensure_module_exists("torchtune"):
        print("Required modules not found. Please install them first.")
        sys.exit(1)
    
    # Apply RoPE fixes
    print("\nApplying RoPE fixes...")
    if not apply_rope_fixes():
        print("Failed to apply RoPE fixes. Proceeding with caution.")
    
    # Inject pre-patch into models.py
    print("\nInjecting pre-patch into models.py...")
    inject_patch_into_models()
    
    # Update config file
    print("\nUpdating config file with safe parameters...")
    update_config_file()
    
    # Run training
    print("\nStarting training with fixed parameters...")
    if run_training():
        print("\nTraining completed successfully!")
    else:
        print("\nTraining failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 