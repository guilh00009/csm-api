#!/usr/bin/env python
"""
Script to apply fixes and run the training process with the 1B model.
"""

import os
import json
import shutil
import sys
import subprocess

def update_config_file():
    """Update config.json with the 1B model parameters."""
    if not os.path.exists("config.json"):
        config = {
            "backbone_flavor": "llama-1B",
            "decoder_flavor": "llama-100M",
            "text_vocab_size": 128256,
            "audio_vocab_size": 2051,
            "audio_num_codebooks": 32,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 3e-5,
            "weight_decay": 0.01,
            "num_epochs": 10,
            "warmup_steps": 1000,
            "max_grad_norm": 1.0,
            "max_seq_len": 2048,
            "mixed_precision": True,
            "device": "cuda"
        }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
        print("Created config.json with 1B model parameters")
    else:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        # Update config with 1B model parameters
        config["backbone_flavor"] = "llama-1B"
        config["max_seq_len"] = 2048
        config["mixed_precision"] = True
        
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
        print("Updated config.json with 1B model parameters")
    
    return config

def apply_model_fixes():
    """Apply fixes to the models.py to use 1B model parameters."""
    print("Applying RoPE fixes...")
    
    # Check if models.py already includes RoPE fixes
    models_file = "models.py"
    if os.path.exists(models_file):
        with open(models_file, "r") as f:
            content = f.read()
        
        if "rope_base=500_000" in content and "scale_factor=32" in content:
            print("models.py already includes the RoPE parameters")
        else:
            print("Updating models.py with 1B parameters...")
            # Update the rope_base and scale_factor in the file
            content = content.replace("rope_base=10000.0", "rope_base=500_000")
            content = content.replace("scale_factor=1.0", "scale_factor=32")
            
            # Remove the 3B-instruct model if it exists
            if "llama3_2_3B_instruct" in content:
                # Simple approach - just ensure the FLAVORS dict excludes it
                if "llama-3B-instruct" in content:
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if "\"llama-3B-instruct\":" in line:
                            lines[i] = ""  # Remove the line
                    content = "\n".join(lines)
            
            with open(models_file, "w") as f:
                f.write(content)
            print("Updated models.py with 1B parameters")
    
    # Force disable KV cache
    patches_file = "patches.py"
    if os.path.exists(patches_file):
        print("Updating patches.py to disable KV cache")
        with open(patches_file, "a") as f:
            f.write("\n\n# Force disable KV cache for stability\ndef force_disable_kv_cache():\n    global DISABLE_KV_CACHE\n    DISABLE_KV_CACHE = True\n    print('Forced KV cache disable for stability')\n\nforce_disable_kv_cache()\n")
    
    # Set environment variable to remind train.py to disable KV cache
    os.environ["DISABLE_KV_CACHE"] = "1"
    
    return True

def create_run_without_kv_script():
    """Create a simple train wrapper that runs without KV cache."""
    script_content = """#!/usr/bin/env python
import os
import sys
import torch

# Force disable KV cache
os.environ["DISABLE_KV_CACHE"] = "1"

# Now import and run the regular training
from train import DISABLE_KV_CACHE, main

# Set global flag to disable KV cache
DISABLE_KV_CACHE = True
print("Running training with KV cache disabled")

if __name__ == "__main__":
    main()
"""
    with open("train_without_kv.py", "w") as f:
        f.write(script_content)
    print("Created train_without_kv.py to run training without KV cache")

def run_training():
    """Run the training script with the 1B model parameters."""
    print("Starting training with 1B model parameters...")
    
    # First try with the wrapper script that disables KV cache
    create_run_without_kv_script()
    cmd = ["python", "train_without_kv.py", "--config", "config.json", "--checkpoint_activations", "--num_workers", "0"]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    """Main function to run the fixes and start training with the 1B model."""
    # Update config file
    config = update_config_file()
    
    # Apply model fixes
    apply_model_fixes()
    
    # Run the training
    run_training()

if __name__ == "__main__":
    main()