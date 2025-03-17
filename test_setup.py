import argparse
import os
import torch
from huggingface_hub import hf_hub_download

from models import Model, ModelArgs
from generator import load_llama3_tokenizer
from moshi.models import loaders
from preprocess_switchboard import download_switchboard


def test_model_loading():
    """Test loading the model architecture."""
    print("Testing model loading...")
    
    model_args = ModelArgs(
        backbone_flavor="llama-3B-instruct",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    
    # Use CPU for testing
    device = "cpu"
    
    # Create model
    model = Model(model_args).to(device=device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created successfully with {total_params/1e6:.2f}M parameters")
    
    # Test tokenizers
    print("Testing tokenizer loading...")
    text_tokenizer = load_llama3_tokenizer()
    print(f"Text tokenizer loaded successfully. Vocab size: {text_tokenizer.vocab_size}")
    
    # Test MIMI loading
    print("Testing MIMI loading...")
    try:
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        audio_tokenizer = loaders.get_mimi(mimi_weight, device=device)
        audio_tokenizer.set_num_codebooks(32)
        print(f"MIMI loaded successfully with {audio_tokenizer.num_codebooks} codebooks")
    except Exception as e:
        print(f"Error loading MIMI: {e}")
        return False
    
    return True


def test_dataset_download():
    """Test downloading a small portion of the dataset."""
    print("Testing dataset download...")
    
    # Create a temporary directory
    temp_dir = "temp_dataset"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Download just a few samples for testing
        dataset = download_switchboard(temp_dir)
        print(f"Dataset downloaded successfully to {dataset}")
        
        # Check if metadata files exist
        for split in ["train", "validation", "test"]:
            metadata_path = os.path.join(temp_dir, split, "metadata.json")
            if os.path.exists(metadata_path):
                print(f"Found metadata for {split} split")
            else:
                print(f"Warning: No metadata found for {split} split")
        
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test CSM-3B setup")
    parser.add_argument("--test_model", action="store_true", help="Test model loading")
    parser.add_argument("--test_dataset", action="store_true", help="Test dataset download")
    parser.add_argument("--test_all", action="store_true", help="Test everything")
    
    args = parser.parse_args()
    
    if args.test_all or not (args.test_model or args.test_dataset):
        # If test_all is specified or no specific test is specified, run all tests
        model_ok = test_model_loading()
        dataset_ok = test_dataset_download()
        
        if model_ok and dataset_ok:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed. Please check the errors above.")
    else:
        # Run specific tests
        if args.test_model:
            model_ok = test_model_loading()
            if model_ok:
                print("\n✅ Model loading test passed!")
            else:
                print("\n❌ Model loading test failed.")
        
        if args.test_dataset:
            dataset_ok = test_dataset_download()
            if dataset_ok:
                print("\n✅ Dataset download test passed!")
            else:
                print("\n❌ Dataset download test failed.")


if __name__ == "__main__":
    main() 