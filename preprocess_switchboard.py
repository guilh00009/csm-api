import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def download_switchboard(output_dir: str) -> str:
    """
    Download the Switchboard dataset from Hugging Face.
    
    Args:
        output_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded dataset
    """
    print("Downloading Switchboard dataset from Hugging Face (hhoangphuoc/switchboard)...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset
    dataset = load_dataset("hhoangphuoc/switchboard")
    
    # Process and save each split
    for split in ["train", "validation", "test"]:
        if split in dataset:
            split_dir = os.path.join(output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            os.makedirs(os.path.join(split_dir, "audio"), exist_ok=True)
            
            print(f"Processing {split} split...")
            samples = []
            
            for i, item in enumerate(tqdm(dataset[split])):
                # Extract audio
                audio = torch.tensor(item["audio"]["array"])
                sample_rate = item["audio"]["sampling_rate"]
                
                # Save audio file
                audio_filename = f"{split}_{i:05d}.wav"
                audio_path = os.path.join(split_dir, "audio", audio_filename)
                torchaudio.save(audio_path, audio.unsqueeze(0), sample_rate)
                
                # Extract transcript and metadata
                transcript = item["transcript"]
                
                # Determine speaker (assuming it's in the metadata or can be derived)
                # For Switchboard, we'll use a simple heuristic based on the index
                # In a real implementation, you would extract this from the dataset properly
                speaker = 0 if i % 2 == 0 else 1
                
                # Create sample metadata
                sample_metadata = {
                    "audio_path": audio_path,
                    "transcript": transcript,
                    "speaker": speaker,
                    "duration": len(audio) / sample_rate,
                    "original_file": f"hhoangphuoc/switchboard/{split}/{i}"
                }
                
                samples.append(sample_metadata)
            
            # Save metadata
            with open(os.path.join(split_dir, "metadata.json"), 'w') as f:
                json.dump(samples, f, indent=2)
            
            print(f"Saved {len(samples)} samples for {split} split")
    
    print(f"Dataset downloaded and processed to {output_dir}")
    return output_dir


def parse_transcript(transcript_path: str) -> List[Dict]:
    """
    Parse a Switchboard transcript file and extract utterances with speaker information.
    
    Args:
        transcript_path: Path to the transcript file
        
    Returns:
        List of dictionaries with speaker and text information
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract utterances with speaker information
    # Format is typically: speaker_id: utterance
    utterances = []
    
    # This regex pattern may need adjustment based on the exact format of your transcripts
    pattern = r'([A-Z]):\s*(.*?)(?=\n[A-Z]:|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for speaker, text in matches:
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Skip empty utterances
        if not text:
            continue
            
        # Map speaker ID to numeric value (A -> 0, B -> 1, etc.)
        speaker_id = ord(speaker) - ord('A')
        
        utterances.append({
            "speaker": speaker_id,
            "text": text
        })
    
    return utterances


def process_audio_file(audio_path: str, target_sr: int = 24000) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Process an audio file and its corresponding transcript.
    
    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_tensor, utterances)
    """
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
    
    # Find corresponding transcript
    transcript_path = audio_path.replace('.wav', '.txt')
    if not os.path.exists(transcript_path):
        # Try alternative extensions
        for ext in ['.trs', '.trn']:
            alt_path = audio_path.replace('.wav', ext)
            if os.path.exists(alt_path):
                transcript_path = alt_path
                break
    
    if not os.path.exists(transcript_path):
        print(f"Warning: No transcript found for {audio_path}")
        return audio.squeeze(0), []
    
    # Parse transcript
    utterances = parse_transcript(transcript_path)
    
    return audio.squeeze(0), utterances


def segment_audio(audio: torch.Tensor, utterances: List[Dict], 
                  audio_path: str, output_dir: str, sample_rate: int = 24000) -> List[Dict]:
    """
    Segment audio based on utterances and save to output directory.
    
    Args:
        audio: Audio tensor
        utterances: List of utterance dictionaries
        audio_path: Original audio path (for naming)
        output_dir: Output directory
        sample_rate: Audio sample rate
        
    Returns:
        List of processed samples with metadata
    """
    # Create output directories
    audio_output_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # For simplicity, we'll divide the audio equally among utterances
    # In a real implementation, you would use timestamps from the transcript
    audio_len = audio.size(0)
    segment_len = audio_len // len(utterances)
    
    processed_samples = []
    
    for i, utterance in enumerate(utterances):
        # Calculate segment boundaries
        start = i * segment_len
        end = (i + 1) * segment_len if i < len(utterances) - 1 else audio_len
        
        # Extract segment
        segment = audio[start:end]
        
        # Skip very short segments
        if segment.size(0) < 0.1 * sample_rate:  # Less than 100ms
            continue
        
        # Save segment
        segment_filename = f"{base_name}_segment_{i:03d}.wav"
        segment_path = os.path.join(audio_output_dir, segment_filename)
        torchaudio.save(segment_path, segment.unsqueeze(0), sample_rate)
        
        # Create metadata
        sample_metadata = {
            "audio_path": segment_path,
            "transcript": utterance["text"],
            "speaker": utterance["speaker"],
            "duration": segment.size(0) / sample_rate,
            "original_file": audio_path
        }
        
        processed_samples.append(sample_metadata)
    
    return processed_samples


def process_switchboard(input_dir: str, output_dir: str, sample_rate: int = 24000) -> None:
    """
    Process the Switchboard dataset.
    
    Args:
        input_dir: Input directory containing Switchboard data
        output_dir: Output directory for processed data
        sample_rate: Target sample rate
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all audio files
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each audio file
    all_samples = []
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        audio, utterances = process_audio_file(audio_path, sample_rate)
        if not utterances:
            continue
            
        samples = segment_audio(audio, utterances, audio_path, output_dir, sample_rate)
        all_samples.extend(samples)
    
    print(f"Processed {len(all_samples)} segments")
    
    # Split into train/validation/test
    # Shuffle samples first for better distribution
    import random
    random.shuffle(all_samples)
    
    # 80% train, 10% validation, 10% test
    train_size = int(0.8 * len(all_samples))
    val_size = int(0.1 * len(all_samples))
    
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size + val_size]
    test_samples = all_samples[train_size + val_size:]
    
    # Create split directories
    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Save metadata
    with open(os.path.join(output_dir, "train", "metadata.json"), 'w') as f:
        json.dump(train_samples, f, indent=2)
    
    with open(os.path.join(output_dir, "validation", "metadata.json"), 'w') as f:
        json.dump(val_samples, f, indent=2)
    
    with open(os.path.join(output_dir, "test", "metadata.json"), 'w') as f:
        json.dump(test_samples, f, indent=2)
    
    print(f"Split into {len(train_samples)} train, {len(val_samples)} validation, {len(test_samples)} test samples")
    print(f"Processed data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Switchboard dataset for CSM-3B training")
    parser.add_argument("--input_dir", type=str, default=None, 
                        help="Input directory containing Switchboard data (if not downloading from HF)")
    parser.add_argument("--output_dir", type=str, default="switchboard", help="Output directory for processed data")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate")
    parser.add_argument("--download", action="store_true", help="Download dataset from Hugging Face")
    
    args = parser.parse_args()
    
    if args.download or args.input_dir is None:
        # Download dataset from Hugging Face
        download_switchboard(args.output_dir)
    else:
        # Process local dataset
        process_switchboard(args.input_dir, args.output_dir, args.sample_rate)


if __name__ == "__main__":
    main() 