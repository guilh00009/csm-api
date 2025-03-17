import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models import Model, ModelArgs
from generator import load_llama3_tokenizer, Segment
from moshi.models import loaders


@dataclass
class TrainingArgs:
    # Model parameters
    backbone_flavor: str = "llama-1B"
    decoder_flavor: str = "llama-100M"
    text_vocab_size: int = 128256
    audio_vocab_size: int = 2051
    audio_num_codebooks: int = 32
    
    # Training parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Dataset parameters
    dataset_path: str = "switchboard"
    train_split: str = "train"
    val_split: str = "validation"
    max_seq_len: int = 2048
    
    # Logging and saving
    log_every: int = 100
    save_every: int = 1000
    output_dir: str = "checkpoints"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mixed_precision: bool = True


class SwitchboardDataset(Dataset):
    def __init__(
        self, 
        dataset_path: str, 
        split: str,
        tokenizer,
        audio_tokenizer,
        max_seq_len: int = 2048,
    ):
        self.dataset_path = os.path.join(dataset_path, split)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.max_seq_len = max_seq_len
        
        # Load dataset metadata
        self.samples = self._load_metadata()
        
    def _load_metadata(self) -> List[Dict]:
        """Load metadata for the dataset samples."""
        # This would typically load a JSON or CSV file with metadata
        # For Switchboard, we'd load the transcripts and audio paths
        # This is a placeholder - you'll need to adapt to your actual dataset format
        
        samples = []
        # Example structure:
        # samples.append({
        #     "audio_path": "path/to/audio.wav",
        #     "transcript": "text transcript",
        #     "speaker": 0,  # speaker ID
        # })
        
        # For demonstration, we'll assume the dataset is organized as:
        # dataset_path/
        #   - metadata.json (contains transcripts and speaker info)
        #   - audio/ (contains audio files)
        
        # In a real implementation, you would parse the actual dataset structure
        import json
        try:
            with open(os.path.join(self.dataset_path, "metadata.json"), "r") as f:
                samples = json.load(f)
        except FileNotFoundError:
            # Fallback to scanning directory structure
            audio_dir = os.path.join(self.dataset_path, "audio")
            if os.path.exists(audio_dir):
                for filename in os.listdir(audio_dir):
                    if filename.endswith(".wav"):
                        # Extract speaker and text from filename or associated text file
                        # This is just a placeholder - adapt to your dataset
                        speaker = 0  # Default speaker
                        transcript = ""  # Default empty transcript
                        
                        # Try to find associated transcript
                        transcript_path = os.path.join(
                            self.dataset_path, "transcripts", 
                            filename.replace(".wav", ".txt")
                        )
                        if os.path.exists(transcript_path):
                            with open(transcript_path, "r") as f:
                                transcript = f.read().strip()
                        
                        samples.append({
                            "audio_path": os.path.join(audio_dir, filename),
                            "transcript": transcript,
                            "speaker": speaker,
                        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load audio
        audio_path = sample["audio_path"]
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio.mean(dim=0)  # Convert to mono
        
        # Resample if needed
        if sample_rate != self.audio_tokenizer.sample_rate:
            audio = torchaudio.functional.resample(
                audio, orig_freq=sample_rate, new_freq=self.audio_tokenizer.sample_rate
            )
        
        # Tokenize text
        text = sample["transcript"]
        speaker = sample["speaker"]
        text_tokens = self.tokenizer.encode(f"[{speaker}]{text}")
        
        # Tokenize audio
        audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        
        # Create frame tokens and masks
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        
        # Add EOS frame to audio tokens
        eos_frame = torch.zeros(audio_tokens.size(0), 1)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
        
        audio_frame = torch.zeros(audio_tokens.size(1), 33).long()
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool()
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True
        
        # Combine text and audio frames
        frames = torch.cat([text_frame, audio_frame], dim=0)
        frames_mask = torch.cat([text_frame_mask, audio_frame_mask], dim=0)
        
        # Truncate if needed
        if frames.size(0) > self.max_seq_len:
            frames = frames[:self.max_seq_len]
            frames_mask = frames_mask[:self.max_seq_len]
        
        return frames, frames_mask


def collate_fn(batch):
    frames, frames_mask = zip(*batch)
    
    # Find max sequence length in batch
    max_len = max(f.size(0) for f in frames)
    
    # Pad sequences to max length
    padded_frames = torch.zeros(len(batch), max_len, 33).long()
    padded_frames_mask = torch.zeros(len(batch), max_len, 33).bool()
    
    for i, (f, m) in enumerate(zip(frames, frames_mask)):
        padded_frames[i, :f.size(0)] = f
        padded_frames_mask[i, :m.size(0)] = m
    
    # Create position indices
    positions = torch.arange(0, max_len).unsqueeze(0).repeat(len(batch), 1)
    
    return padded_frames, padded_frames_mask, positions


def train_one_epoch(
    model: Model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: TrainingArgs,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for step, (frames, frames_mask, positions) in enumerate(progress_bar):
        frames = frames.to(args.device)
        frames_mask = frames_mask.to(args.device)
        positions = positions.to(args.device)
        
        # Forward pass with mixed precision if enabled
        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            # For training, we need to implement a loss function
            # This is a placeholder - you'll need to adapt to your model's output
            loss = model.compute_loss(frames, frames_mask, positions)
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
        
        # Backward pass with mixed precision if enabled
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights if gradient accumulation steps reached
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        total_loss += loss.item() * args.gradient_accumulation_steps
        progress_bar.set_postfix({"loss": total_loss / (step + 1)})
        
        # Log and save
        if (step + 1) % args.log_every == 0:
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {total_loss / (step + 1)}")
        
        if (step + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, args, epoch, step)
    
    return total_loss / len(dataloader)


def validate(
    model: Model,
    dataloader: DataLoader,
    args: TrainingArgs,
) -> float:
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for frames, frames_mask, positions in progress_bar:
            frames = frames.to(args.device)
            frames_mask = frames_mask.to(args.device)
            positions = positions.to(args.device)
            
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                loss = model.compute_loss(frames, frames_mask, positions)
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
    
    return total_loss / len(dataloader)


def save_checkpoint(
    model: Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: TrainingArgs,
    epoch: int,
    step: int,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": args,
    }
    
    torch.save(checkpoint, os.path.join(args.output_dir, f"checkpoint_epoch{epoch}_step{step}.pt"))
    # Save latest checkpoint
    torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint_latest.pt"))
    # Save model only
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch{epoch}_step{step}.pt"))


def load_checkpoint(
    model: Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_path: str,
) -> Tuple[Model, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int, int]:
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    
    return model, optimizer, scheduler, checkpoint["epoch"], checkpoint["step"]


def main():
    parser = argparse.ArgumentParser(description="Train CSM-1B model on Switchboard dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args_cli = parser.parse_args()
    
    # Load config from file if provided
    args = TrainingArgs()
    if args_cli.config is not None:
        import json
        with open(args_cli.config, "r") as f:
            config = json.load(f)
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
    
    # Set up model
    model_args = ModelArgs(
        backbone_flavor=args.backbone_flavor,
        decoder_flavor=args.decoder_flavor,
        text_vocab_size=args.text_vocab_size,
        audio_vocab_size=args.audio_vocab_size,
        audio_num_codebooks=args.audio_num_codebooks,
    )
    
    model = Model(model_args).to(device=args.device)
    
    # Add compute_loss method to Model class
    def compute_loss(self, frames, frames_mask, positions):
        """
        Compute the loss for training the model.
        
        Args:
            frames: (batch_size, seq_len, audio_num_codebooks+1)
            frames_mask: (batch_size, seq_len, audio_num_codebooks+1)
            positions: (batch_size, seq_len)
            
        Returns:
            loss: scalar loss value
        """
        b, s, _ = frames.size()
        
        # Embed tokens
        embeds = self._embed_tokens(frames)
        masked_embeds = embeds * frames_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        
        # Create causal mask
        device = h.device
        causal_mask = torch.tril(torch.ones(s, s, dtype=torch.bool, device=device))
        batch_mask = causal_mask[positions, :]
        
        # Forward pass through backbone
        backbone_out = self.backbone(h, input_pos=positions, mask=batch_mask)
        
        # Compute loss for text tokens (last dimension)
        text_logits = self.text_embeddings.weight @ backbone_out.transpose(1, 2)
        text_targets = frames[:, 1:, -1]  # Shift right for next token prediction
        text_loss = nn.CrossEntropyLoss()(
            text_logits[:, :-1].reshape(-1, self.args.text_vocab_size),
            text_targets.reshape(-1)
        )
        
        # Compute loss for audio tokens (first dimensions)
        audio_loss = 0
        for i in range(self.args.audio_num_codebooks):
            if i == 0:
                audio_logits = self.codebook0_head(backbone_out)
            else:
                # Project backbone output to decoder dimension
                decoder_input = self.projection(backbone_out)
                
                # Forward pass through decoder
                decoder_out = self.decoder(decoder_input, input_pos=positions, mask=batch_mask)
                
                # Compute logits for current codebook
                audio_logits = torch.einsum("bsd,dv->bsv", decoder_out, self.audio_head[i-1])
            
            # Get targets for current codebook
            audio_targets = frames[:, 1:, i]  # Shift right for next token prediction
            
            # Compute loss
            audio_loss += nn.CrossEntropyLoss()(
                audio_logits[:, :-1].reshape(-1, self.args.audio_vocab_size),
                audio_targets.reshape(-1)
            )
        
        # Combine losses
        loss = text_loss + audio_loss
        return loss
    
    # Add compute_loss method to Model class
    Model.compute_loss = compute_loss
    
    # Set up tokenizers
    text_tokenizer = load_llama3_tokenizer()
    
    # Load audio tokenizer (MIMI)
    device = next(model.parameters()).device
    mimi_weight = loaders.get_mimi_weights()
    audio_tokenizer = loaders.get_mimi(mimi_weight, device=device)
    audio_tokenizer.set_num_codebooks(args.audio_num_codebooks)
    
    # Set up datasets and dataloaders
    train_dataset = SwitchboardDataset(
        args.dataset_path,
        args.train_split,
        text_tokenizer,
        audio_tokenizer,
        args.max_seq_len,
    )
    
    val_dataset = SwitchboardDataset(
        args.dataset_path,
        args.val_split,
        text_tokenizer,
        audio_tokenizer,
        args.max_seq_len,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Set up mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args_cli.resume is not None:
        model, optimizer, scheduler, start_epoch, _ = load_checkpoint(
            model, optimizer, scheduler, args_cli.resume
        )
        print(f"Resumed from checkpoint: {args_cli.resume}")
    
    # Setup KV caches for efficient training
    model.setup_caches(args.batch_size)
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, args, epoch, scaler
        )
        
        val_loss = validate(model, val_dataloader, args)
        
        print(f"Epoch {epoch+1} completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint at the end of each epoch
        save_checkpoint(model, optimizer, scheduler, args, epoch, len(train_dataloader)-1)


if __name__ == "__main__":
    main() 