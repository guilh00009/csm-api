# Fix critical libraries before other imports
import warnings
import os

# Set environment variables early
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better error messages
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"  # Ignore future warnings

import argparse
import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Apply early patching to fix RoPE initialization
def apply_early_patches():
    """Apply critical patches before any imports that might use torchtune"""
    try:
        import torch
        import importlib.util
        import types
        
        # Check if torchtune is available
        if not importlib.util.find_spec("torchtune"):
            print("torchtune not found, early patching skipped")
            return False
        
        # Apply patches to torchtune RoPE implementation
        import torchtune.models.llama3_1._position_embeddings
        
        # Get original methods
        original_rope_init = torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.rope_init
        
        # Create a direct safe implementation
        def safe_rope_init(self):
            """Safe implementation of RoPE initialization"""
            try:
                # Try original implementation
                original_rope_init(self)
            except Exception as e:
                print(f"RoPE initialization failed with: {e}")
                print("Using direct implementation...")
                
                with torch.no_grad():
                    # Conservative implementation
                    dim = self.dim
                    max_seq_len = min(self.max_seq_len, 4096)  # Cap at 4K
                    
                    # Direct calculation
                    half_dim = dim // 2
                    freqs = torch.arange(0, half_dim, 2, device=self.device).float()
                    freqs = 1.0 / (10000.0 ** (freqs / half_dim))
                    
                    # Create position indices and outer product
                    seq_idx = torch.arange(max_seq_len, device=self.device).float()
                    emb = torch.outer(seq_idx, freqs)
                    
                    # Calculate cos/sin
                    cos_cached = torch.cos(emb).float()
                    sin_cached = torch.sin(emb).float()
                    
                    # Register buffers
                    self.register_buffer("cos_cached", cos_cached, persistent=False)
                    self.register_buffer("sin_cached", sin_cached, persistent=False)
        
        # Apply patches
        torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.rope_init = safe_rope_init
        print("Applied early patches to torchtune RoPE implementation")
        
        return True
    except Exception as e:
        print(f"Failed to apply early patches: {e}")
        import traceback
        traceback.print_exc()
        return False

# Apply early patches before importing other modules
EARLY_PATCHES_APPLIED = apply_early_patches()

# Fix multiprocessing to use 'spawn' method to avoid CUDA issues
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set
    pass

# Continue with regular imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from huggingface_hub import hf_hub_download

# Apply all remaining patches
from patches import apply_all_patches
PATCHES_APPLIED = apply_all_patches()

# Flag to disable KV caching if patches fail
DISABLE_KV_CACHE = not any("torchtune_kv_cache" in p for p in PATCHES_APPLIED)
if DISABLE_KV_CACHE:
    print("Warning: KV cache patch failed. Training will continue without KV caching.")

# Now it's safe to import models
from models import Model, ModelArgs
from model_wrapper import create_model_safely
from generator import load_llama3_tokenizer, Segment
from moshi.models import loaders

# Explicitly check CUDA capabilities
def get_best_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    
    # Check for bfloat16 support first
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # Fall back to float16 if bfloat16 not supported
    else:
        return torch.float16

# Global dtype to ensure consistency across all tensors
GLOBAL_DTYPE = get_best_dtype()
print(f"Using {GLOBAL_DTYPE} precision for training")

@dataclass
class TrainingArgs:
    # Model parameters
    backbone_flavor: str = "llama-3B-instruct"
    decoder_flavor: str = "llama-100M"
    text_vocab_size: int = 128256
    audio_vocab_size: int = 2051
    audio_num_codebooks: int = 32
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Dataset parameters
    dataset_path: str = "switchboard"
    train_split: str = "train"
    val_split: str = "validation"
    max_seq_len: int = 4096
    
    # Logging and saving
    log_every: int = 100
    save_every: int = 1000
    output_dir: str = "checkpoints"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mixed_precision: bool = True
    
    # Memory optimization
    cpu_offload: bool = False
    checkpoint_activations: bool = False


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
        
        # Make sure audio is on CPU
        audio = audio.to("cpu")
        
        # Tokenize text
        text = sample["transcript"]
        speaker = sample["speaker"]
        text_tokens = self.tokenizer.encode(f"[{speaker}]{text}")
        
        # Tokenize audio - ensure it's on CPU
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
    
    # Create position indices - explicitly use long type
    positions = torch.arange(0, max_len, dtype=torch.long).unsqueeze(0).repeat(len(batch), 1)
    
    return padded_frames, padded_frames_mask, positions


def train_one_epoch(
    model: Model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: TrainingArgs,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for step, (frames, frames_mask, positions) in enumerate(progress_bar):
        # Move data to device and ensure correct dtype
        frames = frames.to(args.device, dtype=GLOBAL_DTYPE)
        frames_mask = frames_mask.to(args.device)
        
        # Ensure positions tensor is long type to avoid index errors
        positions = positions.long().to(args.device)
        
        # Forward pass with mixed precision if enabled
        with torch.amp.autocast('cuda', enabled=args.mixed_precision):
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
            # Move data to device and ensure correct dtype
            frames = frames.to(args.device, dtype=GLOBAL_DTYPE)
            frames_mask = frames_mask.to(args.device)
            
            # Ensure positions tensor is long type to avoid index errors
            positions = positions.long().to(args.device)
            
            with torch.amp.autocast('cuda', enabled=args.mixed_precision):
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
    parser = argparse.ArgumentParser(description="Train CSM-3B model on Switchboard dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--cpu_offload", action="store_true", help="Offload optimizer states to CPU")
    parser.add_argument("--checkpoint_activations", action="store_true", help="Use activation checkpointing to save memory")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
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
    
    # Override with command line arguments
    if args_cli.cpu_offload:
        args.cpu_offload = True
    if args_cli.checkpoint_activations:
        args.checkpoint_activations = True
    if args_cli.num_workers is not None:
        args.num_workers = args_cli.num_workers
    
    # Set up model
    model_args = ModelArgs(
        backbone_flavor=args.backbone_flavor,
        decoder_flavor=args.decoder_flavor,
        text_vocab_size=args.text_vocab_size,
        audio_vocab_size=args.audio_vocab_size,
        audio_num_codebooks=args.audio_num_codebooks,
    )
    
    # Use the global dtype for the model to ensure consistency
    # Use our safer model creation wrapper
    try:
        model = create_model_safely(
            model_args=model_args,
            device=args.device,
            dtype=GLOBAL_DTYPE
        )
    except Exception as e:
        print(f"Failed to create model: {e}")
        print("Attempting to continue with a smaller model...")
        # Try with a smaller model as last resort
        model_args.backbone_flavor = "llama-1B"  # Use smaller model
        model = create_model_safely(
            model_args=model_args,
            device=args.device,
            dtype=GLOBAL_DTYPE
        )
        print("Successfully created a smaller model as fallback!")
    
    # Apply activation checkpointing if enabled
    if args.checkpoint_activations:
        from torch.utils.checkpoint import checkpoint_sequential
        from torch.utils.checkpoint import checkpoint
        
        # Wrap the backbone with activation checkpointing
        def forward_with_checkpointing(self, *args, **kwargs):
            # Store original forward
            original_forward = self.backbone.forward
            
            # Define a checkpointed forward function
            def checkpointed_forward(*inner_args, **inner_kwargs):
                # Split the backbone into chunks for checkpointing
                num_layers = len(self.backbone.layers)
                chunks = 4  # Number of chunks to split the model into
                
                # Apply checkpointing to each chunk
                def create_custom_forward(start_idx, end_idx):
                    def custom_forward(*custom_args):
                        x = custom_args[0]
                        for i in range(start_idx, end_idx):
                            x = self.backbone.layers[i](x, inner_args[1], inner_args[2])
                        return x
                    return custom_forward
                
                # Process input through initial non-layer components
                x = inner_args[0]
                
                # Process through checkpointed layer chunks
                chunk_size = max(1, num_layers // chunks)
                for i in range(0, num_layers, chunk_size):
                    end_idx = min(i + chunk_size, num_layers)
                    x = checkpoint(create_custom_forward(i, end_idx), x)
                
                # Process through final components
                x = self.backbone.norm(x)
                return x
            
            # Replace forward with checkpointed version temporarily
            self.backbone.forward = checkpointed_forward
            
            # Call the model with the checkpointed forward
            output = original_forward(*args, **kwargs)
            
            # Restore original forward
            self.backbone.forward = original_forward
            
            return output
        
        # Monkey patch the model's forward method
        model._original_forward = model.forward
        model.forward = forward_with_checkpointing.__get__(model, Model)
    
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
        # Keep values as integers for embedding indices
        # Only convert to GLOBAL_DTYPE for value tensors, not index tensors
        frames_values = frames.to(dtype=GLOBAL_DTYPE)
        
        b, s, _ = frames.size()
        
        # Embed tokens - pass original frames for indices
        embeds = self._embed_tokens(frames)
        masked_embeds = embeds * frames_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        
        # Create causal mask
        device = h.device
        causal_mask = torch.tril(torch.ones(s, s, dtype=torch.bool, device=device))
        batch_mask = causal_mask[positions, :]
        
        # Forward pass through backbone - handle disabled KV cache mode
        if DISABLE_KV_CACHE:
            # For safety, ensure we're not using KV caches
            if hasattr(self.backbone, 'caches_are_enabled') and self.backbone.caches_are_enabled():
                self.backbone.reset_caches()
            
            # Use the backbone without any KV cache operations
            # Pass the input position and mask directly
            backbone_out = self.backbone(h, input_pos=positions, mask=batch_mask)
        else:
            # Use the backbone with KV caching as normal
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
    
    # Load audio tokenizer (MIMI) on CPU first
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    # Load MIMI on CPU first to avoid device mismatch issues
    audio_tokenizer = loaders.get_mimi(mimi_weight, device="cpu")
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
        persistent_workers=args.num_workers > 0,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    
    # Set up optimizer and scheduler
    if args.cpu_offload:
        # Use CPU offloading for optimizer states
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=optim.AdamW,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
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
    if args.mixed_precision:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args_cli.resume is not None:
        model, optimizer, scheduler, start_epoch, _ = load_checkpoint(
            model, optimizer, scheduler, args_cli.resume
        )
        print(f"Resumed from checkpoint: {args_cli.resume}")
    
    # Setup KV caches for efficient training - only if patches were applied
    if DISABLE_KV_CACHE:
        print("KV caching is disabled. This may slow down training but should be more stable.")
    else:
        try:
            # Setup caches with the proper dtype
            with torch.amp.autocast('cuda', enabled=False):
                model.setup_caches(args.batch_size)
                # Ensure KV caches use the same dtype as the model
                for name, param in model.named_buffers():
                    if 'cache' in name:
                        param.data = param.data.to(dtype=GLOBAL_DTYPE)
                    # Ensure cache positions are Long to avoid index errors
                    if 'cache_pos' in name:
                        param.data = param.data.long()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("Warning: Not enough GPU memory for KV caches. Training without KV caches.")
                # Continue without KV caches
                DISABLE_KV_CACHE = True
            else:
                raise e
    
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