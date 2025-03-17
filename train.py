import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import Model, ModelArgs
from generator import Segment, load_llama3_tokenizer
from moshi.models import loaders
from huggingface_hub import hf_hub_download


@dataclass
class TrainingArgs:
    dataset_dir: str
    output_dir: str
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 100
    audio_num_codebooks: int = 32
    audio_vocab_size: int = 2051
    text_vocab_size: int = 128256
    backbone_flavor: str = "llama-1B"
    decoder_flavor: str = "llama-100M"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    from_checkpoint: Optional[str] = None


class SwitchboardDataset(Dataset):
    def __init__(self, metadata_path: str, tokenizer=None):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.tokenizer = tokenizer if tokenizer else load_llama3_tokenizer()
        self.sample_rate = 24000
        # Initialize audio tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=device)
        self.mimi.set_num_codebooks(32)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load audio
        audio_path = item["audio_path"]
        # Handle relative paths
        if not os.path.isabs(audio_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            audio_path = os.path.join(base_dir, audio_path)
        
        audio, sr = torchaudio.load(audio_path)
        audio = audio.mean(dim=0)  # Convert to mono if needed
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)
        
        # Tokenize text
        text = item["transcript"]
        speaker = item["speaker"]
        
        # Create segment
        segment = Segment(
            speaker=speaker,
            text=text,
            audio=audio
        )
        
        return segment


def collate_fn(batch):
    # Simple collate function that just returns the list of segments
    # The trainer will handle tokenization
    return batch


class Trainer:
    def __init__(self, args: TrainingArgs):
        self.args = args
        self.device = torch.device(args.device)
        
        # Set random seed for reproducibility
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # Initialize model
        model_args = ModelArgs(
            backbone_flavor=args.backbone_flavor,
            decoder_flavor=args.decoder_flavor,
            text_vocab_size=args.text_vocab_size,
            audio_vocab_size=args.audio_vocab_size,
            audio_num_codebooks=args.audio_num_codebooks,
        )
        self.model = Model(model_args).to(device=self.device, dtype=torch.bfloat16)
        
        # Initialize tokenizers
        self.text_tokenizer = load_llama3_tokenizer()
        
        # Initialize audio tokenizer
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(args.audio_num_codebooks)
        
        # Initialize datasets and dataloaders
        train_dataset = SwitchboardDataset(
            os.path.join(args.dataset_dir, "train", "metadata.json"),
            tokenizer=self.text_tokenizer
        )
        val_dataset = SwitchboardDataset(
            os.path.join(args.dataset_dir, "validation", "metadata.json"),
            tokenizer=self.text_tokenizer
        )
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Initialize scheduler
        total_steps = len(self.train_dataloader) * args.num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        # Load from checkpoint if provided
        self.global_step = 0
        if args.from_checkpoint:
            self._load_checkpoint(args.from_checkpoint)
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
    
    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a text segment."""
        text_tokens = self.text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        
        return text_frame.to(self.device), text_frame_mask.to(self.device)
    
    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize an audio segment."""
        audio = audio.to(self.device)
        audio_tokens = self.mimi.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True
        
        return audio_frame, audio_frame_mask
    
    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a full segment."""
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)
    
    def _prepare_batch(self, batch: List[Segment]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare a batch for training."""
        # Tokenize each segment
        tokens_list = []
        masks_list = []
        
        for segment in batch:
            tokens, masks = self._tokenize_segment(segment)
            tokens_list.append(tokens)
            masks_list.append(masks)
        
        # Get maximum sequence length
        max_len = max(tokens.size(0) for tokens in tokens_list)
        
        # Pad sequences to max length
        padded_tokens = []
        padded_masks = []
        for tokens, masks in zip(tokens_list, masks_list):
            padding_len = max_len - tokens.size(0)
            if padding_len > 0:
                pad_tokens = torch.zeros(padding_len, 33).long().to(self.device)
                pad_masks = torch.zeros(padding_len, 33).bool().to(self.device)
                
                tokens = torch.cat([tokens, pad_tokens], dim=0)
                masks = torch.cat([masks, pad_masks], dim=0)
            
            padded_tokens.append(tokens)
            padded_masks.append(masks)
        
        # Stack into batches
        tokens_batch = torch.stack(padded_tokens, dim=0)
        masks_batch = torch.stack(padded_masks, dim=0)
        
        # Create position tensor
        positions = torch.arange(0, max_len).unsqueeze(0).repeat(len(batch), 1).to(self.device)
        
        return tokens_batch, masks_batch, positions
    
    def _compute_loss(
        self, 
        tokens: torch.Tensor, 
        tokens_mask: torch.Tensor, 
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for a batch."""
        batch_size, seq_len, _ = tokens.size()
        
        # Create causal mask
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device)
        )
        batch_causal_mask = causal_mask[positions]
        
        # We only want to predict audio tokens, not text tokens
        # Find where audio tokens start (after text tokens)
        audio_start_positions = []
        for i in range(batch_size):
            # Text tokens are in the last column (index -1)
            text_mask = tokens_mask[i, :, -1]
            # Find the first position where text mask is False (audio starts)
            audio_start = torch.nonzero(~text_mask, as_tuple=True)[0]
            if len(audio_start) > 0:
                audio_start_positions.append(audio_start[0])
            else:
                # If no audio tokens, use the end of sequence
                audio_start_positions.append(seq_len)
        
        # Embed tokens
        embeds = self.model._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        
        # Get backbone outputs
        backbone_outputs = self.model.backbone(h, input_pos=positions, mask=batch_causal_mask)
        
        # Initialize loss
        loss = 0.0
        
        # For each item in batch
        for i in range(batch_size):
            audio_start = audio_start_positions[i]
            if audio_start >= seq_len - 1:
                # Skip if there's no audio to predict
                continue
            
            # Get the backbone output just before audio starts
            last_h = backbone_outputs[i, audio_start-1:audio_start, :]
            
            # Compute loss for first codebook
            c0_logits = self.model.codebook0_head(last_h)
            c0_target = tokens[i, audio_start, 0]  # First codebook token
            c0_loss = nn.functional.cross_entropy(
                c0_logits.squeeze(0), 
                c0_target.unsqueeze(0)
            )
            loss += c0_loss
            
            # Extract the audio codebook targets
            audio_targets = tokens[i, audio_start:, :-1]  # All codebooks
            
            # Process each position in the audio sequence
            for pos in range(audio_start, min(audio_start + 50, seq_len)):
                if pos == audio_start:
                    # Already handled the first codebook (c0)
                    continue
                    
                # Get current position's features
                curr_pos_tensor = torch.tensor([pos-1], device=self.device).unsqueeze(0)
                prev_h = backbone_outputs[i, pos-1:pos, :]
                
                # Project to decoder dimensions
                decoder_input = self.model.projection(prev_h)
                
                # Get decoder output for current position
                decoder_mask = causal_mask[:1, :1]  # 1x1 mask
                decoder_output = self.model.decoder(
                    decoder_input, 
                    input_pos=curr_pos_tensor, 
                    mask=decoder_mask
                )
                
                # For each codebook (except first which was handled above)
                for j in range(1, self.args.audio_num_codebooks):
                    # Get logits for current codebook
                    cj_logits = torch.mm(decoder_output.squeeze(0), self.model.audio_head[j-1])
                    
                    # Get target for current codebook
                    if pos < seq_len:
                        cj_target = tokens[i, pos, j]
                        # Compute loss
                        cj_loss = nn.functional.cross_entropy(
                            cj_logits, 
                            cj_target.unsqueeze(0)
                        )
                        loss += cj_loss
        
        # Normalize by batch size
        loss = loss / batch_size
        return loss
    
    def train_step(self, batch: List[Segment]) -> float:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Prepare batch
        tokens, masks, positions = self._prepare_batch(batch)
        
        # Compute loss
        loss = self._compute_loss(tokens, masks, positions)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_dataloader, desc="Validating"):
            # Prepare batch
            tokens, masks, positions = self._prepare_batch(batch)
            
            # Compute loss
            loss = self._compute_loss(tokens, masks, positions)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def save_checkpoint(self, name: str = "model") -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "args": self.args,
        }
        
        checkpoint_path = os.path.join(self.args.output_dir, f"{name}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = checkpoint["global_step"]
        
        print(f"Loaded checkpoint from {checkpoint_path} (global_step: {self.global_step})")
    
    def train(self) -> None:
        """Train the model."""
        print("Starting training...")
        best_val_loss = float("inf")
        
        for epoch in range(self.args.num_epochs):
            print(f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc="Training")):
                # Train step
                loss = self.train_step(batch)
                self.global_step += 1
                
                # Logging
                if self.global_step % self.args.log_every == 0:
                    print(f"Step {self.global_step}: loss = {loss:.4f}, lr = {self.scheduler.get_last_lr()[0]:.6f}")
                
                # Evaluation
                if self.global_step % self.args.eval_every == 0:
                    val_loss = self.validate()
                    print(f"Validation loss: {val_loss:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint("model_best")
                
                # Checkpointing
                if self.global_step % self.args.save_every == 0:
                    self.save_checkpoint(f"model_step_{self.global_step}")
            
            # Save at the end of each epoch
            self.save_checkpoint(f"model_epoch_{epoch+1}")
        
        print("Training completed.")


def main():
    parser = argparse.ArgumentParser(description="Train CSM-3B on Switchboard dataset")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing preprocessed data")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to train on (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Convert args to TrainingArgs
    training_args = TrainingArgs(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        from_checkpoint=args.from_checkpoint,
    )
    
    # Initialize trainer
    trainer = Trainer(training_args)
    
    # Train model
    trainer.train()


if __name__ == "__main__":
    main() 