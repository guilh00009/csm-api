# Conversational Speech Model (CSM) Training

This repository contains code for training a Conversational Speech Model (CSM) on the Switchboard dataset. The model can generate realistic speech from text for conversational agents.

## Features

- Text-to-speech generation with speaker conditioning
- Training on conversational data
- Audio watermarking for responsible AI use
- Based on the CSM architecture with Llama 3.2 backbone

## Setup

1. Install dependencies:

```bash
pip install torch torchaudio transformers huggingface_hub tqdm silentcipher
```

2. Preprocess the Switchboard dataset:

```bash
python preprocess.py --download --output_dir ./data/switchboard
```

This will download the Switchboard dataset from Hugging Face and process it for training.

## Training

To train the model:

```bash
python train.py --dataset_dir ./data/switchboard --output_dir ./checkpoints --batch_size 4
```

### Training Arguments

- `--dataset_dir`: Directory containing preprocessed data (required)
- `--output_dir`: Output directory for checkpoints (default: "checkpoints")
- `--num_epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 4)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--from_checkpoint`: Resume training from checkpoint
- `--device`: Device to train on (cuda/cpu)

## Model Architecture

The model consists of:
- A Llama 3.2 1B backbone for context processing
- A Llama 3.2 100M decoder for audio token generation
- 32 codebooks for high-quality audio generation

## Inference

After training, you can use the model for inference:

```python
from generator import load_csm_1b, Segment

# Load model
generator = load_csm_1b("checkpoints/model_best.pt")

# Context (optional)
context = [
    Segment(speaker=0, text="Hello, how are you?", audio=audio_tensor_1),
    Segment(speaker=1, text="I'm doing well, thank you.", audio=audio_tensor_2)
]

# Generate speech
audio = generator.generate(
    text="That's great to hear!",
    speaker=0,
    context=context,
    temperature=0.9
)

# Save audio
import torchaudio
torchaudio.save("output.wav", audio.unsqueeze(0), generator.sample_rate)
```

## Responsible AI

This implementation includes audio watermarking to identify AI-generated speech. Please use this technology responsibly and maintain the watermarking functionality to promote transparency.

## License

This project is for research purposes only. 