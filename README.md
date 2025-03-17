# CSM-3B Training Pipeline

This repository contains code for training the CSM-3B (Conversational Speech Model) on the Switchboard dataset. The model is designed for speech synthesis with conversational context.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download the Switchboard dataset. The dataset can be obtained from [LDC](https://catalog.ldc.upenn.edu/LDC97S62).

## Preprocessing

Before training, you need to preprocess the Switchboard dataset:

```bash
python preprocess_switchboard.py --input_dir /path/to/switchboard --output_dir switchboard
```

This will:
- Process all audio files and their transcripts
- Segment the audio based on utterances
- Create train/validation/test splits
- Save metadata in JSON format

## Training

To train the model:

```bash
python train.py --config config.json
```

You can customize the training parameters by editing the `config.json` file.

To resume training from a checkpoint:

```bash
python train.py --resume checkpoints/checkpoint_latest.pt
```

## Model Architecture

The CSM-3B model consists of:
- A backbone transformer (Llama-3.2-3B-Instruct with 28 layers, 24 attention heads)
- A decoder transformer (Llama-3.2 100M)
- Text and audio embedding layers
- Projection layers between backbone and decoder

The model is trained to predict both text and audio tokens in a conversational context.

## Dataset

The Switchboard dataset contains telephone conversations between two speakers. The dataset has been preprocessed to:
- Upsample audio to 16kHz
- Separate channels for different speakers
- Process transcripts with special markers for paralinguistic events (laughter, etc.)

## Inference

After training, you can use the model for inference:

```python
from generator import load_csm_1b, Segment

# Load the model
generator = load_csm_1b("checkpoints/model_epoch9_step1000.pt", backbone_flavor="llama-3B-instruct")

# Create context segments
context = [
    Segment(speaker=0, text="Hello, how are you?", audio=...),
    Segment(speaker=1, text="I'm doing well, thanks!", audio=...),
]

# Generate audio for a new utterance
audio = generator.generate(
    text="That's great to hear!",
    speaker=0,
    context=context,
    temperature=0.9,
    topk=50,
)
```

## Watermarking

The generated audio includes an imperceptible watermark to identify it as AI-generated. This ensures transparency and enables traceability. Please keep the watermarking in place when using this model.

## License

This code is provided for research purposes only. Please check the license of the Switchboard dataset before using it for commercial purposes. 