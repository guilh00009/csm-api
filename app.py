import os
import tempfile
import base64
from pathlib import Path
import torch
import torchaudio
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
import json

app = Flask(__name__)

# Initialize the CSM-1B model
def initialize_tts_model():
    print("Initializing TTS model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Download the model if it doesn't exist
    model_path = os.path.join(os.getcwd(), "ckpt.pt")
    if not os.path.exists(model_path):
        print("Downloading model from Hugging Face...")
        model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
    
    print(f"Loading model from {model_path}")
    generator = load_csm_1b(model_path, device)
    print("TTS model initialized successfully")
    return generator

# Initialize the TTS model
tts_generator = initialize_tts_model()

# Helper function to load audio from file
def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=tts_generator.sample_rate
    )
    return audio_tensor

@app.route('/api/tts/generate', methods=['POST'])
def generate_speech():
    """
    Generate speech from text without context
    
    Request JSON format:
    {
        "text": "Text to convert to speech",
        "speaker": 0,  // optional, defaults to 0
        "max_audio_length_ms": 10000,  // optional, defaults to 10000
        "temperature": 0.9,  // optional, defaults to 0.9
        "return_format": "url"  // optional, "url" or "base64", defaults to "url"
    }
    """
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing required parameter: text"}), 400
        
        text = data.get('text')
        speaker = data.get('speaker', 0)
        max_audio_length_ms = data.get('max_audio_length_ms', 10000)
        temperature = data.get('temperature', 0.9)
        return_format = data.get('return_format', 'url')
        
        # Generate audio
        audio = tts_generator.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
        )
        
        # Save audio to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(temp_file.name, audio.unsqueeze(0).cpu(), tts_generator.sample_rate)
        
        if return_format == 'base64':
            # Return audio as base64
            with open(temp_file.name, 'rb') as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return jsonify({
                "audio": audio_base64,
                "sample_rate": tts_generator.sample_rate
            })
        else:
            # Return audio URL
            filename = Path(temp_file.name).name
            return jsonify({
                "audio_url": f"/api/audio/{filename}",
                "sample_rate": tts_generator.sample_rate
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/tts/generate_with_context', methods=['POST'])
def generate_speech_with_context():
    """
    Generate speech from text with context
    
    Request JSON format:
    {
        "text": "Text to convert to speech",
        "speaker": 0,  // optional, defaults to 0
        "context": [
            {
                "text": "Previous utterance 1",
                "speaker": 0,
                "audio_base64": "base64_encoded_audio"  // base64 encoded WAV audio
            },
            // more context segments...
        ],
        "max_audio_length_ms": 10000,  // optional, defaults to 10000
        "temperature": 0.9,  // optional, defaults to 0.9
        "return_format": "url"  // optional, "url" or "base64", defaults to "url"
    }
    """
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing required parameter: text"}), 400
        
        text = data.get('text')
        speaker = data.get('speaker', 0)
        context_data = data.get('context', [])
        max_audio_length_ms = data.get('max_audio_length_ms', 10000)
        temperature = data.get('temperature', 0.9)
        return_format = data.get('return_format', 'url')
        
        # Process context segments
        segments = []
        for ctx in context_data:
            if 'text' not in ctx or 'speaker' not in ctx or 'audio_base64' not in ctx:
                return jsonify({"error": "Context segments must contain text, speaker, and audio_base64"}), 400
            
            # Decode base64 audio and save to temp file
            audio_bytes = base64.b64decode(ctx['audio_base64'])
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with open(temp_audio.name, 'wb') as f:
                f.write(audio_bytes)
            
            # Load audio
            audio_tensor = load_audio(temp_audio.name)
            
            # Create segment
            segments.append(Segment(
                text=ctx['text'],
                speaker=ctx['speaker'],
                audio=audio_tensor
            ))
            
            # Clean up temp file
            os.unlink(temp_audio.name)
        
        # Generate audio with context
        audio = tts_generator.generate(
            text=text,
            speaker=speaker,
            context=segments,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
        )
        
        # Save audio to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(temp_file.name, audio.unsqueeze(0).cpu(), tts_generator.sample_rate)
        
        if return_format == 'base64':
            # Return audio as base64
            with open(temp_file.name, 'rb') as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return jsonify({
                "audio": audio_base64,
                "sample_rate": tts_generator.sample_rate
            })
        else:
            # Return audio URL
            filename = Path(temp_file.name).name
            return jsonify({
                "audio_url": f"/api/audio/{filename}",
                "sample_rate": tts_generator.sample_rate
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/tts/generate_streaming', methods=['POST'])
def generate_speech_streaming():
    """
    Generate speech from text and stream it as audio chunks using Server-Sent Events
    
    Request JSON format:
    {
        "text": "Text to convert to speech",
        "speaker": 0,  // optional, defaults to 0
        "context": [  // optional
            {
                "text": "Previous utterance 1",
                "speaker": 0,
                "audio_base64": "base64_encoded_audio"
            },
            // more context segments...
        ],
        "max_audio_length_ms": 10000,  // optional, defaults to 10000
        "temperature": 0.9,  // optional, defaults to 0.9
        "apply_watermark": true,  // optional, defaults to true
    }
    
    Response: A stream of Server-Sent Events, each containing a base64-encoded audio chunk
    Each event has the format:
    event: audio
    data: {"audio": "base64_encoded_audio_chunk", "sample_rate": 24000, "chunk_id": 0}
    
    The final event has:
    event: end
    data: {"sample_rate": 24000, "total_chunks": 5}
    """
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing required parameter: text"}), 400
        
        text = data.get('text')
        speaker = data.get('speaker', 0)
        context_data = data.get('context', [])
        max_audio_length_ms = data.get('max_audio_length_ms', 10000)
        temperature = data.get('temperature', 0.9)
        apply_watermark = data.get('apply_watermark', True)
        
        # Process context segments if provided
        segments = []
        for ctx in context_data:
            if 'text' not in ctx or 'speaker' not in ctx or 'audio_base64' not in ctx:
                return jsonify({"error": "Context segments must contain text, speaker, and audio_base64"}), 400
            
            # Decode base64 audio and save to temp file
            audio_bytes = base64.b64decode(ctx['audio_base64'])
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with open(temp_audio.name, 'wb') as f:
                f.write(audio_bytes)
            
            # Load audio
            audio_tensor = load_audio(temp_audio.name)
            
            # Create segment
            segments.append(Segment(
                text=ctx['text'],
                speaker=ctx['speaker'],
                audio=audio_tensor
            ))
            
            # Clean up temp file
            os.unlink(temp_audio.name)
        
        def generate():
            # Send sample rate and metadata information
            yield f"event: info\ndata: {json.dumps({'sample_rate': tts_generator.sample_rate})}\n\n"
            
            # Generate audio chunks
            chunk_id = 0
            for chunk in tts_generator.generate_streaming(
                text=text,
                speaker=speaker,
                context=segments,
                max_audio_length_ms=max_audio_length_ms,
                temperature=temperature,
                apply_watermark=apply_watermark,
            ):
                # Convert chunk to bytes
                temp_chunk = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                torchaudio.save(temp_chunk.name, chunk.unsqueeze(0).cpu(), tts_generator.sample_rate)
                
                with open(temp_chunk.name, 'rb') as f:
                    chunk_bytes = f.read()
                
                os.unlink(temp_chunk.name)
                
                # Yield chunk as SSE event with metadata
                chunk_data = {
                    'audio': base64.b64encode(chunk_bytes).decode('utf-8'),
                    'sample_rate': tts_generator.sample_rate,
                    'chunk_id': chunk_id
                }
                yield f"event: audio\ndata: {json.dumps(chunk_data)}\n\n"
                chunk_id += 1
            
            # Signal end of stream with metadata
            yield f"event: end\ndata: {json.dumps({'sample_rate': tts_generator.sample_rate, 'total_chunks': chunk_id})}\n\n"
        
        # Return streaming response
        return Response(stream_with_context(generate()), 
                       mimetype='text/event-stream',
                       headers={
                           'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive',
                           'X-Accel-Buffering': 'no'  # Disable buffering for Nginx
                       })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/audio/<filename>')
def serve_audio(filename):
    """Serve generated audio files"""
    temp_dir = tempfile.gettempdir()
    return send_from_directory(temp_dir, filename)

@app.route('/api/info', methods=['GET'])
def get_info():
    """Get information about the TTS model"""
    return jsonify({
        "model": "CSM-1B",
        "sample_rate": tts_generator.sample_rate,
        "description": "Conversational Speech Model from Sesame AI Labs",
        "endpoints": [
            {
                "path": "/api/tts/generate",
                "method": "POST",
                "description": "Generate speech from text without context"
            },
            {
                "path": "/api/tts/generate_with_context",
                "method": "POST",
                "description": "Generate speech from text with context"
            },
            {
                "path": "/api/tts/generate_streaming",
                "method": "POST",
                "description": "Generate speech from text and stream it as audio chunks"
            },
            {
                "path": "/api/audio/<filename>",
                "method": "GET",
                "description": "Serve generated audio files"
            },
            {
                "path": "/api/info",
                "method": "GET",
                "description": "Get information about the TTS model"
            }
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=3000) 