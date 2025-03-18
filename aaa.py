import requests
import json

# Define the API endpoint
url = "https://4u9hh5rz7jyrll-3000.proxy.runpod.net/api/tts/generate"

# Define the request payload
data = {
    "text": "Hello! im a female as you can see",
    "speaker": 0,
    "max_audio_length_ms": 10000,
    "temperature": 0.9,
    "return_format": "url"  # Change to "base64" if you want base64-encoded audio
}

# Send the request
response = requests.post(url, json=data)

# Check response
if response.status_code == 200:
    response_json = response.json()
    if "audio_url" in response_json:
        print(f"Audio generated! Download it from: {url.replace('/api/tts/generate', '')}{response_json['audio_url']}")
    elif "audio" in response_json:
        print("Audio generated in base64 format.")
else:
    print(f"Error: {response.status_code}, {response.text}")
