import os

import librosa
import torch
import transformers

# Check if CUDA (GPU) is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpu_name}")
else:
    print("GPU not available. Falling back to CPU.")

# Initialize the pipeline with the correct device
pipe = transformers.pipeline(
    model="fixie-ai/ultravox-v0_4_1-llama-3_1-8b",
    trust_remote_code=True,
    device=0 if device.type == "cuda" else -1,  # 0 for GPU, -1 for CPU
)

# Initialize conversation turns
turns = [
    {
        "role": "system",
        "content": "You are a friendly and helpful character. You love to answer questions for people.",
    },
]

# Load the audio file
audio_file_path = os.curdir  # Replace with your actual audio file path
if not os.path.exists(audio_file_path):
    print("Audio file not found!")
    raise FileNotFoundError(f"The file {audio_file_path} does not exist.")

# Load the audio file using librosa
audio, sr = librosa.load(audio_file_path, sr=16000)  # Ensure a 16kHz sample rate
print(f"Loaded audio file: {audio_file_path} (Sample rate: {sr} Hz)")

# Build the input for the pipeline
input_data = {"audio": audio, "turns": turns, "sampling_rate": sr}

# Call the pipeline and get the response
try:
    response = pipe(input_data, max_new_tokens=100)
    print("Assistant's Response:")
    print(response)
except Exception as e:
    print(f"Error during pipeline processing: {e}")
