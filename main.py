# app/main.py
import os
from collections import defaultdict
from io import BytesIO

import numpy as np
import torch
import transformers
from fastapi import Depends, FastAPI, File, Request, UploadFile
from pydub import AudioSegment
from scipy.signal import resample

app = FastAPI()

# Initialize the pipeline as before
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe = transformers.pipeline(
    model="fixie-ai/ultravox-v0_4_1-llama-3_1-8b",
    trust_remote_code=True,
    device=0 if device.type == "cuda" else -1,
)

# In-memory store for conversation history
client_conversations = defaultdict(list)


async def get_client_id(request: Request):
    client_id = request.headers.get("X-Client-ID")
    if not client_id:
        if not request.client:
            return "unknown"
        client_id = f"{request.client.host}:{request.client.port}"
    return client_id


@app.get("/")
async def hello():
    return "Hello World!"


@app.post("/process-audio/")
async def process_audio(
    file: UploadFile = File(...), client_id: str = Depends(get_client_id)
):
    # Read the audio bytes
    audio_bytes = await file.read()
    if not file.filename:
        return {"error": "No file name provided"}

    file_extension = os.path.splitext(file.filename)[1][
        1:
    ]  # Get extension without the dot

    # Load audio using PyDub
    try:
        audio_segment = AudioSegment.from_file(
            BytesIO(audio_bytes), format=file_extension
        )

        # Convert to mono if necessary
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)

        # Get the raw audio data as a numpy array
        audio_data = np.array(audio_segment.get_array_of_samples()).astype(np.float32)

        # Normalize audio data to range [-1.0, 1.0]
        audio_data /= np.iinfo(np.int16).max  # Assuming 16-bit audio

        original_sr = audio_segment.frame_rate

        # Resample to 16000 Hz if necessary
        target_sr = 16000
        if original_sr != target_sr:
            # Calculate the number of samples after resampling
            num_samples = int(len(audio_data) * float(target_sr) / original_sr)
            # Resample the audio data
            audio_data = resample(audio_data, num_samples)
            sr = target_sr
        else:
            sr = original_sr

    except Exception as e:
        return {"error": f"Failed to load or process audio file: {e}"}

    # Retrieve the client's conversation history
    turns = client_conversations[client_id]
    if not turns:
        turns.append(
            {
                "role": "system",
                "content": "You are a friendly and helpful character. You love to answer questions for people.",
            }
        )

    tf_input = [d for d in turns]

    input_data = {"audio": audio_data, "turns": tf_input, "sampling_rate": sr}

    # Call the pipeline
    try:
        response = pipe(input_data, max_new_tokens=512)

        # Log the response for debugging
        print("Pipeline response:", response)

        # Format the assistant's response correctly
        assistant_response = {
            "role": "assistant",
            "content": response,  # Adjust if response has a different structure
        }

        # Update the conversation history
        client_conversations[client_id].append(assistant_response)

        # Return the response
        return {"response": assistant_response}
    except Exception as e:
        return {"error": f"Error during pipeline processing: {e}"}
