# app/main.py
from collections import defaultdict
from io import BytesIO

import librosa
import torch
import transformers
from fastapi import Depends, FastAPI, File, Request, UploadFile

app = FastAPI()

# Set device for GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the pipeline globally
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
        # Use client's IP and port as a simple client_id
        if not request.client:
            return "unknown"
        client_id = request.client.host + ":" + str(request.client.port)
    return client_id


@app.post("/process-audio/")
async def process_audio(
    file: UploadFile = File(...), client_id: str = Depends(get_client_id)
):
    # Read the audio file
    audio_bytes = await file.read()
    # Load the audio file using librosa
    audio_stream = BytesIO(audio_bytes)
    try:
        audio, sr = librosa.load(audio_stream, sr=16000)
    except Exception as e:
        return {"error": f"Failed to load audio file: {e}"}
    # Retrieve the client's conversation history
    turns = client_conversations[client_id]
    if not turns:
        # Initialize conversation if empty
        turns.append(
            {
                "role": "system",
                "content": "You are a friendly and helpful character. You love to answer questions for people.",
            }
        )
    # Prepare input data for the pipeline
    input_data = {"audio": audio, "turns": turns, "sampling_rate": sr}
    # Call the pipeline
    try:
        response = pipe(input_data, max_new_tokens=300)
        # Update the conversation history
        client_conversations[client_id].append(response)
        # Return the response
        return {"response": response}
    except Exception as e:
        return {"error": f"Error during pipeline processing: {e}"}
