import sounddevice as sd
import torch
import transformers

# Check if MPS (Metal) or CPU is available
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

# Initialize the pipeline with the correct device
pipe = transformers.pipeline(
    model="fixie-ai/ultravox-v0_4_1-llama-3_1-8b",
    trust_remote_code=True,
    device=device,  # Pass the selected device
)

# Initialize conversation turns
turns = [
    {
        "role": "system",
        "content": "You are a friendly and helpful character. You love to answer questions for people.",
    },
]

sr = 16000  # Sample rate

while True:
    # Record audio from the microphone
    duration = 5  # Duration of recording in seconds
    print("Please speak now...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    audio = audio.flatten()  # Flatten the audio array

    # Build the input for the pipeline
    input_data = {"audio": audio, "turns": turns, "sampling_rate": sr}

    # Call the pipeline and get the response
    try:
        response = pipe(input_data, max_new_tokens=100)
        print("Assistant's Response:")
        print(response['generated_text'])

        # Append user's transcription and assistant's response to the conversation
        user_transcription = response.get('text', '')
        turns.append({"role": "user", "content": user_transcription})
        assistant_response = response.get('generated_text', '')
        turns.append({"role": "assistant", "content": assistant_response})
    except Exception as e:
        print(f"Error during pipeline processing: {e}")

    # Ask if the user wants to continue
    cont = input("Do you want to continue? (y/n): ")
    if cont.lower() != 'y':
        break
