# pip install transformers peft librosa

import librosa
import transformers

pipe = transformers.pipeline(
    model="fixie-ai/ultravox-v0_4_1-llama-3_1-8b", trust_remote_code=True
)

path = "<path-to-input-audio>"  # TODO: pass the audio here
audio, sr = librosa.load(path, sr=16000)


turns = [
    {
        "role": "system",
        "content": "You are a friendly and helpful character. You love to answer questions for people.",
    },
]
pipe({"audio": audio, "turns": turns, "sampling_rate": sr}, max_new_tokens=30)
