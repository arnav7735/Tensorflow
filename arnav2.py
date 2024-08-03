import torch
import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pretrained model and processor
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Record audio from microphone
def record_audio(duration, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return audio.flatten()

# Process audio
def process_audio(audio, sample_rate=16000):
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

# Main function
def main():
    duration = 7  # Duration in seconds
    sample_rate = 16000  # Sample rate for wav2vec2-base-960h
    audio = record_audio(duration, sample_rate)
    transcription = process_audio(audio, sample_rate)
    print(f"Transcription: {transcription}")

if __name__ == "__main__":
    main()
