import sounddevice as sd
import queue
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pynput.keyboard import Key, Controller
import sys
import time

# Initialize model, processor, and keyboard controller
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
keyboard = Controller()

# Create a queue for audio data
audio_queue = queue.Queue()

# Define the callback function for audio data
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())  # Put the audio data into the queue

# Process audio to get transcription
def process_audio(audio, sample_rate=16000):
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

def press_key(key):
    keyboard.press(key)
    keyboard.release(key)

def type_key(key):
    keyboard.type(key)

def press_keys(keys):
    for key in keys:
        keyboard.press(key)
    for key in keys:
        keyboard.release(key)

def exit_program():
    global listening
    print("Exit command received. Stopping...")
    listening = False

# Define actions dictionary
actions = {
    'move leftwards': lambda: press_key(Key.left),
    'move upwards': lambda: press_key(Key.up),
    'move downwards': lambda: press_key(Key.down),
    'move rightwards': lambda: press_key(Key.right),
    'double downwards': lambda: press_keys([Key.down, Key.down]),
    'double upwards': lambda: press_keys([Key.up, Key.up]),
    'double leftwards': lambda: press_keys([Key.left, Key.left]),
    'double rightwards': lambda: press_keys([Key.right, Key.right]),
    'top leftwards': lambda: press_keys([Key.up, Key.left]),
    'top rightwards': lambda: press_keys([Key.up, Key.right]),
    'bottom leftwards': lambda: press_keys([Key.down, Key.left]),
    'bottom rightwards': lambda: press_keys([Key.down, Key.right]),
    'input one': lambda: type_key('1'),
    'input two': lambda: type_key('2'),
    'input three': lambda: type_key('3'),
    'input four': lambda: type_key('4'),
    'input five': lambda: type_key('5'),
    'input six': lambda: type_key('6'),
    'input seven': lambda: type_key('7'),
    'input eight': lambda: type_key('8'),
    'input nine': lambda: type_key('9'),
    'exit': lambda: exit_program()
}

def main():
    sample_rate = 16000  # Sample rate
    channels = 1  # Number of audio channels
    listening = True  # Flag to control the loop
    duration = 5  # Duration to record audio for each instruction (seconds)
    wait_time = 5  # Time to wait before recording the next instruction (seconds)
    buffer_size = duration * sample_rate  # Buffer size in samples

    # Start recording and processing audio
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
        print("Listening for commands...")
        while listening:
            # Initialize a buffer to accumulate audio data
            audio_buffer = []
            while not audio_queue.empty():
                audio_queue.get()  # Clear the audio queue

            start_time = time.time()

            # Indicate to the user to start speaking
            print("Speak now")

            # Accumulate audio data for the specified duration
            while time.time() - start_time < duration and len(audio_buffer) < buffer_size:
                if not audio_queue.empty():
                    audio_data = audio_queue.get()
                    audio_buffer.extend(audio_data.flatten())
                else:
                    # To avoid busy-waiting, sleep briefly
                    time.sleep(0.01)

            # Indicate to the user to stop speaking
            print("Processing...")

            # Restrict audio buffer to the buffer size limit
            if len(audio_buffer) > buffer_size:
                audio_buffer = audio_buffer[:buffer_size]

            # Convert buffer to numpy array
            audio = np.array(audio_buffer[:buffer_size])

            # Process accumulated audio data
            transcription = process_audio(audio, sample_rate)
            print(f"Transcription: {transcription}")

            # Execute keyboard actions based on the transcription
            if transcription in actions:
                actions[transcription]()

            # Wait before recording the next instruction
            time.sleep(wait_time)

if __name__ == "__main__":
    main()
