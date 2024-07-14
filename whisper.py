import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import queue
import time
from pocketsphinx import LiveSpeech
import winsound

# Initialize model and processor
processor = WhisperProcessor.from_pretrained("D:\\Models\\openai-whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("D:\\Models\\openai-whisper-small")
model.config.forced_decoder_ids = None

# Parameters for recording
sample_rate = 16000
record_duration = 5  # duration of recording after wake word detection

# Create a queue to store the audio data
audio_queue = queue.Queue()

# Function to process and transcribe audio
def transcribe_audio(audio_data):
    input_features = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# Function to record audio after wake word is detected
def record_audio(duration, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")
    winsound.Beep(1000, 200)
    return audio

# Initialize wake word detection
speech = LiveSpeech(lm=False, dic='custom.dict',
    keyphrase='jasper', kws_threshold=1e-20)

print("Listening for wake word...")

for phrase in speech:
    winsound.Beep(1000, 200)
    os.system('echo \a')
    print("Wake word detected")
    # Record audio
    audio_data = record_audio(record_duration)
    # Preprocess audio data
    audio_data = audio_data.mean(axis=1) if audio_data.ndim > 1 else audio_data
    # Transcribe audio data
    transcription = transcribe_audio(audio_data)
    print("Transcription:", transcription)
    print("Listening for wake word...")
