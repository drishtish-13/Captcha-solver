import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

MODEL_NAME = "facebook/wav2vec2-large-960h"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

def load_audio(file_path, target_sr=16000):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio

def transcribe_audio(file_path):
    audio = load_audio(file_path)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

if __name__ == "__main__":
    audio_path = r"data/audio/captcha_11.wav"  

    try:
        transcription = transcribe_audio(audio_path)
        print(f"✅ Transcription: {transcription}")
    except Exception as e:
        print(f"❌ Error: {e}")

