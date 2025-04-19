# model.py

import torch, torchaudio, librosa, numpy as np, re
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd

# Load Wav2Vec model
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

def transcribe(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    
    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0]).lower()


# Extract features
def extract_features(text, audio_path):
    def count_hesitations(text):
        return len(re.findall(r'\b(uh|um|erm|eh|hmm|ah)\b', text.lower()))
    def speech_rate(text, dur): return len(text.split()) / dur if dur > 0 else 0
    def pitch_variability(path):
        y, sr = librosa.load(path, sr=None)
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        values = pitches[pitches > 0]
        return np.std(values) if len(values) > 0 else 0

    duration = librosa.get_duration(path=audio_path)
    return {
        "hesitation_count": count_hesitations(text),
        "pause_count": count_hesitations(text),
        "speech_rate": speech_rate(text, duration),
        "pitch_variability": pitch_variability(audio_path),
        "semantic_anomaly": 0.1,  # Placeholder
        "vague_word_count": sum(text.lower().split().count(w) for w in ['thing', 'stuff', 'something']),
        "incomplete_sentence": int(text.endswith(("and", "but", "then", ","))),
        "lost_words": len(re.findall(r'\b(that thing|you know|like)\b', text.lower()))
    }

def predict_risk(features, scaler, model):
    X = pd.DataFrame([features])
    X_scaled = scaler.transform(X)
    result = model.predict(X_scaled)[0]
    return "At Risk" if result == -1 else "Normal"
