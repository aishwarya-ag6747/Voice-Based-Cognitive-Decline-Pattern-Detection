
# 🧠 Voice-Based Cognitive Decline Pattern Detection

This project detects early signs of **cognitive decline** by analyzing speech recordings. It uses **Wav2Vec2** for transcription, extracts **linguistic and acoustic features**, and applies **unsupervised learning** (Isolation Forest & K-Means) to flag individuals potentially at risk. A **Streamlit web app** allows for interactive analysis and visualization.

---

## 🚀 Features

- 🔊 Speech-to-text transcription using Facebook's **Wav2Vec2**
- 🧠 Cognitive markers from speech (hesitations, speech rate, pitch, semantics)
- 🤖 Unsupervised learning for anomaly & cluster detection
- 📉 PCA-based cluster visualization
- 🌐 User-friendly Streamlit interface

---

## 🛠️ Tech Stack

- Python 3.8+
- Libraries: `transformers`, `torchaudio`, `librosa`, `scikit-learn`, `streamlit`, `pandas`, `matplotlib`, `gensim`
- Model: `facebook/wav2vec2-base-960h` (inference only)

---

## 📁 Project Structure

```
Voice-Based-Cognitive-Decline-Pattern-Detection/
│
├── dataset/                  # Contains audio files (.wav)
├── app.py                   # Streamlit web application
├── model.py                 # Core processing (transcription, features, clustering)
├── README.md                # Project documentation
```

---

## ✅ Setup Instructions

> Make sure you have Python 3.8+ installed.

### 1. Clone the repository

```bash
git clone https://github.com/aishwarya-ag6747/Voice-Based-Cognitive-Decline-Pattern-Detection.git
cd Voice-Based-Cognitive-Decline-Pattern-Detection
```

### 2. Install dependencies

```bash
pip install transformers torchaudio librosa soundfile nltk gensim scikit-learn streamlit matplotlib pandas
```

---

## ▶️ Running the Web App

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually [http://localhost:8501](http://localhost:8501)).

---

## 📂 How It Works

1. **Upload** an audio file in `.wav` format.
2. **Transcription** is performed using `Wav2Vec2`.
3. **Features** like hesitations, pitch, rate, and semantics are extracted.
4. **Clustering & anomaly detection** identify speech at risk.
5. **Results** are shown via cluster plot and predicted label.

---

## 📝 Notes

- Works offline; no external API calls.
- Designed for demo/research — not clinically validated.
- Wav2Vec2 used for inference only (not fine-tuned).

---

## 📧 Contact

For queries or contributions:  
**Aishwarya Mondal**  
GitHub: [aishwarya-ag6747](https://github.com/aishwarya-ag6747)

---



