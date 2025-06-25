# 🎙️ Emotion Classification on Speech Data

This project presents a complete end-to-end pipeline for classifying human emotions from speech signals using deep learning (BiLSTM). It processes raw `.wav` files, extracts MFCC features, trains a neural network, evaluates the model, and provides a Streamlit-based web app for real-time predictions.

---

## 📌 Objective

The aim is to detect and classify emotions conveyed in speech using machine learning. The system leverages audio feature extraction and a deep learning model to accurately predict one of the following emotions:
- **Neutral**
- **Calm**
- **Happy**
- **Sad**
- **Angry**
- **Fearful**
- **Disgust**
- **Surprised**

---

## 📁 Dataset

The dataset used is [RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976), which contains 24 actors vocalizing two lexically-matched statements in a range of emotions.

**Data Format:**
Audio_Speech_Actors_01-24/
├── Actor_01/
│ ├── 03-01-01-01-01-01-01.wav
│ └── ...
├── Actor_02/
└── ...
## 🔧 Technologies Used

- Python
- Librosa (for audio processing)
- TensorFlow/Keras (for model training)
- Scikit-learn (for metrics & preprocessing)
- Streamlit (for web deployment)

---

## 🧠 Model Architecture

- Input: MFCC features (40 coefficients per frame)
- Model: Bidirectional LSTM + LSTM + Dense
- Output: 8-class softmax classifier

```text
Input (MFCC) → BiLSTM(128) → Dropout → LSTM(64) → Dense(64, relu) → Dense(8, softmax)
📌 How to Use

Clone the repository
git clone https://github.com/your-username/emotion-classification-speech.git
Install dependencies
pip install -r requirements.txt
Train the model (or use the pre-trained one)
python train.py
Run the Streamlit app
streamlit run app.py
