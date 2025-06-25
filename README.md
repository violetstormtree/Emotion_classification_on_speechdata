# 🎙️ Emotion Classification on Speech Data

This project presents a complete end-to-end pipeline for classifying human emotions from speech signals using deep learning (BiLSTM + LSTM). It processes raw `.wav` files, extracts MFCC features, trains a neural network, evaluates the model, and provides a Streamlit-based web app for real-time predictions.

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


Each filename encodes emotion, intensity, and actor ID.

---

## 🔧 Technologies Used

- Python
- Librosa (for audio processing)
- TensorFlow/Keras (for model training)
- Scikit-learn (for metrics & preprocessing)
- Streamlit (for web deployment)

---

## 🧠 Model Architecture

The model is a sequential neural network with stacked recurrent layers and dropout regularization for generalization. Here is the architecture:

```text
Input (MFCC: 40x174) 
→ Bidirectional LSTM (128 units, return_sequences=True) 
→ Dropout (0.5) 
→ LSTM (64 units) 
→ Dropout (0.5) 
→ Dense (64 units, ReLU) 
→ Dense (8 units, Softmax)
Implemented in Keras:

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(y_onehot.shape[1], activation='softmax')
])
📊 Evaluation Metrics

The model is evaluated using:

Accuracy
F1-Score
Per-class recall
Confusion Matrix
Target Benchmarks:

F1-score > 80%
Accuracy per class > 75%
Overall accuracy > 80%
✅ Results

Example performance (on test set):

Class	Precision	Recall	F1-Score
Angry	0.88	0.87	0.87
Happy	0.91	0.89	0.90
Sad	0.85	0.86	0.85
Neutral	0.80	0.83	0.81
...	...	...	...
Confusion matrix and classification report are displayed in the app output.
🚀 Streamlit Web App

You can upload your own .wav file to see real-time emotion prediction:

🖥️ Features:
Accepts audio upload
Extracts MFCC features
Uses trained model to predict emotion
Run Locally:
streamlit run app.py
Or deploy to Streamlit Cloud.

📦 Repository Structure

├── app.py                  # Streamlit web app
├── emotion_model.h5        # Trained Keras model
├── label_encoder.pkl       # Label encoder for mapping classes
├── README.md
├── requirements.txt
└── Audio_Speech_Actors_01-24/  # Dataset (local or linked externally)
📌 How to Use

Clone the repository
git clone https://github.com/your-username/emotion-classification-speech.git
Install dependencies
pip install -r requirements.txt
Train the model (or use the pre-trained one)
python train.py
Run the Streamlit app
streamlit run app.py
