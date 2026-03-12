# 🧠 MindFit — Mental Health Fitness

An AI-powered mental health condition prediction dashboard built with **Streamlit**, **scikit-learn**, and **PyTorch**.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)

## 📌 Overview

MindFit uses machine learning to predict mental health conditions based on lifestyle, behavioral, and clinical factors. It features a **dual-model ensemble** (Random Forest + MLP Neural Network) for more reliable predictions, along with a built-in **honesty/consistency detector**.

### Key Features

- **🏠 Dashboard** — Interactive data exploration with condition distribution visualizations
- **📝 Smart Assessment** — Sectioned questionnaire covering demographics, lifestyle, mood, social life, and clinical history
- **🕵️ Honesty Detection** — AI-powered consistency analysis that flags contradictory responses
- **🔮 Dual-Model Prediction** — Random Forest (ML) + MLP Neural Network (DL) ensemble predictions
- **📅 History Tracking** — Save and track assessment results over time per user
- **📊 Data Explorer** — Deep dive into the dataset with interactive charts
- **📈 Model Insights** — Feature importance, model performance metrics, and comparisons

## 🗂️ Project Structure

```
My_project/
├── app.py                    # Main Streamlit application
├── model_training.py         # Random Forest model training
├── mlp_training.py           # MLP Neural Network training
├── questionnaire.py          # Assessment questionnaire definitions
├── lie_detector.py           # Honesty/consistency analysis
├── history.py                # Assessment history management
├── save_training_results.py  # Training results export utility
├── mental_health_dataset.csv # Dataset
├── requirements.txt          # Python dependencies
├── models/                   # Trained model files
│   ├── rf_model.joblib
│   ├── mlp_model.pth
│   ├── scaler.joblib
│   ├── gender_encoder.joblib
│   ├── target_encoder.joblib
│   └── feature_names.joblib
├── training_results/         # Saved training visualizations & metrics
└── viva_notes/               # Project documentation & notes
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/MindFit.git
   cd MindFit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (optional — pre-trained models are included)
   ```bash
   python model_training.py   # Train Random Forest
   python mlp_training.py     # Train MLP Neural Network
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

   The app will open at `http://localhost:8501`

## 📊 Models

| Model | Type | Description |
|-------|------|-------------|
| Random Forest | Machine Learning | Ensemble of decision trees for robust classification |
| MLP Neural Network | Deep Learning | Multi-layer perceptron with batch normalization and dropout |
| Ensemble | Combined | Averaged probabilities from both models for best accuracy |

## 🎯 Conditions Predicted

- ✅ Normal
- 💙 Depression
- ⚡ Anxiety
- 🔥 Stress
- 🎭 Bipolar Disorder
- 🌀 Schizophrenia

## ⚠️ Disclaimer

This is an AI-based screening tool and **NOT a clinical diagnosis**. Please consult a qualified mental health professional for proper evaluation.

## 📝 License

This project is for educational purposes.
