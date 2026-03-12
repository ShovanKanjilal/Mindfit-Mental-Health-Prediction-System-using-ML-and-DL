# MindFit — Mental Health Fitness Prediction System

## Project Report

---

**Project Title:** MindFit — AI-Powered Mental Health Condition Prediction System  
**Technology Stack:** Python, Streamlit, Scikit-learn, PyTorch, Plotly  
**Date:** March 2026

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Literature Survey](#3-literature-survey)
4. [System Architecture & Design](#4-system-architecture--design)
5. [Dataset Description](#5-dataset-description)
6. [Methodology](#6-methodology)
7. [Implementation Details](#7-implementation-details)
8. [Results & Discussion](#8-results--discussion)
9. [Features & User Interface](#9-features--user-interface)
10. [Conclusion & Future Work](#10-conclusion--future-work)
11. [References](#11-references)
12. [Appendix](#12-appendix)

---

## 1. Abstract

Mental health disorders are a growing global concern, with millions of individuals remaining undiagnosed due to stigma, lack of access to healthcare, and insufficient screening mechanisms. This project presents **MindFit**, an AI-powered mental health fitness prediction system that leverages machine learning and deep learning techniques to classify individuals into six mental health condition categories: Normal, Depression, Anxiety, Stress, Bipolar Disorder, and Schizophrenia.

The system employs a dual-model architecture comprising a **Random Forest Classifier** (achieving 94.58% accuracy) and a **Multi-Layer Perceptron (MLP) Neural Network** (achieving 92.50% accuracy), combined through an **ensemble averaging** approach to yield a final classification accuracy of **95.00%**. The application features a comprehensive questionnaire-based assessment with an integrated **lie/inconsistency detection engine** that evaluates the honesty and consistency of user responses. Built using Streamlit, the system provides an interactive, premium-quality web dashboard for data exploration, model insights, and assessment history tracking.

**Keywords:** Mental Health Prediction, Machine Learning, Deep Learning, Random Forest, Neural Network, Ensemble Learning, Lie Detection, Streamlit, Classification

---

## 2. Introduction

### 2.1 Background

Mental health disorders affect approximately 1 in 8 people globally (WHO, 2022), making them one of the most significant public health challenges of the 21st century. Despite the prevalence of conditions such as depression, anxiety, and stress, a substantial proportion of affected individuals remain undiagnosed or untreated. Traditional diagnostic methods rely heavily on clinical interviews and standardized questionnaires administered by trained mental health professionals, which can be time-consuming, subjective, and inaccessible to many populations.

The application of artificial intelligence (AI) and machine learning (ML) to mental health screening has emerged as a promising avenue for early detection and intervention. By analyzing patterns in behavioral, physiological, and self-reported data, ML models can provide preliminary assessments that complement professional clinical evaluation.

### 2.2 Problem Statement

The primary challenge addressed by this project is the development of an accessible, reliable, and user-friendly mental health screening tool that can:

1. Classify individuals into six distinct mental health condition categories using self-reported data.
2. Provide accurate predictions using multiple ML/DL models with ensemble techniques.
3. Detect inconsistencies and potential dishonesty in user responses to improve prediction reliability.
4. Track assessment history over time to monitor changes in mental health condition.
5. Present data insights and model performance metrics through an interactive dashboard.

### 2.3 Objectives

- Design and implement a multi-model prediction system for mental health classification.
- Develop a comprehensive questionnaire with trap questions for consistency verification.
- Build an honesty/lie detection engine using contradiction analysis, social desirability bias detection, and statistical anomaly detection.
- Create a premium, interactive web application using Streamlit with data exploration capabilities.
- Achieve high classification accuracy (>90%) across all six mental health conditions.

### 2.4 Scope

The system is designed as a **screening tool** and not a clinical diagnostic instrument. It serves as a preliminary assessment that can guide individuals toward seeking professional help. The predictions are based on self-reported data and machine learning models trained on structured datasets.

---

## 3. Literature Survey

### 3.1 Machine Learning in Mental Health

The application of machine learning to mental health prediction has gained significant traction in recent years. Various studies have explored the use of supervised and unsupervised learning algorithms for detecting depression, anxiety, and other mental health conditions.

| Study | Approach | Condition | Accuracy |
|-------|----------|-----------|----------|
| Priya et al. (2020) | Decision Trees, SVM | Depression | 85% |
| Sau & Bhakta (2019) | Random Forest, Naive Bayes | Multiple disorders | 89% |
| Islam et al. (2018) | Deep Learning (CNN) | Depression, Anxiety | 91% |
| Su et al. (2020) | Ensemble Methods | Stress Detection | 87% |
| Srividya et al. (2018) | SVM, KNN | Mental illness | 84% |

### 3.2 Random Forest for Classification

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification tasks. Its advantages include:
- Robust to overfitting due to bagging
- Handles high-dimensional data effectively
- Provides feature importance rankings
- Works well with imbalanced datasets (with class weighting)

### 3.3 Multi-Layer Perceptron (MLP)

MLPs are a class of feedforward artificial neural networks consisting of multiple layers of perceptrons. They are particularly effective for tabular data classification tasks. Key features include:
- Non-linear decision boundary learning
- Flexible architecture design
- Ability to capture complex feature interactions
- Effective with proper regularization (Dropout, BatchNorm)

### 3.4 Ensemble Learning

Ensemble methods combine predictions from multiple models to improve overall performance. Average ensembling, as used in this project, computes the mean of class probabilities from individual models, typically resulting in more robust and accurate predictions than any single model.

### 3.5 Response Consistency Detection

Previous work on detecting inconsistent or dishonest self-reporting in psychological assessments includes:
- Validity scales in instruments like MMPI-2
- Social desirability detection (Crowne & Marlowe scale)
- Infrequency items and trap questions
- Statistical anomaly detection in response patterns

---

## 4. System Architecture & Design

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MindFit Application (Streamlit)               │
├─────────┬──────────┬───────────┬──────────────┬────────────────┤
│  Home   │Assessment│  History  │Data Explorer  │ Model Insights │
│  Page   │  Page    │   Page    │    Page       │    Page        │
├─────────┴──────────┴───────────┴──────────────┴────────────────┤
│                    Application Layer (app.py)                    │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│ Questionnaire│ Lie Detector │   History    │  Model Training   │
│   Module     │   Engine     │   Module     │   Pipeline        │
├──────────────┴──────────────┴──────────────┴───────────────────┤
│              ML Models Layer                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │Random Forest  │  │  MLP Neural  │  │  Ensemble Averaging  │  │
│  │ (sklearn)     │  │  Net (PyTorch)│  │  (RF + MLP)          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  Data Layer: mental_health_dataset.csv | assessment_history.json │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Module Description

| Module | File | Purpose |
|--------|------|---------|
| Main Application | `app.py` (1153 lines) | Streamlit multi-page dashboard, UI, model inference |
| Questionnaire | `questionnaire.py` (283 lines) | Question definitions, feature mapping, trap questions |
| Lie Detector | `lie_detector.py` (391 lines) | Contradiction, bias, and anomaly detection engine |
| History | `history.py` (119 lines) | Assessment storage and retrieval (JSON-based) |
| RF Training | `model_training.py` (102 lines) | Random Forest training and evaluation pipeline |
| MLP Training | `mlp_training.py` (210 lines) | PyTorch MLP training with early stopping |
| Results Saver | `save_training_results.py` (317 lines) | Visualization and metrics generation |

### 4.3 Data Flow

```
User Input (Questionnaire)
        │
        ▼
  Answer Collection
        │
        ├──────────────────────────┐
        ▼                          ▼
  Feature Extraction          Lie Detection
  (answers_to_features)       (analyze_honesty)
        │                          │
        ▼                          ▼
  Feature Scaling             Honesty Score
  (StandardScaler)            + Flag Details
        │
        ├──────────┐
        ▼          ▼
   RF Model    MLP Model
   Predict     Predict
        │          │
        ▼          ▼
   RF Probs    MLP Probs
        │          │
        └────┬─────┘
             ▼
      Ensemble Average
             │
             ▼
     Final Prediction
     + Confidence Score
             │
             ▼
     Save to History
     + Display Results
```

---

## 5. Dataset Description

### 5.1 Overview

The project uses the `mental_health_dataset.csv` containing structured mental health survey data.

| Property | Value |
|----------|-------|
| Total Samples | 1,200 |
| Total Features | 19 (+ Patient_ID + Target) |
| Target Variable | Mental_Health_Condition |
| Number of Classes | 6 |
| File Size | ~81 KB |

### 5.2 Features

| # | Feature | Type | Range |
|---|---------|------|-------|
| 1 | Age | Numerical | 18–65 |
| 2 | Gender | Categorical | Male, Female, Other |
| 3 | Sleep_Hours_Per_Day | Numerical | 3.0–10.0 |
| 4 | Work_Hours_Per_Week | Numerical | 0–60 |
| 5 | Exercise_Frequency_Per_Week | Numerical | 0–7 |
| 6 | Mood_Score (0-100) | Numerical | 0–100 |
| 7 | Anxiety_Score (0-100) | Numerical | 0–100 |
| 8 | Depression_Score (0-100) | Numerical | 0–100 |
| 9 | Stress_Score (0-100) | Numerical | 0–100 |
| 10 | Energy_Level (0-100) | Numerical | 0–100 |
| 11 | Social_Interaction (1-10) | Numerical | 1–10 |
| 12 | Concentration (1-10) | Numerical | 1–10 |
| 13 | Appetite (1-10) | Numerical | 1–10 |
| 14 | Life_Satisfaction (0-100) | Numerical | 0–100 |
| 15 | Therapy_Sessions_Per_Month | Numerical | 0–10 |
| 16 | Medication | Binary | 0 (No), 1 (Yes) |
| 17 | Family_History | Binary | 0 (No), 1 (Yes) |
| 18 | Hallucinations | Binary | 0 (No), 1 (Yes) |
| 19 | Manic_Episodes_Per_Year | Numerical | 0–6 |

### 5.3 Target Classes

| Class | Description |
|-------|-------------|
| Normal | No significant mental health concerns |
| Depression | Persistent feelings of sadness and loss of interest |
| Anxiety | Excessive worry, nervousness, and fear |
| Stress | High levels of mental or emotional tension |
| Bipolar Disorder | Extreme mood swings between mania and depression |
| Schizophrenia | Distorted thinking, hallucinations, delusions |

### 5.4 Data Preprocessing

1. **Patient_ID Removal:** The Patient_ID column is dropped as it carries no predictive value.
2. **Gender Encoding:** Categorical gender values (Male, Female, Other) are encoded using `LabelEncoder`.
3. **Target Encoding:** The target variable (Mental_Health_Condition) is label-encoded into integer values (0–5).
4. **Feature Scaling:** All 19 numerical features are standardized using `StandardScaler` (zero mean, unit variance).
5. **Train-Test Split:** 80/20 stratified split (960 training, 240 testing), with `random_state=42` for reproducibility.

---

## 6. Methodology

### 6.1 Model 1: Random Forest Classifier

**Algorithm:** Ensemble of decision trees with bagging.

| Hyperparameter | Value |
|----------------|-------|
| n_estimators | 200 |
| max_depth | 20 |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| class_weight | balanced |
| n_jobs | -1 (parallel) |
| random_state | 42 |

**Key Characteristics:**
- Uses 200 individual decision trees
- Balanced class weighting to handle any class imbalance
- Provides feature importance scores for interpretability
- Supports probability estimates for each class

### 6.2 Model 2: MLP Neural Network

**Framework:** PyTorch

**Architecture:**
```
Input Layer (19 features)
    ↓
Dense Layer (128 neurons) → BatchNorm1d → ReLU → Dropout(0.3)
    ↓
Dense Layer (64 neurons) → BatchNorm1d → ReLU → Dropout(0.3)
    ↓
Dense Layer (32 neurons) → BatchNorm1d → ReLU → Dropout(0.2)
    ↓
Output Layer (6 classes)
```

| Training Parameter | Value |
|--------------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| LR Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Max Epochs | 150 |
| Batch Size | 32 |
| Early Stopping Patience | 20 epochs |
| Loss Function | CrossEntropyLoss (weighted) |

**Regularization Techniques:**
- **Batch Normalization** after each hidden layer for training stability
- **Dropout** (0.3 for layers 1-2, 0.2 for layer 3) to prevent overfitting
- **Early Stopping** to halt training when validation loss stops improving
- **Class Weights** in loss function to handle imbalanced data
- **Weight Decay** (L2 regularization) in optimizer

### 6.3 Model 3: Ensemble (RF + MLP)

The ensemble model combines predictions from both the Random Forest and MLP models:

```
Ensemble_Probability[class_i] = (RF_Probability[class_i] + MLP_Probability[class_i]) / 2
Final_Prediction = argmax(Ensemble_Probability)
```

This averaging approach leverages the complementary strengths of tree-based and neural network models.

### 6.4 Lie / Inconsistency Detection Engine

The lie detection system employs four independent detection methods:

#### 6.4.1 Contradiction Detection
Analyzes 10 predefined feature pairs for logical contradictions:
- **Negative correlations:** Flags when inversely related features are both extreme (e.g., high mood + high depression)
- **Positive correlations:** Flags when correlated features diverge significantly (e.g., mood vs. life satisfaction)

Example contradiction pairs:
| Feature A | Feature B | Expected | Flag When |
|-----------|-----------|----------|-----------|
| Mood Score | Depression Score | Negative | Both > 70% |
| Energy Level | Depression Score | Negative | Both > 70% |
| Mood Score | Life Satisfaction | Positive | Difference > 50% |
| Sleep Hours | Energy Level | Positive | Very low sleep + high energy |

#### 6.4.2 Trap Question Consistency
The questionnaire includes 4 trap questions that rephrase or invert earlier questions:
- `trap_mood`: "Do you generally feel positive?" (checks against mood score)
- `trap_energy`: "Do you often feel tired?" (inverse check against energy level)
- `trap_social`: "Do you prefer to stay alone?" (inverse check against social interaction)
- `trap_depression`: "Do you enjoy hobbies?" (inverse check against depression score)

Significant disagreement between a trap question and its original yields a flag.

#### 6.4.3 Social Desirability Bias
Detects patterns where users present an unrealistically positive profile:
- Flags when ≥80% of positive traits are at maximum AND ≥80% of negative traits are at minimum
- Assigns severity based on the degree of idealization

#### 6.4.4 Statistical Anomaly Detection
Uses z-score analysis to identify responses that are statistically extreme:
- Computes z-scores against dataset statistics for key mental health features
- Flags when ≥3 features have z-scores > 2.5

**Honesty Score Calculation:**
```
Honesty_Score = 100 - Σ(flag_severity × 15)
Risk Levels:
  - Low (≥75): Responses mostly trustworthy
  - Medium (50-74): Several inconsistencies
  - High (25-49): Significant inconsistencies
  - Critical (<25): Major contradictions
```

---

## 7. Implementation Details

### 7.1 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.x |
| Web Framework | Streamlit | 1.41.0 |
| ML Framework | Scikit-learn | 1.6.1 |
| DL Framework | PyTorch | ≥2.0.0 |
| Data Processing | Pandas | 2.2.3 |
| Numerical Computing | NumPy | 2.2.2 |
| Interactive Charts | Plotly | 6.0.0 |
| Static Charts | Matplotlib | 3.10.0 |
| Statistical Viz | Seaborn | 0.13.2 |
| Model Persistence | Joblib | 1.4.2 |

### 7.2 Project Structure

```
My_project/
├── app.py                       # Main Streamlit application (1153 lines)
├── questionnaire.py             # Question definitions & feature mapping
├── lie_detector.py              # Honesty/consistency detection engine
├── history.py                   # Assessment history management
├── model_training.py            # Random Forest training pipeline
├── mlp_training.py              # MLP Neural Network training pipeline
├── save_training_results.py     # Results visualization generator
├── mental_health_dataset.csv    # Training dataset (1200 records)
├── requirements.txt             # Python dependencies
├── models/                      # Trained model artifacts
│   ├── rf_model.joblib          # Random Forest model (~2.7 MB)
│   ├── mlp_model.pth            # MLP model (~64 KB)
│   ├── scaler.joblib            # StandardScaler
│   ├── gender_encoder.joblib    # Gender LabelEncoder
│   ├── target_encoder.joblib    # Target LabelEncoder
│   └── feature_names.joblib     # Feature name list
├── data/
│   └── assessment_history.json  # User assessment history
└── training_results/            # Generated training visualizations
    ├── confusion_matrices.png
    ├── accuracy_comparison.png
    ├── f1_score_comparison.png
    ├── feature_importance.png
    ├── class_distribution.png
    ├── mlp_architecture.png
    ├── classification_reports.txt
    └── training_metrics.json
```

### 7.3 Application Pages

The Streamlit application consists of 5 main pages:

1. **🏠 Home** — Dashboard with dataset statistics, condition distribution charts, and feature range highlights
2. **📝 Assessment** — Complete questionnaire with real-time progress tracking, lie detection analysis, dual-model comparison, and ensemble prediction
3. **📅 History** — Per-user assessment tracking with trend charts (mental scores, confidence/honesty, prediction distribution)
4. **📊 Data Explorer** — Interactive dataset exploration with distributions, correlation heatmap, box plots, and raw data view
5. **📈 Model Insights** — Model performance metrics (accuracy, F1, precision, recall), feature importance, confusion matrix, and classification report

### 7.4 UI Design

The application features a premium dark-mode interface with:
- **Glassmorphism cards** with blur backdrop and hover animations
- **Gradient backgrounds** (deep indigo to purple palette)
- **Inter font** from Google Fonts for modern typography
- **Color-coded condition badges** (e.g., green for Normal, blue for Depression, yellow for Anxiety)
- **Interactive Plotly charts** with consistent dark theme styling
- **Progress bar** for questionnaire completion tracking
- **Responsive layout** using Streamlit's column system

---

## 8. Results & Discussion

### 8.1 Model Performance Summary

| Metric | Random Forest | MLP Neural Net | Ensemble (RF + MLP) |
|--------|:------------:|:--------------:|:-------------------:|
| **Accuracy** | 94.58% | 92.50% | **95.00%** |
| **F1 Score (Weighted)** | 94.56% | 92.52% | **95.01%** |
| **Precision (Weighted)** | 94.90% | 92.69% | **95.14%** |
| **Recall (Weighted)** | 94.58% | 92.50% | **95.00%** |

### 8.2 Per-Class Performance (Ensemble)

| Condition | Precision | Recall | F1 Score | Support |
|-----------|:---------:|:------:|:--------:|:-------:|
| Anxiety | 90.00% | 86.54% | 88.24% | 52 |
| Bipolar Disorder | **100.00%** | **100.00%** | **100.00%** | 24 |
| Depression | **100.00%** | **100.00%** | **100.00%** | 46 |
| Normal | **100.00%** | **100.00%** | **100.00%** | 51 |
| Schizophrenia | **100.00%** | 88.89% | 94.12% | 18 |
| Stress | 86.79% | 93.88% | 90.20% | 49 |

### 8.3 Key Observations

1. **Ensemble superiority:** The ensemble model (95.00%) outperforms both individual models, demonstrating the effectiveness of combining tree-based and neural network approaches.

2. **Perfect classification:** Three conditions — Bipolar Disorder, Depression, and Normal — achieve perfect F1 scores of 100% in the ensemble model, indicating highly distinguishable feature patterns.

3. **Challenging classes:** Anxiety and Stress have the lowest F1 scores (88.24% and 90.20%), likely due to overlapping symptom profiles between these conditions.

4. **Random Forest vs MLP:** The Random Forest (94.58%) outperforms the MLP (92.50%) by ~2%, which is common for tabular data with a moderate sample size (1200 records). Tree-based models typically excel on structured data.

5. **Model complementarity:** The ensemble achieves higher accuracy than either model alone, confirming that RF and MLP capture different aspects of the data distribution.

### 8.4 Feature Importance Analysis

The Random Forest model provides feature importance rankings. The most influential features for mental health classification include:
- **Mood Score** — The strongest predictor of mental health condition
- **Depression Score** — Direct indicator differentiating depression from other conditions
- **Anxiety Score** — Critical for distinguishing anxiety disorders
- **Stress Score** — Key differentiator for stress-related conditions
- **Energy Level** — Important indicator inversely correlated with depression
- **Life Satisfaction** — Broad quality-of-life metric contributing to overall classification
- **Manic Episodes** — Highly specific for identifying Bipolar Disorder
- **Hallucinations** — Highly specific for identifying Schizophrenia

### 8.5 Lie Detection Performance

The lie detection engine provides an additional layer of reliability:
- **Contradiction Detection** identifies logically inconsistent feature pairs
- **Trap Questions** catch users who respond inconsistently to rephrased questions
- **Social Desirability Bias** detects unrealistically positive self-assessments
- **Statistical Anomaly Detection** flags extreme response patterns

When the honesty score drops below 25%, the system warns users that the prediction may not be reliable, encouraging more honest self-reporting.

---

## 9. Features & User Interface

### 9.1 Core Feature Summary

| Feature | Description |
|---------|-------------|
| **Dual-Model Prediction** | Random Forest + MLP with side-by-side comparison |
| **Ensemble Prediction** | Averaged probabilities from both models for robust results |
| **22-Question Assessment** | Comprehensive questionnaire covering demographics, lifestyle, mood, social life, and clinical history |
| **4 Trap Questions** | Hidden consistency check questions for lie detection |
| **Honesty Score** | 0–100 score with risk levels (low, medium, high, critical) |
| **Contradiction Detection** | 10 feature-pair analysis for logical inconsistencies |
| **Social Desirability Bias** | Detection of unrealistically positive profiles |
| **Statistical Anomaly Detection** | Z-score analysis for extreme responses |
| **Assessment History** | Per-user JSON storage with trend tracking |
| **Trend Charts** | Mental score, confidence, and honesty trends over time |
| **Data Explorer** | Interactive distributions, correlations, box plots, and raw data |
| **Model Insights** | Feature importance, confusion matrix, classification report |
| **Premium Dark UI** | Glassmorphism, gradients, Inter font, custom CSS |
| **Probability Breakdown** | Visual comparison of per-class probabilities across models |

### 9.2 Questionnaire Design

The 22-question assessment is organized into 6 sections:

| Section | Questions | Type |
|---------|-----------|------|
| Demographics | Age, Gender | Number, Choice |
| Lifestyle | Sleep, Work, Exercise | Range, Number, Choice |
| How You Feel | Mood, Anxiety, Depression, Stress, Energy | 5-point Scale |
| Social & Daily Life | Social Interaction, Concentration, Appetite, Life Satisfaction | 5-point Scale |
| Clinical History | Therapy, Medication, Family History, Hallucinations, Manic Episodes | Number, Yes/No |
| Additional (Trap) | 4 rephrased verification questions | 5-point Scale |

Scale conversions:
- **Scale 5 → 0-100:** 1→0, 2→25, 3→50, 4→75, 5→100
- **Scale 5 → 1-10:** 1→1, 2→3, 3→5, 4→7, 5→10

---

## 10. Conclusion & Future Work

### 10.1 Conclusion

This project successfully demonstrates the application of machine learning and deep learning techniques for mental health condition prediction. The key achievements include:

1. **High Accuracy:** The ensemble model achieves **95.00% accuracy** across six mental health conditions, with three conditions (Bipolar Disorder, Depression, Normal) achieving perfect classification.

2. **Dual-Model Architecture:** The combination of Random Forest (traditional ML) and MLP (deep learning) provides complementary perspectives, resulting in more robust predictions than either model alone.

3. **Honesty Verification:** The four-method lie detection engine adds a unique reliability layer not commonly found in mental health screening tools, addressing the well-known issue of response bias in self-report assessments.

4. **History Tracking:** The ability to save and visualize assessment history over time enables longitudinal monitoring of mental health trends.

5. **Premium User Experience:** The Streamlit-based dashboard provides an accessible, visually appealing interface that makes mental health screening approachable and engaging.

### 10.2 Limitations

- The system relies on self-reported data, which may be subject to bias even with lie detection.
- The dataset size (1,200 samples) is relatively small compared to clinical-scale studies.
- The model is trained on a specific dataset and may not generalize well to all populations.
- The system is a screening tool and cannot replace professional clinical diagnosis.

### 10.3 Future Work

1. **Larger and more diverse datasets** to improve generalization across demographics and cultures.
2. **Integration of NLP-based analysis** for free-text responses to detect hidden symptoms through linguistic patterns.
3. **Audio biomarker analysis** to capture speech patterns indicative of mental health conditions.
4. **Confidence-based adaptive follow-up questions** that probe deeper when model confidence is low.
5. **Mobile application** for broader accessibility.
6. **Integration with healthcare providers** for seamless referral pathways.
7. **Explainable AI (XAI)** techniques like SHAP values for individual prediction explanations.
8. **Real-time model retraining** as more assessment data is collected.

---

## 11. References

1. World Health Organization (2022). "Mental disorders fact sheet." WHO.
2. Priya, A., Garg, S., & Tigga, N. P. (2020). "Predicting Anxiety, Depression and Stress in Modern Life using Machine Learning Algorithms." *Procedia Computer Science*, 167, 1258-1267.
3. Sau, A., & Bhakta, I. (2019). "Predicting anxiety and depression in elderly patients using machine learning technology." *Healthcare technology letters*, 6(4), 89-93.
4. Islam, M. R., et al. (2018). "Depression detection from social network data using machine learning techniques." *Health Information Science and Systems*, 6(1), 1-12.
5. Su, C., et al. (2020). "Deep learning in mental health outcome research: a scoping review." *Translational Psychiatry*, 10(1), 116.
6. Srividya, M., Mohanavalli, S., & Bhalaji, N. (2018). "Behavioral Modeling for Mental Health using Machine Learning Algorithms." *Journal of Medical Systems*, 42(5), 88.
7. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.
8. Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization." *arXiv preprint arXiv:1412.6980*.
9. Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *Journal of Machine Learning Research*, 15, 1929-1958.
10. Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training." *ICML 2015*.

---

## 12. Appendix

### A. How to Run the Project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the Random Forest model
python model_training.py

# 3. Train the MLP Neural Network
python mlp_training.py

# 4. (Optional) Generate training result visualizations
python save_training_results.py

# 5. Run the Streamlit application
streamlit run app.py
```

### B. Model Artifacts

| File | Description | Size |
|------|-------------|------|
| `rf_model.joblib` | Trained Random Forest model | ~2.7 MB |
| `mlp_model.pth` | Trained MLP PyTorch model | ~64 KB |
| `scaler.joblib` | Fitted StandardScaler | ~1.8 KB |
| `gender_encoder.joblib` | Gender LabelEncoder | ~0.5 KB |
| `target_encoder.joblib` | Target LabelEncoder | ~0.6 KB |
| `feature_names.joblib` | Ordered feature name list | ~0.4 KB |

### C. Classification Reports (Full)

#### Random Forest
```
                  precision    recall  f1-score   support

         Anxiety       0.93      0.83      0.88        52
Bipolar Disorder       1.00      1.00      1.00        24
      Depression       0.98      1.00      0.99        46
          Normal       1.00      1.00      1.00        51
   Schizophrenia       1.00      0.89      0.94        18
          Stress       0.84      0.96      0.90        49

        accuracy                           0.95       240
       macro avg       0.96      0.95      0.95       240
    weighted avg       0.95      0.95      0.95       240
```

#### MLP Neural Network
```
                  precision    recall  f1-score   support

         Anxiety       0.88      0.81      0.84        52
Bipolar Disorder       1.00      1.00      1.00        24
      Depression       1.00      1.00      1.00        46
          Normal       1.00      1.00      1.00        51
   Schizophrenia       0.94      0.89      0.91        18
          Stress       0.80      0.88      0.83        49

        accuracy                           0.93       240
       macro avg       0.94      0.93      0.93       240
    weighted avg       0.93      0.93      0.93       240
```

#### Ensemble (RF + MLP)
```
                  precision    recall  f1-score   support

         Anxiety       0.90      0.87      0.88        52
Bipolar Disorder       1.00      1.00      1.00        24
      Depression       1.00      1.00      1.00        46
          Normal       1.00      1.00      1.00        51
   Schizophrenia       1.00      0.89      0.94        18
          Stress       0.87      0.94      0.90        49

        accuracy                           0.95       240
       macro avg       0.96      0.95      0.95       240
    weighted avg       0.95      0.95      0.95       240
```

---

*This report was generated for the MindFit — Mental Health Fitness Prediction System project.*
