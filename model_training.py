"""
Mental Health Fitness — Model Training Pipeline
Trains a Random Forest classifier on mental_health_dataset.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# ─── Configuration ────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "mental_health_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ─── Load & Preprocess ───────────────────────────────────────
def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    # Drop patient ID — not a feature
    df = df.drop(columns=["Patient_ID"])

    # Encode Gender
    gender_encoder = LabelEncoder()
    df["Gender"] = gender_encoder.fit_transform(df["Gender"])

    # Separate features and target
    X = df.drop(columns=["Mental_Health_Condition"])
    y = df["Mental_Health_Condition"]

    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, scaler, gender_encoder, target_encoder, X.columns.tolist()


# ─── Train ────────────────────────────────────────────────────
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, accuracy, report, cm, X_test, y_test, y_pred


# ─── Save ─────────────────────────────────────────────────────
def save_artifacts(model, scaler, gender_encoder, target_encoder, feature_names):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(gender_encoder, os.path.join(MODEL_DIR, "gender_encoder.joblib"))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.joblib"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.joblib"))
    print(f"✅ Artifacts saved to {MODEL_DIR}/")


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Mental Health Fitness — Model Training")
    print("=" * 60)

    print("\n📂 Loading dataset...")
    X, y, scaler, gender_enc, target_enc, feat_names = load_and_preprocess()
    print(f"   Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
    print(f"   Classes: {list(target_enc.classes_)}")

    print("\n🏋️  Training Random Forest...")
    model, acc, report, cm, X_test, y_test, y_pred = train_model(X, y)

    print(f"\n🎯 Accuracy: {acc:.4f}  ({acc*100:.1f}%)")
    print(f"\n📋 Classification Report:\n{report}")

    save_artifacts(model, scaler, gender_enc, target_enc, feat_names)
    print("\n🚀 Training complete!")
