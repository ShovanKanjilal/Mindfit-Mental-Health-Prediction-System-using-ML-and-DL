"""
Generate and save all important training results, visualizations,
and metrics for both Random Forest and MLP models into a
training_results/ subfolder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib

from model_training import load_and_preprocess

# ─── Config ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "training_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Premium dark theme for all plots
plt.style.use("dark_background")
COLORS = ["#a78bfa", "#60a5fa", "#34d399", "#f472b6", "#fbbf24", "#f87171"]


def save_fig(fig, name):
    """Save figure as PNG."""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0f0c29", edgecolor="none")
    plt.close(fig)
    print(f"  📊 Saved: {name}")


# ─── Load Data ────────────────────────────────────────────────
print("=" * 60)
print("  Generating Training Results & Visualizations")
print("=" * 60)

print("\n📂 Loading data...")
X, y, scaler, gender_enc, target_enc, feat_names = load_and_preprocess()
class_names = list(target_enc.classes_)
num_classes = len(class_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ─── Load Models ──────────────────────────────────────────────
print("\n📦 Loading models...")
rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))

# MLP
from mlp_training import MentalHealthMLP
checkpoint = torch.load(os.path.join(MODEL_DIR, "mlp_model.pth"), map_location="cpu", weights_only=True)
mlp_model = MentalHealthMLP(checkpoint["input_size"], checkpoint["num_classes"])
mlp_model.load_state_dict(checkpoint["model_state_dict"])
mlp_model.eval()

# ─── Predictions ──────────────────────────────────────────────
print("\n🔮 Generating predictions...")
# RF
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# MLP
with torch.no_grad():
    mlp_out = mlp_model(torch.FloatTensor(X_test))
    mlp_proba = torch.softmax(mlp_out, dim=1).numpy()
    mlp_pred = mlp_proba.argmax(axis=1)
mlp_acc = accuracy_score(y_test, mlp_pred)

# Ensemble
ens_proba = (rf_proba + mlp_proba) / 2.0
ens_pred = ens_proba.argmax(axis=1)
ens_acc = accuracy_score(y_test, ens_pred)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. CONFUSION MATRICES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n📊 1. Confusion Matrices...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Confusion Matrices — RF vs MLP vs Ensemble", fontsize=16, color="#e0d4ff", fontweight="bold")

for ax, (name, preds) in zip(axes, [("Random Forest", rf_pred), ("MLP Neural Net", mlp_pred), ("Ensemble", ens_pred)]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, linecolor="#1a1333", cbar_kws={"shrink": 0.8})
    ax.set_title(name, fontsize=13, color="#c4b5fd", fontweight="600")
    ax.set_xlabel("Predicted", color="#9ca3af")
    ax.set_ylabel("Actual", color="#9ca3af")
    ax.tick_params(colors="#9ca3af", labelsize=8)
save_fig(fig, "confusion_matrices.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. ACCURACY COMPARISON BAR CHART
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("📊 2. Accuracy Comparison...")

fig, ax = plt.subplots(figsize=(8, 5))
models = ["Random Forest", "MLP Neural Net", "Ensemble"]
accs = [rf_acc * 100, mlp_acc * 100, ens_acc * 100]
bars = ax.bar(models, accs, color=["#60a5fa", "#a78bfa", "#34d399"], edgecolor="none", width=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=14, fontweight="bold", color="#e0d4ff")
ax.set_ylabel("Accuracy (%)", color="#9ca3af", fontsize=12)
ax.set_title("Model Accuracy Comparison", fontsize=15, color="#e0d4ff", fontweight="bold")
ax.set_ylim(0, 105)
ax.tick_params(colors="#9ca3af")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#333")
ax.spines["bottom"].set_color("#333")
ax.set_facecolor("#0f0c29")
save_fig(fig, "accuracy_comparison.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. PER-CLASS F1 SCORE COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("📊 3. Per-class F1 Scores...")

rf_report = classification_report(y_test, rf_pred, target_names=class_names, output_dict=True)
mlp_report = classification_report(y_test, mlp_pred, target_names=class_names, output_dict=True)
ens_report = classification_report(y_test, ens_pred, target_names=class_names, output_dict=True)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(class_names))
width = 0.25
ax.bar(x - width, [rf_report[c]["f1-score"] * 100 for c in class_names], width, label="Random Forest", color="#60a5fa")
ax.bar(x, [mlp_report[c]["f1-score"] * 100 for c in class_names], width, label="MLP Neural Net", color="#a78bfa")
ax.bar(x + width, [ens_report[c]["f1-score"] * 100 for c in class_names], width, label="Ensemble", color="#34d399")
ax.set_xlabel("Condition", color="#9ca3af", fontsize=12)
ax.set_ylabel("F1 Score (%)", color="#9ca3af", fontsize=12)
ax.set_title("Per-Class F1 Score Comparison", fontsize=15, color="#e0d4ff", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=10)
ax.legend(facecolor="#1a1333", edgecolor="#333", labelcolor="#c4b5fd")
ax.set_ylim(0, 110)
ax.tick_params(colors="#9ca3af")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#333")
ax.spines["bottom"].set_color("#333")
ax.set_facecolor("#0f0c29")
save_fig(fig, "f1_score_comparison.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. FEATURE IMPORTANCE (Random Forest)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("📊 4. Feature Importance...")

importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(np.array(feat_names)[sorted_idx], importances[sorted_idx],
        color=plt.cm.Purples(np.linspace(0.3, 0.9, len(feat_names))))
ax.set_xlabel("Importance", color="#9ca3af", fontsize=12)
ax.set_title("Random Forest — Feature Importance", fontsize=15, color="#e0d4ff", fontweight="bold")
ax.tick_params(colors="#9ca3af")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#333")
ax.spines["bottom"].set_color("#333")
ax.set_facecolor("#0f0c29")
save_fig(fig, "feature_importance.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. CLASS DISTRIBUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("📊 5. Class Distribution...")

full_labels = target_enc.inverse_transform(y)
counts = pd.Series(full_labels).value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(counts.index, counts.values, color=COLORS[:len(counts)], edgecolor="none", width=0.6)
for i, (cond, count) in enumerate(counts.items()):
    ax.text(i, count + 3, str(count), ha="center", va="bottom", color="#e0d4ff", fontweight="bold", fontsize=11)
ax.set_ylabel("Count", color="#9ca3af", fontsize=12)
ax.set_title("Dataset — Class Distribution", fontsize=15, color="#e0d4ff", fontweight="bold")
ax.tick_params(colors="#9ca3af")
ax.set_xticklabels(counts.index, rotation=20, ha="right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#333")
ax.spines["bottom"].set_color("#333")
ax.set_facecolor("#0f0c29")
save_fig(fig, "class_distribution.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. MODEL ARCHITECTURE SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("📊 6. MLP Architecture Diagram...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_facecolor("#0f0c29")

layers = [
    ("Input\n19 features", 1, "#60a5fa"),
    ("Dense(128)\nBN + ReLU\nDropout(0.3)", 3, "#a78bfa"),
    ("Dense(64)\nBN + ReLU\nDropout(0.3)", 5, "#c084fc"),
    ("Dense(32)\nBN + ReLU\nDropout(0.2)", 7, "#f472b6"),
    ("Output\n6 classes", 9, "#34d399"),
]

for label, x_pos, color in layers:
    rect = plt.Rectangle((x_pos - 0.6, 2.5), 1.2, 3, linewidth=2, edgecolor=color,
                          facecolor=color + "22", linestyle="-", zorder=2)
    ax.add_patch(rect)
    ax.text(x_pos, 4, label, ha="center", va="center", fontsize=9, color="#e0d4ff", fontweight="500", zorder=3)

for i in range(len(layers) - 1):
    ax.annotate("", xy=(layers[i + 1][1] - 0.6, 4), xytext=(layers[i][1] + 0.6, 4),
                arrowprops=dict(arrowstyle="->", color="#a78bfa", lw=2))

ax.text(5, 7.2, "MLP Neural Network Architecture", ha="center", fontsize=16, color="#e0d4ff", fontweight="bold")
ax.text(5, 6.5, "Multi-Layer Perceptron for Mental Health Classification", ha="center", fontsize=10, color="#9ca3af")
save_fig(fig, "mlp_architecture.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7. SAVE METRICS AS JSON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n📋 7. Saving metrics...")

metrics = {
    "dataset": {
        "total_samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "classes": class_names,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
    },
    "random_forest": {
        "accuracy": round(rf_acc * 100, 2),
        "f1_weighted": round(f1_score(y_test, rf_pred, average="weighted") * 100, 2),
        "precision_weighted": round(precision_score(y_test, rf_pred, average="weighted") * 100, 2),
        "recall_weighted": round(recall_score(y_test, rf_pred, average="weighted") * 100, 2),
        "per_class": {c: {k: round(v, 4) for k, v in rf_report[c].items()} for c in class_names},
    },
    "mlp": {
        "accuracy": round(mlp_acc * 100, 2),
        "f1_weighted": round(f1_score(y_test, mlp_pred, average="weighted") * 100, 2),
        "precision_weighted": round(precision_score(y_test, mlp_pred, average="weighted") * 100, 2),
        "recall_weighted": round(recall_score(y_test, mlp_pred, average="weighted") * 100, 2),
        "architecture": "19 → 128 → 64 → 32 → 6",
        "per_class": {c: {k: round(v, 4) for k, v in mlp_report[c].items()} for c in class_names},
    },
    "ensemble": {
        "accuracy": round(ens_acc * 100, 2),
        "f1_weighted": round(f1_score(y_test, ens_pred, average="weighted") * 100, 2),
        "precision_weighted": round(precision_score(y_test, ens_pred, average="weighted") * 100, 2),
        "recall_weighted": round(recall_score(y_test, ens_pred, average="weighted") * 100, 2),
        "per_class": {c: {k: round(v, 4) for k, v in ens_report[c].items()} for c in class_names},
    },
}

with open(os.path.join(OUTPUT_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("  📋 Saved: training_metrics.json")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  8. CLASSIFICATION REPORTS AS TEXT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("📋 8. Saving classification reports...")

report_txt = "=" * 60 + "\n"
report_txt += "  TRAINING RESULTS — Model Comparison\n"
report_txt += "=" * 60 + "\n\n"

for name, preds, acc in [("RANDOM FOREST", rf_pred, rf_acc), ("MLP NEURAL NETWORK", mlp_pred, mlp_acc), ("ENSEMBLE (RF + MLP)", ens_pred, ens_acc)]:
    report_txt += f"── {name} ──\n"
    report_txt += f"Accuracy: {acc*100:.2f}%\n\n"
    report_txt += classification_report(y_test, preds, target_names=class_names)
    report_txt += "\n\n"

with open(os.path.join(OUTPUT_DIR, "classification_reports.txt"), "w") as f:
    f.write(report_txt)
print("  📋 Saved: classification_reports.txt")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DONE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print(f"  ✅ All results saved to: {OUTPUT_DIR}/")
print("=" * 60)
print("\n📁 Files generated:")
for f_name in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f_name))
    print(f"   • {f_name:40s} ({size/1024:.1f} KB)")
