import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
import argparse

# ----------------------------
# Configuration (portable paths)
# ----------------------------

script_dir = Path(__file__).parent

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    type=str,
    default=str(script_dir / "project2_raw_data"),
    help="Directory containing raw .txt files"
)

parser.add_argument(
    "--feature_map",
    type=str,
    default=str(script_dir / "feature_name_to_number_mapping.csv"),
    help="CSV mapping feature numbers to names"
)

parser.add_argument(
    "--output_model",
    type=str,
    default=str(script_dir / "application_risk_model.pkl"),
    help="Output model filename"
)

args = parser.parse_args()

RAW_DATA_DIR = Path(args.data_dir)
FEATURE_MAP_FILE = Path(args.feature_map)
MODEL_OUTPUT = Path(args.output_model)

THRESHOLD = 0.45
RANDOM_STATE = 42

# ----------------------------
# Load feature mapping
# ----------------------------

feature_map = pd.read_csv(FEATURE_MAP_FILE).sort_values("feature_number")

FEATURE_COUNT = feature_map.shape[0]
FEATURE_NAMES = feature_map["feature_name"].tolist()

# ----------------------------
# Load raw data
# ----------------------------

def load_raw_file(file_path):

    X = []
    y = []

    with open(file_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 2:
                continue

            risk_score = float(parts[0])
            label = int(risk_score >= 0.3)

            y.append(label)

            features = np.zeros(FEATURE_COUNT, dtype=np.float32)

            for item in parts[1:]:

                idx, val = item.split(":")

                idx = int(idx)

                if idx < FEATURE_COUNT:
                    features[idx] = float(val)

            X.append(features)

    return np.array(X), np.array(y)

# ----------------------------
# Combine dataset
# ----------------------------

X_all = []
y_all = []

for file in RAW_DATA_DIR.glob("*.txt"):

    X_part, y_part = load_raw_file(file)

    if len(X_part) > 0:
        X_all.append(X_part)
        y_all.append(y_part)

X = np.vstack(X_all)
y = np.concatenate(y_all)

X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

print("Dataset shape:", X_df.shape)
print("Positive class ratio:", y.mean())

# ----------------------------
# Data cleaning
# ----------------------------

# combine features and labels to keep them aligned
data_df = X_df.copy()
data_df["label"] = y

# remove duplicates
data_df = data_df.drop_duplicates()

# replace infinite values
data_df = data_df.replace([np.inf, -np.inf], np.nan)

# fill missing values
data_df = data_df.fillna(0)

# clip extreme outliers
data_df = data_df.clip(-1e6, 1e6)

# separate X and y again
y = data_df["label"].values
X = data_df.drop(columns=["label"]).values

# ----------------------------
# Train / Validation / Test split
# ----------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

# ----------------------------
# Model training
# ----------------------------

model = XGBClassifier(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1,
    scale_pos_weight=1.2,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

# ----------------------------
# Validation evaluation
# ----------------------------

y_proba_val = model.predict_proba(X_val)[:, 1]
y_pred_val = (y_proba_val >= THRESHOLD).astype(int)

precision_val = precision_score(y_val, y_pred_val)
recall_val = recall_score(y_val, y_pred_val)

print("\nValidation Results")
print("Precision:", precision_val)
print("Recall:", recall_val)

print(classification_report(y_val, y_pred_val))

# ----------------------------
# Test evaluation
# ----------------------------

y_proba_test = model.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= THRESHOLD).astype(int)

precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)

print("\nTest Results")
print("Precision:", precision_test)
print("Recall:", recall_test)

print(classification_report(y_test, y_pred_test))

# ----------------------------
# Save evaluation report
# ----------------------------

with open("model_evaluation.txt", "w") as f:

    f.write("Validation Precision: " + str(precision_val) + "\n")
    f.write("Validation Recall: " + str(recall_val) + "\n\n")

    f.write("Test Precision: " + str(precision_test) + "\n")
    f.write("Test Recall: " + str(recall_test) + "\n\n")

    f.write("Validation Classification Report:\n")
    f.write(classification_report(y_val, y_pred_val))

    f.write("\nTest Classification Report:\n")
    f.write(classification_report(y_test, y_pred_test))

print("Evaluation report saved to model_evaluation.txt")

# ----------------------------
# Feature importance plot
# ----------------------------

importances = model.feature_importances_

top_idx = np.argsort(importances)[-20:]

plt.figure(figsize=(10,6))

plt.barh(range(len(top_idx)), importances[top_idx])

plt.yticks(range(len(top_idx)), [FEATURE_NAMES[i] for i in top_idx])

plt.title("Top 20 Feature Importances")

plt.tight_layout()

plt.savefig("feature_importance.png")

print("Feature importance plot saved as feature_importance.png")

# ----------------------------
# Save model
# ----------------------------

package = {
    "model": model,
    "feature_names": FEATURE_NAMES,
    "threshold": THRESHOLD
}

with open(MODEL_OUTPUT, "wb") as f:
    pickle.dump(package, f)

print("Model package saved to:", MODEL_OUTPUT)