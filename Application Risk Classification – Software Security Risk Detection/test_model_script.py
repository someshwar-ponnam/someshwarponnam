import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def load_model_package(model_path):

    with open(model_path, "rb") as f:
        package = pickle.load(f)

    model = package["model"]
    feature_names = package["feature_names"]
    threshold = package["threshold"]

    print("Model loaded successfully")
    print("Expected feature count:", len(feature_names))

    return model, feature_names, threshold


def load_input_csv(csv_path):

    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    data = pd.read_csv(csv_path)

    print("Input samples:", len(data))

    return data


def align_features(data, feature_names):

    # add missing columns
    missing_cols = [c for c in feature_names if c not in data.columns]

    if missing_cols:
        missing_df = pd.DataFrame(
            np.zeros((len(data), len(missing_cols))),
            columns=missing_cols
        )
        data = pd.concat([data, missing_df], axis=1)

    # remove extra columns
    data = data[feature_names]

    return data


def run_predictions(model, data, threshold):

    probabilities = model.predict_proba(data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print("\nPrediction Results\n")

    for i, (pred, prob) in enumerate(zip(predictions, probabilities), start=1):

        label = "High Risk" if pred == 1 else "Low Risk"

        print(f"Sample {i}: {label} (Probability: {prob:.3f})")


def main():

    parser = argparse.ArgumentParser(description="Test packaged ML model")

    parser.add_argument(
        "--model",
        default="application_risk_model.pkl",
        help="Path to model package"
    )

    parser.add_argument(
        "--input",
        default="sample_input.csv",
        help="Input CSV file"
    )

    args = parser.parse_args()

    #model_path = Path(args.model)
    script_dir = Path(__file__).parent
    model_path = script_dir / args.model
    #input_path = Path(args.input)
    input_path = script_dir / args.input

    model, feature_names, threshold = load_model_package(model_path)

    data = load_input_csv(input_path)

    data = align_features(data, feature_names)

    run_predictions(model, data, threshold)


if __name__ == "__main__":
    main()