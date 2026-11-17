"""Classify patient nanopore events with the unchanged original Random Forest model.

Place this script in 6_Clinical_Event_Classification and patient CSV/XLSX files in
5_Patient_Event_Feature_Matrices. Outputs are written to
6_Clinical_Event_Classification/results. No model fitting or modification occurs.
"""
from pathlib import Path
import csv
import hashlib
import joblib
import numpy as np
import pandas as pd

CONFIDENCE_THRESHOLD = 0.90
FEATURE_MAP = {
    "Relative_Blockade": "\u76f8\u5bf9\u6df1\u5ea6",
    "Current_SD": "\u566a\u97f3(SD)",
    "Current_Range": "\u632f\u5e45",
}
CLASS_MAP = {"D-2-HG.xlsx": "D-2-HG", "\u975e\u7a81\u53d8\u4e09\u7ec4N.xlsx": "Others"}


def project_root():
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "6_Clinical_Event_Classification":
        return script_dir.parent
    if (script_dir / "4_Trained_Models.joblib").is_file():
        return script_dir
    raise FileNotFoundError(
        "Place this script in the 6_Clinical_Event_Classification folder "
        "inside the project, or directly in the project root."
    )


def load_patient_table(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.read_excel(path)


def main():
    root = project_root()
    model_path = root / "4_Trained_Models.joblib"
    patient_dir = root / "5_Patient_Event_Feature_Matrices"
    output_dir = root / "6_Clinical_Event_Classification" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient feature folder not found: {patient_dir}")
    patient_files = sorted(
        p for p in patient_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
        and not p.name.startswith("~$")
    )
    if not patient_files:
        raise FileNotFoundError(f"No patient CSV/XLSX feature matrices in {patient_dir}")

    model_sha256_before = hashlib.sha256(model_path.read_bytes()).hexdigest()
    bundle = joblib.load(model_path)  # Only load a trusted model file.
    model = bundle["model"]
    original_features = list(bundle["feature_cols"])
    expected_features = list(FEATURE_MAP.values())
    if original_features != expected_features:
        raise ValueError(f"Unexpected saved model feature order: {original_features}")
    classes = list(model.classes_)
    if set(classes) != set(CLASS_MAP):
        raise ValueError(f"Unexpected saved model classes: {classes}")
    positive_index = classes.index("D-2-HG.xlsx")

    summary = []
    for path in patient_files:
        df = load_patient_table(path)
        missing = [name for name in FEATURE_MAP if name not in df.columns]
        if missing:
            raise KeyError(f"{path.name} missing feature columns: {missing}")
        # Keep the exact feature names and order used when fitting the original model.
        X = df[list(FEATURE_MAP)].rename(columns=FEATURE_MAP)
        X = X[original_features].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite input feature value in {path.name}")
        if "IsNoise" in df.columns:
            valid = pd.to_numeric(df["IsNoise"], errors="raise") != -1
        else:
            valid = pd.Series(True, index=df.index)
        result = df.copy()
        result["Predicted_Class"] = "Excluded_Noise"
        result["D-2-HG_Probability"] = np.nan
        result["Prediction_Confidence"] = np.nan
        result["Confidence_Filtered_Class"] = "Excluded_Noise"
        if valid.any():
            selected = X.loc[valid]
            raw = model.predict(selected)
            probabilities = model.predict_proba(selected)
            max_probability = probabilities.max(axis=1)
            predicted = [CLASS_MAP[x] for x in raw]
            filtered = [label if conf >= CONFIDENCE_THRESHOLD else "Unclassified"
                        for label, conf in zip(predicted, max_probability)]
            result.loc[valid, "Predicted_Class"] = predicted
            result.loc[valid, "D-2-HG_Probability"] = probabilities[:, positive_index]
            result.loc[valid, "Prediction_Confidence"] = max_probability
            result.loc[valid, "Confidence_Filtered_Class"] = filtered
        output_path = output_dir / f"{path.stem}_classified.csv"
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        counts = result.loc[valid, "Confidence_Filtered_Class"].value_counts()
        n_valid = int(valid.sum())
        summary.append({
            "Sample_File": path.name,
            "Total_Events": len(df),
            "Analyzed_Events": n_valid,
            "Excluded_Noise_Events": int((~valid).sum()),
            "D-2-HG_Events": int(counts.get("D-2-HG", 0)),
            "Others_Events": int(counts.get("Others", 0)),
            "Unclassified_Events": int(counts.get("Unclassified", 0)),
            "D-2-HG_Fraction_of_Analyzed_Events":
                float(counts.get("D-2-HG", 0) / n_valid) if n_valid else np.nan,
            "Confidence_Threshold": CONFIDENCE_THRESHOLD,
        })
        print(f"Classified: {path.name} ({n_valid} analyzed events)")
    pd.DataFrame(summary).to_csv(output_dir / "patient_event_summary.csv", index=False)
    model_sha256_after = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_sha256_after != model_sha256_before:
        raise RuntimeError("Model file unexpectedly changed")
    print(f"Results saved to: {output_dir}")
    print(f"Original model unchanged (SHA-256: {model_sha256_after})")


if __name__ == "__main__":
    main()
