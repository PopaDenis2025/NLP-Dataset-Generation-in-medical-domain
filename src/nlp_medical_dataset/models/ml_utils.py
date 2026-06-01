import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_medical_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["symptom", "disease"])
    df["symptom"] = df["symptom"].astype(str)
    df["disease"] = df["disease"].astype(str)
    return df


def split_dataset(df: pd.DataFrame):
    stratify = df["disease"] if df["disease"].value_counts().min() >= 2 else None
    return train_test_split(
        df["symptom"],
        df["disease"],
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )


def evaluate_model(model, x_test, y_test) -> Dict[str, Any]:
    predictions = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def save_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
