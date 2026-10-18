"""One-time holdout evaluation with reproducible quantitative and qualitative errors."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate(data_dir="data", model_path="models/classifier.joblib", out="results"):
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(Path(data_dir)/"test.csv").dropna()
    model = joblib.load(model_path)  # Only load trusted models that you trained.
    predictions = model.predict(test.text)
    labels = sorted(set(test.label.astype(str)).union(map(str, predictions)))
    metrics = {"test_accuracy": float(accuracy_score(test.label, predictions)), "test_macro_f1": float(f1_score(test.label, predictions, average="macro", zero_division=0)), "test_samples": len(test)}
    (dest/"test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(test.label, predictions, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(dest/"classification_report.csv")
    errors = test.copy()
    errors["prediction"] = predictions
    errors["correct"] = errors.label == errors.prediction
    errors.to_csv(dest/"per_example_predictions.csv", index=False)
    matrix = confusion_matrix(test.label, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(dest/"confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    evaluate()
