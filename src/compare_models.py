"""
compare_models.py
------------------
Reads results/metrics.json (populated by train_baseline.py, train_cnn.py,
and train_transfer.py) and produces:
  - a printed comparison table
  - results/comparison_table.csv
  - results/comparison_chart.png (accuracy / F1 bar chart)

Usage:
    python src/compare_models.py
"""
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR


def main():
    metrics_path = RESULTS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "No results/metrics.json found. Run at least one training script first "
            "(e.g. python src/train_baseline.py)."
        )

    all_results = json.loads(metrics_path.read_text())

    rows = []
    for name, m in all_results.items():
        rows.append({
            "Model": name,
            "Val Accuracy": round(m["val_accuracy"], 4),
            "Test Accuracy": round(m["test_accuracy"], 4),
            "Test F1 (macro)": round(m["test_f1_macro"], 4),
            "Train Time (s)": round(m["train_time_sec"], 1),
            "# Params": m.get("n_params"),
        })

    df = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False)
    print(df.to_string(index=False))

    csv_path = RESULTS_DIR / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved table to {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    ax.bar(x, df["Test Accuracy"], width=0.4, label="Test Accuracy", align="edge")
    ax.bar([i + 0.4 for i in x], df["Test F1 (macro)"], width=0.4, label="Test F1 (macro)", align="edge")
    ax.set_xticks([i + 0.4 for i in x])
    ax.set_xticklabels(df["Model"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison — trash image classification")
    ax.legend()
    fig.tight_layout()

    chart_path = RESULTS_DIR / "comparison_chart.png"
    fig.savefig(chart_path, dpi=150)
    print(f"Saved chart to {chart_path}")


if __name__ == "__main__":
    main()
