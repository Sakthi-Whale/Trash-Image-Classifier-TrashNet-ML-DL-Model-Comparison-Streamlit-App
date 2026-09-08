"""
train_baseline.py
------------------
Classical machine-learning baselines for trash image classification,
using HOG (Histogram of Oriented Gradients) features + SVM / Random Forest.

These are fast to train on CPU (no GPU needed) and give a useful lower
bound to compare the deep-learning models against.

Usage:
    python src/train_baseline.py
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, CLASSES, MODELS_DIR, RESULTS_DIR

FEATURE_IMG_SIZE = (128, 128)  # smaller size just for HOG features -> faster


def load_split(split_dir: Path):
    X, y = [], []
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = Image.open(f).convert("RGB").resize(FEATURE_IMG_SIZE)
            X.append(extract_hog(np.array(img)))
            y.append(label_idx)
    return np.array(X), np.array(y)


def extract_hog(img_array: np.ndarray) -> np.ndarray:
    gray = rgb2gray(img_array)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features


def evaluate(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    report = classification_report(y, preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y, preds).tolist()
    return acc, f1, report, cm, preds


def main():
    print("Loading images and extracting HOG features ...")
    t0 = time.time()
    X_train, y_train = load_split(TRAIN_DIR)
    X_val, y_val = load_split(VAL_DIR)
    X_test, y_test = load_split(TEST_DIR)
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}  "
          f"({time.time() - t0:.1f}s)")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = {}

    # ---- Model 1: SVM (RBF kernel) ----
    print("\nTraining SVM (RBF kernel) ...")
    t0 = time.time()
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train_s, y_train)
    train_time = time.time() - t0

    val_acc, val_f1, _, _, _ = evaluate(svm, X_val_s, y_val)
    test_acc, test_f1, report, cm, _ = evaluate(svm, X_test_s, y_test)
    print(f"  SVM  val_acc={val_acc:.3f}  test_acc={test_acc:.3f}  "
          f"test_f1={test_f1:.3f}  train_time={train_time:.1f}s")

    joblib.dump({"model": svm, "scaler": scaler}, MODELS_DIR / "hog_svm.joblib")
    results["HOG + SVM"] = {
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "train_time_sec": train_time,
        "n_params": None,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    # ---- Model 2: Random Forest ----
    print("\nTraining Random Forest ...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    rf.fit(X_train_s, y_train)
    train_time = time.time() - t0

    val_acc, val_f1, _, _, _ = evaluate(rf, X_val_s, y_val)
    test_acc, test_f1, report, cm, _ = evaluate(rf, X_test_s, y_test)
    print(f"  RF   val_acc={val_acc:.3f}  test_acc={test_acc:.3f}  "
          f"test_f1={test_f1:.3f}  train_time={train_time:.1f}s")

    joblib.dump({"model": rf, "scaler": scaler}, MODELS_DIR / "hog_rf.joblib")
    results["HOG + Random Forest"] = {
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "train_time_sec": train_time,
        "n_params": None,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    # Merge into shared results/metrics.json (used by streamlit app + compare_models.py)
    metrics_path = RESULTS_DIR / "metrics.json"
    all_results = {}
    if metrics_path.exists():
        all_results = json.loads(metrics_path.read_text())
    all_results.update(results)
    metrics_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
