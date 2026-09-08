"""
train_cnn.py
------------
Trains a small convolutional neural network FROM SCRATCH (no pretrained
weights) as one of the models in the comparison.

Needs TensorFlow/Keras. Recommended to run on a machine with a GPU
(Colab is fine) - on CPU it will still work but will be slower.

Usage:
    python src/train_cnn.py --epochs 25
"""
import argparse
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, CLASSES, IMG_SIZE, BATCH_SIZE, MODELS_DIR, RESULTS_DIR

tf.random.set_seed(42)


def make_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels="inferred", label_mode="int", class_names=CLASSES,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True, seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, labels="inferred", label_mode="int", class_names=CLASSES,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, labels="inferred", label_mode="int", class_names=CLASSES,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False,
    )

    normalize = layers.Rescaling(1.0 / 255)
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (augment(normalize(x), training=True), y))
    val_ds = val_ds.map(lambda x, y: (normalize(x), y))
    test_ds = test_ds.map(lambda x, y: (normalize(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    return (train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE), test_ds.prefetch(AUTOTUNE))


def build_model(num_classes: int):
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="simple_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_on_test(model, test_ds):
    y_true, y_pred = [], []
    for x, y in test_ds:
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(y.numpy())
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return acc, f1, report, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    train_ds, val_ds, test_ds = make_datasets()
    model = build_model(len(CLASSES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
    ]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    train_time = time.time() - t0

    val_acc = max(history.history["val_accuracy"])
    test_acc, test_f1, report, cm = evaluate_on_test(model, test_ds)
    n_params = model.count_params()

    print(f"\nSimpleCNN  val_acc={val_acc:.3f}  test_acc={test_acc:.3f}  "
          f"test_f1={test_f1:.3f}  params={n_params:,}  train_time={train_time:.1f}s")

    model.save(MODELS_DIR / "simple_cnn.keras")

    metrics_path = RESULTS_DIR / "metrics.json"
    all_results = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_results["Simple CNN (from scratch)"] = {
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "train_time_sec": train_time,
        "n_params": int(n_params),
        "classification_report": report,
        "confusion_matrix": cm,
    }
    metrics_path.write_text(json.dumps(all_results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
