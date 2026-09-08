"""
train_transfer.py
------------------
Transfer-learning models: a frozen ImageNet-pretrained backbone
(MobileNetV2 or ResNet50) with a small trainable classification head,
followed by an optional fine-tuning phase where the top backbone layers
are unfrozen with a low learning rate.

Usage:
    python src/train_transfer.py --backbone mobilenet --epochs 15 --fine_tune_epochs 10
    python src/train_transfer.py --backbone resnet50   --epochs 15 --fine_tune_epochs 10
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

BACKBONES = {
    "mobilenet": {
        "app": tf.keras.applications.MobileNetV2,
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
        "display_name": "MobileNetV2 (transfer learning)",
    },
    "resnet50": {
        "app": tf.keras.applications.ResNet50,
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
        "display_name": "ResNet50 (transfer learning)",
    },
}


def make_datasets(preprocess_fn):
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

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (preprocess_fn(augment(x, training=True)), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_fn(x), y))
    test_ds = test_ds.map(lambda x, y: (preprocess_fn(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    return (train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE), test_ds.prefetch(AUTOTUNE))


def build_model(backbone_app, num_classes: int):
    base = backbone_app(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model, base


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
    parser.add_argument("--backbone", choices=list(BACKBONES.keys()), default="mobilenet")
    parser.add_argument("--epochs", type=int, default=15, help="epochs with frozen backbone")
    parser.add_argument("--fine_tune_epochs", type=int, default=10, help="epochs after unfreezing top layers")
    args = parser.parse_args()

    cfg = BACKBONES[args.backbone]
    train_ds, val_ds, test_ds = make_datasets(cfg["preprocess"])
    model, base = build_model(cfg["app"], len(CLASSES))

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    ]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # ---- Fine-tuning phase: unfreeze the top part of the backbone ----
    if args.fine_tune_epochs > 0:
        base.trainable = True
        # keep the first ~70% of layers frozen, fine-tune the rest
        freeze_until = int(len(base.layers) * 0.7)
        for layer in base.layers[:freeze_until]:
            layer.trainable = False

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history_ft = model.fit(train_ds, validation_data=val_ds,
                                epochs=args.fine_tune_epochs, callbacks=callbacks)
        val_acc = max(history.history["val_accuracy"] + history_ft.history["val_accuracy"])
    else:
        val_acc = max(history.history["val_accuracy"])

    train_time = time.time() - t0
    test_acc, test_f1, report, cm = evaluate_on_test(model, test_ds)
    n_params = model.count_params()

    display_name = cfg["display_name"]
    print(f"\n{display_name}  val_acc={val_acc:.3f}  test_acc={test_acc:.3f}  "
          f"test_f1={test_f1:.3f}  params={n_params:,}  train_time={train_time:.1f}s")

    out_name = f"{args.backbone}.keras"
    model.save(MODELS_DIR / out_name)

    metrics_path = RESULTS_DIR / "metrics.json"
    all_results = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_results[display_name] = {
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "train_time_sec": train_time,
        "n_params": int(n_params),
        "classification_report": report,
        "confusion_matrix": cm,
        "model_file": out_name,
        "backbone": args.backbone,
    }
    metrics_path.write_text(json.dumps(all_results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
