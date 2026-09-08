"""
train_transfer_local.py
------------------------
Same idea as train_transfer.py, but loads ImageNet weights from a LOCAL
.h5 file (pretrained_weights/mobilenet_v2_notop.h5) instead of letting
Keras auto-download them - useful in network-restricted environments.

Speed trick: since the backbone starts frozen, we run every image through
it ONCE, cache the resulting feature vectors, then train the classifier
head on those cached features (this takes seconds instead of minutes on
CPU). After that we assemble a full end-to-end model (backbone + trained
head) for saving/inference, and optionally fine-tune the top of the
backbone for a few more epochs.

Usage:
    python src/train_transfer_local.py --fine_tune_epochs 6
"""
import argparse
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, CLASSES, MODELS_DIR, RESULTS_DIR, PROJECT_ROOT

IMG_SIZE = (160, 160)          # bigger -> better features, still fast since backbone is frozen for the cached pass
WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights" / "mobilenet_v2_notop.h5"


def load_images(split_dir):
    """Load raw images + labels as a tf.data pipeline (no shuffling, deterministic order)."""
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(f))
                labels.append(idx)
    return paths, np.array(labels)


def build_backbone():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights=None, alpha=1.0,
    )
    base.load_weights(str(WEIGHTS_PATH))
    base.trainable = False
    return base


def extract_features(base, paths, augment_reps=1, augment=False):
    """Run every image through the frozen backbone once (or several augmented
    passes for training data) and return pooled feature vectors."""
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    aug_layer = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    all_feats = []
    all_rep_idx = []
    batch_size = 32

    for rep in range(augment_reps):
        feats_rep = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                img = tf.io.read_file(p)
                img = tf.io.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, IMG_SIZE)
                imgs.append(img)
            batch = tf.stack(imgs)
            if augment:
                batch = aug_layer(batch, training=True)
            batch = preprocess(batch)
            feats = base(batch, training=False)
            feats = tf.reduce_mean(feats, axis=[1, 2])  # global average pool
            feats_rep.append(feats.numpy())
        all_feats.append(np.concatenate(feats_rep, axis=0))
    return all_feats


def build_head(num_classes):
    head = models.Sequential([
        layers.Input(shape=(1280,)),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="head")
    return head


def evaluate(model_fn, X, y):
    probs = model_fn(X)
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == y))
    f1 = f1_score(y, preds, average="macro")
    report = classification_report(y, preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y, preds).tolist()
    return acc, f1, report, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_epochs", type=int, default=60)
    parser.add_argument("--fine_tune_epochs", type=int, default=6)
    parser.add_argument("--train_augment_reps", type=int, default=3,
                         help="number of augmented passes over the training set to cache")
    parser.add_argument("--class_weight", action="store_true")
    args = parser.parse_args()

    t_start = time.time()

    print("Building MobileNetV2 backbone with local ImageNet weights ...")
    base = build_backbone()

    print("Loading file lists ...")
    train_paths, y_train = load_images(TRAIN_DIR)
    val_paths, y_val = load_images(VAL_DIR)
    test_paths, y_test = load_images(TEST_DIR)
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    print(f"\nExtracting cached bottleneck features "
          f"({args.train_augment_reps}x augmented passes on train) ...")
    t0 = time.time()
    train_feats_reps = extract_features(base, train_paths, augment_reps=args.train_augment_reps, augment=True)
    X_train = np.concatenate(train_feats_reps, axis=0)
    y_train_full = np.tile(y_train, args.train_augment_reps)

    X_val = extract_features(base, val_paths, augment_reps=1, augment=False)[0]
    X_test = extract_features(base, test_paths, augment_reps=1, augment=False)[0]
    print(f"  done in {time.time() - t0:.1f}s  "
          f"(train feats: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape})")

    print("\nTraining classifier head on cached features ...")
    head = build_head(len(CLASSES))
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss"),
    ]
    class_weight = None
    if args.class_weight:
        from sklearn.utils.class_weight import compute_class_weight
        classes_arr = np.arange(len(CLASSES))
        weights = compute_class_weight("balanced", classes=classes_arr, y=y_train_full)
        class_weight = {i: w for i, w in enumerate(weights)}
        print(f"  class weights: {class_weight}")

    t0 = time.time()
    history = head.fit(X_train, y_train_full, validation_data=(X_val, y_val),
                        epochs=args.head_epochs, batch_size=64, callbacks=callbacks, verbose=2,
                        class_weight=class_weight)
    head_time = time.time() - t0
    val_acc_head = max(history.history["val_accuracy"])
    print(f"  head training done in {head_time:.1f}s, best val_acc={val_acc_head:.3f}")

    # ---- Assemble the full end-to-end model (backbone + trained head) ----
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = head(x)
    full_model = models.Model(inputs, outputs, name="mobilenetv2_transfer")

    # ---- Optional fine-tuning of the top of the backbone, end-to-end ----
    fine_tune_time = 0.0
    if args.fine_tune_epochs > 0:
        print(f"\nFine-tuning top of backbone for {args.fine_tune_epochs} epochs (end-to-end, low LR) ...")
        base.trainable = True
        freeze_until = int(len(base.layers) * 0.75)
        for layer in base.layers[:freeze_until]:
            layer.trainable = False

        full_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                            loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

        def make_ds(paths, labels, training):
            def _load(p, y):
                img = tf.io.read_file(p)
                img = tf.io.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, IMG_SIZE)
                return img, y
            ds = tf.data.Dataset.from_tensor_slices((paths, labels))
            ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
            if training:
                aug = tf.keras.Sequential([
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.06),
                    layers.RandomZoom(0.1),
                ])
                ds = ds.shuffle(512, seed=42)
                ds = ds.map(lambda x, y: (preprocess(aug(x, training=True)), y))
            else:
                ds = ds.map(lambda x, y: (preprocess(x), y))
            return ds.batch(32).prefetch(tf.data.AUTOTUNE)

        train_ds_ft = make_ds(train_paths, y_train, training=True)
        val_ds_ft = make_ds(val_paths, y_val, training=False)

        t0 = time.time()
        history_ft = full_model.fit(train_ds_ft, validation_data=val_ds_ft,
                                     epochs=args.fine_tune_epochs,
                                     callbacks=[tf.keras.callbacks.EarlyStopping(
                                         patience=4, restore_best_weights=True, monitor="val_accuracy")])
        fine_tune_time = time.time() - t0
        val_acc = max([val_acc_head] + history_ft.history["val_accuracy"])
    else:
        val_acc = val_acc_head

    # ---- Final evaluation on the untouched test set ----
    def predict_fn(X_or_paths):
        # X_test/X_val are cached features -> not usable post fine-tune (backbone changed),
        # so re-run raw images through the now-updated full_model for the final test score.
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        probs_all = []
        for i in range(0, len(test_paths), 32):
            batch_paths = test_paths[i:i + 32]
            imgs = []
            for p in batch_paths:
                img = tf.io.read_file(p)
                img = tf.io.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, IMG_SIZE)
                imgs.append(img)
            batch = preprocess(tf.stack(imgs))
            probs_all.append(full_model(batch, training=False).numpy())
        return np.concatenate(probs_all, axis=0)

    print("\nEvaluating final model on the held-out test set ...")
    test_probs = predict_fn(None)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = float(np.mean(test_preds == y_test))
    test_f1 = f1_score(y_test, test_preds, average="macro")
    report = classification_report(y_test, test_preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_test, test_preds).tolist()

    total_time = time.time() - t_start
    n_params = full_model.count_params()

    print(f"\n{'='*60}")
    print("MobileNetV2 (transfer learning, local weights)")
    print(f"  best val_accuracy : {val_acc:.4f}")
    print(f"  TEST accuracy     : {test_acc:.4f}")
    print(f"  TEST F1 (macro)   : {test_f1:.4f}")
    print(f"  params            : {n_params:,}")
    print(f"  total time        : {total_time:.1f}s "
          f"(features={time.time()-t_start-head_time-fine_tune_time:.0f}s, "
          f"head={head_time:.0f}s, fine-tune={fine_tune_time:.0f}s)")
    print(f"{'='*60}")

    full_model.save(MODELS_DIR / "mobilenet.keras")

    metrics_path = RESULTS_DIR / "metrics.json"
    all_results = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_results["MobileNetV2 (transfer learning)"] = {
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "train_time_sec": total_time,
        "n_params": int(n_params),
        "classification_report": report,
        "confusion_matrix": cm,
        "model_file": "mobilenet.keras",
        "backbone": "mobilenet",
    }
    metrics_path.write_text(json.dumps(all_results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
