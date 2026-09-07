"""
Flash Flood Crowd-Info: Photo Severity Classifier
==================================================
Trains a CNN (EfficientNetB0 transfer learning) to classify crowd-uploaded
disaster photos into: Normal / Warning / Severe.

Based on the MEDIC dataset (QCRI) damage_severity labels:
    little_or_none -> Normal
    mild           -> Warning
    severe         -> Severe

Usage:
    1. Prepare data (see prepare_data_from_hf() below), OR point --data_dir
       to a folder already structured as:
           data/
             train/
               normal/*.jpg
               warning/*.jpg
               severe/*.jpg
             val/
               normal/*.jpg
               warning/*.jpg
               severe/*.jpg

    2. Run:
           python train_severity_model.py --data_dir data --epochs 10

    3. Outputs:
           severity_model.keras   (full Keras model)
           severity_model.tflite  (lightweight export for deployment)
           class_names.json       (label order used by the model)

Requirements:
    pip install "tensorflow[and-cuda]" datasets pillow scikit-learn tqdm
"""

import os
import json
import argparse
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf


# --------------------------------------------------------------------------
# 0. GPU check
# --------------------------------------------------------------------------
def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO] {len(gpus)} GPU(s) detected: {[g.name for g in gpus]}")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError as e:
                print(f"[WARN] Could not set memory growth: {e}")
    else:
        print("[WARN] No GPU detected. Training will run on CPU (much slower).")
        print("       If you expected a GPU, check `nvidia-smi` and your "
              "tensorflow[and-cuda] / CUDA install.")
    return len(gpus) > 0


# --------------------------------------------------------------------------
# 1. Optional: build the folder structure directly from the HF MEDIC dataset
# --------------------------------------------------------------------------
def prepare_data_from_hf(output_dir="data", val_split=0.15, test_split=0.15,
                          seed=42, limit=None, split_manifest="test_split.json"):
    """
    Downloads QCRI/MEDIC from Hugging Face, filters to rows that have a
    damage_severity label, remaps to normal/warning/severe, and writes
    images into output_dir/{train,val,test}/<class>/.

    A held-out TEST set is carved out first and its filenames are written to
    `split_manifest` so evaluation (test_severity_model.py) can always
    reload the exact same test set later, even in a fresh Colab session,
    as long as this manifest travels with the repo/Drive folder.

    Requires: pip install datasets pillow
    """
    from datasets import load_dataset

    print("[INFO] Loading QCRI/MEDIC from Hugging Face (this can take a while)...")
    ds = load_dataset("QCRI/MEDIC")
    split_data = ds["train"]

    # Current QCRI/MEDIC damage_severity values.
    # IMPORTANT: these names must match the Hugging Face dataset exactly.
    # MEDIC exposes damage_severity as a Hugging Face ClassLabel.
    # The stored values are integer IDs:
    #   0 = little_or_none -> normal
    #   1 = mild           -> warning
    #   2 = severe         -> severe
    label_map = {
        0: "normal",
        1: "warning",
        2: "severe",
    }

    # Keep only rows with a usable damage_severity ClassLabel ID.
    split_data = split_data.filter(
        lambda ex: ex.get("damage_severity") in label_map
    )

    if len(split_data) == 0:
        raise RuntimeError(
            "No MEDIC rows matched damage_severity IDs 0, 1, 2. "
            "The dataset labels may have changed."
        )
    print(f"[INFO] Matched {len(split_data)} MEDIC images with severity labels.")

    if limit:
        split_data = split_data.select(range(min(limit, len(split_data))))

    split_data = split_data.shuffle(seed=seed)
    n = len(split_data)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test_ds = split_data.select(range(0, n_test))
    val_ds = split_data.select(range(n_test, n_test + n_val))
    train_ds = split_data.select(range(n_test + n_val, n))

    manifest = {"normal": [], "warning": [], "severe": []}

    for subset_name, subset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        for cls in label_map.values():
            Path(f"{output_dir}/{subset_name}/{cls}").mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Writing {subset_name} split ({len(subset)} images)...")
        for i, ex in enumerate(subset):
            cls = label_map[int(ex["damage_severity"])]
            img = ex["image"]  # PIL.Image object in HF datasets image feature
            if img.mode != "RGB":
                img = img.convert("RGB")
            filename = f"{i}.jpg"
            out_path = f"{output_dir}/{subset_name}/{cls}/{filename}"
            img.save(out_path, "JPEG", quality=90)

            if subset_name == "test":
                manifest[cls].append(filename)

    with open(split_manifest, "w") as f:
        json.dump({"seed": seed, "test_split": test_split, "files": manifest}, f, indent=2)
    print(f"[INFO] Test-set manifest saved to '{split_manifest}' "
          f"({n_test} images) — keep this file with your model checkpoint "
          f"so evaluation always uses the exact same held-out test set.")

    print(f"[INFO] Data prepared at '{output_dir}/' "
          f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}).")


# --------------------------------------------------------------------------
# 2. Data loading
# --------------------------------------------------------------------------
def load_datasets(data_dir, img_size=(224, 224), batch_size=32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"[INFO] Classes found (index order): {class_names}")

    # Prefetch for GPU throughput
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


def compute_class_weights(data_dir, class_names):
    """Handle class imbalance -- Severe is usually the rarest but most
    important class to catch."""
    from sklearn.utils.class_weight import compute_class_weight

    labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, "train", cls)
        n = len([f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        labels.extend([idx] * n)
    labels = np.array(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    class_weight_dict = {i: w for i, w in enumerate(weights)}
    print(f"[INFO] Class weights: {class_weight_dict}")
    return class_weight_dict


# --------------------------------------------------------------------------
# 3. Model
# --------------------------------------------------------------------------
def build_model(num_classes, img_size=(224, 224)):
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = augment(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


# --------------------------------------------------------------------------
# 4. Train
# --------------------------------------------------------------------------
def train(args):
    check_gpu()

    if args.prepare_from_hf:
        prepare_data_from_hf(
            output_dir=args.data_dir,
            limit=args.hf_limit,
            split_manifest=args.split_manifest,
        )

    train_ds, val_ds, class_names = load_datasets(
        args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size
    )

    class_weight_dict = None
    if args.use_class_weights:
        class_weight_dict = compute_class_weights(args.data_dir, class_names)

    model, base_model = build_model(
        num_classes=len(class_names), img_size=(args.img_size, args.img_size)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_frozen.keras", monitor="val_accuracy", save_best_only=True
        ),
    ]

    print("\n[STAGE 1] Training with frozen backbone...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    # ---- Fine-tuning stage ----
    if args.fine_tune_epochs > 0:
        print("\n[STAGE 2] Fine-tuning top layers of backbone...")
        base_model.trainable = True
        for layer in base_model.layers[:-args.unfreeze_layers]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        ft_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "best_finetuned.keras", monitor="val_accuracy", save_best_only=True
            ),
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weight_dict,
            callbacks=ft_callbacks,
        )

    # ---- Evaluate ----
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\n[RESULT] Final validation accuracy: {val_acc:.4f}")

    # ---- Save outputs ----
    model.save(args.output_model)
    print(f"[INFO] Saved Keras model to {args.output_model}")

    with open(args.class_names_file, "w") as f:
        json.dump(class_names, f)
    print(f"[INFO] Saved class names to {args.class_names_file}")

    if args.export_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = args.output_model.replace(".keras", ".tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"[INFO] Saved TFLite model to {tflite_path}")

    return model, class_names


# --------------------------------------------------------------------------
# 5. Inference helper (use this in your Flask/FastAPI backend)
# --------------------------------------------------------------------------
def predict_severity(model_path, class_names_file, image_path, img_size=(224, 224)):
    model = tf.keras.models.load_model(model_path)
    with open(class_names_file) as f:
        class_names = json.load(f)

    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0)

    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    return {
        "label": class_names[idx],
        "confidence": float(preds[idx]),
        "all_scores": {class_names[i]: float(p) for i, p in enumerate(preds)},
    }


# --------------------------------------------------------------------------
# 6. CLI
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train flash-flood photo severity classifier")
    p.add_argument("--data_dir", type=str, default="data",
                    help="Folder with train/ and val/ subfolders of class images")
    p.add_argument("--prepare_from_hf", action="store_true",
                    help="Download and prepare data from QCRI/MEDIC on Hugging Face first")
    p.add_argument("--hf_limit", type=int, default=None,
                    help="Limit number of HF examples used (for quick testing)")
    p.add_argument("--split_manifest", type=str, default="test_split.json",
                    help="Where to save/read the held-out test-set file manifest")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10, help="Frozen-backbone epochs")
    p.add_argument("--fine_tune_epochs", type=int, default=5,
                    help="Fine-tuning epochs (0 to skip fine-tuning stage)")
    p.add_argument("--unfreeze_layers", type=int, default=20,
                    help="Number of final backbone layers to unfreeze for fine-tuning")
    p.add_argument("--use_class_weights", action="store_true", default=True)
    p.add_argument("--output_model", type=str, default="severity_model.keras")
    p.add_argument("--class_names_file", type=str, default="class_names.json")
    p.add_argument("--export_tflite", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
