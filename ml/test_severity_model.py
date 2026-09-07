"""
Test / evaluate the trained crowd-photo severity classifier on held-out
MEDIC test images.

What this does:
1. Reloads the exact same held-out test set used at data-prep time, via
   test_split.json (the manifest written by train_severity_model.py's
   --prepare_from_hf step), so we evaluate on images the model never
   trained or validated on.
2. Reloads the saved Keras model and class_names.json.
3. Reports overall accuracy plus per-class precision/recall/F1 and a
   confusion matrix (Severe recall is the number that matters most for
   this feature -- missing a Severe photo is the costly failure mode).
4. Saves a grid of example predictions (image | true label | predicted
   label | confidence) to test_predictions/, split into "correct" and
   "misclassified" so you can eyeball failure cases.

Run:
    python test_severity_model.py \
        --data_dir data \
        --model_path severity_model.keras \
        --class_names_file class_names.json \
        --split_manifest test_split.json

Requirements:
    pip install tensorflow scikit-learn matplotlib
"""

import os
import json
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Load the exact test set used during data prep
# ---------------------------------------------------------------------------
def load_test_manifest(data_dir, split_manifest):
    if not os.path.exists(split_manifest):
        raise FileNotFoundError(
            f"'{split_manifest}' not found. This file is written automatically "
            f"by train_severity_model.py's --prepare_from_hf step. Without it "
            f"we can't guarantee we're evaluating on the exact same held-out "
            f"test images the model never saw during training -- copy it over "
            f"from wherever you ran data prep (e.g. your Drive folder)."
        )

    with open(split_manifest) as f:
        manifest = json.load(f)

    file_label_pairs = []
    for cls, filenames in manifest["files"].items():
        for fname in filenames:
            path = os.path.join(data_dir, "test", cls, fname)
            if os.path.exists(path):
                file_label_pairs.append((path, cls))
            else:
                print(f"[WARN] Missing test file (skipped): {path}")

    print(f"[INFO] Loaded {len(file_label_pairs)} test images from manifest "
          f"(seed={manifest.get('seed')}, test_split={manifest.get('test_split')}).")
    return file_label_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.class_names_file) as f:
        class_names = json.load(f)
    print(f"[INFO] Classes (index order): {class_names}")
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    test_pairs = load_test_manifest(args.data_dir, args.split_manifest)
    if not test_pairs:
        raise RuntimeError("No test images found -- check --data_dir points at "
                            "the same folder used for data prep.")

    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(args.model_path)

    y_true, y_pred, confidences, image_paths = [], [], [], []

    print("[INFO] Running inference on test set...")
    for path, true_cls in test_pairs:
        img = tf.keras.utils.load_img(path, target_size=(args.img_size, args.img_size))
        arr = tf.keras.utils.img_to_array(img)
        arr = tf.expand_dims(arr, 0)

        preds = model.predict(arr, verbose=0)[0]
        pred_idx = int(np.argmax(preds))

        y_true.append(class_to_idx[true_cls])
        y_pred.append(pred_idx)
        confidences.append(float(preds[pred_idx]))
        image_paths.append(path)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- Overall + per-class metrics ---
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    print(f"\n[RESULT] Overall test accuracy: {acc:.4f}\n")
    print(f"{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Support':<10}")
    for i, cls in enumerate(class_names):
        print(f"{cls:<10}{precision[i]:<12.3f}{recall[i]:<12.3f}{f1[i]:<12.3f}{support[i]:<10}")

    severe_idx = class_to_idx.get("severe")
    if severe_idx is not None:
        print(f"\n[NOTE] 'Severe' recall = {recall[severe_idx]:.3f} -- this is the "
              f"most important number for this feature: it's the fraction of "
              f"genuinely severe photos the model actually flags. If low, "
              f"prioritize more Severe examples / higher class weight over "
              f"chasing overall accuracy.")

    print("\n[INFO] Full classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    print("[INFO] Confusion matrix (rows=true, cols=predicted):")
    print(f"{'':<10}" + "".join(f"{c:<10}" for c in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<10}" + "".join(f"{v:<10}" for v in row))

    _plot_confusion_matrix(cm, class_names, args.output_dir)

    # --- Save example prediction grids ---
    _save_example_grid(
        image_paths, y_true, y_pred, confidences, class_names,
        correct=True, out_dir=args.output_dir, n=args.num_examples,
    )
    _save_example_grid(
        image_paths, y_true, y_pred, confidences, class_names,
        correct=False, out_dir=args.output_dir, n=args.num_examples,
    )

    print(f"\n[INFO] Done. See '{args.output_dir}/' for confusion matrix "
          f"and example prediction images.")


def _plot_confusion_matrix(cm, class_names, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Saved {path}")


def _save_example_grid(image_paths, y_true, y_pred, confidences, class_names,
                        correct, out_dir, n=6):
    mask = (y_true == y_pred) if correct else (y_true != y_pred)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return
    idxs = idxs[:n]

    cols = min(3, len(idxs))
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_i, idx in enumerate(idxs):
        img = tf.keras.utils.load_img(image_paths[idx])
        axes[ax_i].imshow(img)
        true_lbl = class_names[y_true[idx]]
        pred_lbl = class_names[y_pred[idx]]
        conf = confidences[idx]
        axes[ax_i].set_title(f"true={true_lbl}\npred={pred_lbl} ({conf:.2f})", fontsize=10)
        axes[ax_i].axis("off")

    for ax_i in range(len(idxs), len(axes)):
        axes[ax_i].axis("off")

    plt.tight_layout()
    tag = "correct" if correct else "misclassified"
    path = os.path.join(out_dir, f"examples_{tag}.png")
    plt.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[INFO] Saved {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the photo severity classifier")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--model_path", type=str, default="severity_model.keras")
    p.add_argument("--class_names_file", type=str, default="class_names.json")
    p.add_argument("--split_manifest", type=str, default="test_split.json")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--output_dir", type=str, default="test_predictions")
    p.add_argument("--num_examples", type=int, default=6)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
