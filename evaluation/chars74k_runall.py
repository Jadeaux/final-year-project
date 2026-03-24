import os
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from pipeline.preprocessing_chars import preprocess_step1, preprocess_letters, thin
from pipeline.classification import classify_with_blobs_from_A
from pipeline.classification_letters_california import classify_letter

ROOT = r"E:\EnglishImg (1)\English\Img\GoodImg\Bmp"
OUT_DIR = r"E:\chars74k_eval_after_fix"
os.makedirs(OUT_DIR, exist_ok=True)

ALL_RESULTS_CSV = os.path.join(OUT_DIR, "all_predictions_new.csv")
DIGIT_CM_CSV = os.path.join(OUT_DIR, "digit_confusion_matrix_new.csv")
LETTER_CM_CSV = os.path.join(OUT_DIR, "letter_confusion_matrix_new.csv")
DIGIT_PER_CLASS_CSV = os.path.join(OUT_DIR, "digit_per_class_accuracy_new.csv")
LETTER_PER_CLASS_CSV = os.path.join(OUT_DIR, "letter_per_class_accuracy_new.csv")

DIGIT_LABELS = [str(i) for i in range(10)]
LETTER_LABELS = [chr(ord("A") + i) for i in range(26)]

def sample_folder_to_label(folder_name):
    if not folder_name.startswith("Sample"):
        return None

    idx = int(folder_name.replace("Sample", ""))

    if 1 <= idx <= 10:
        return str(idx - 1)

    if 11 <= idx <= 36:
        return chr(ord("A") + (idx - 11))

    return None

def is_digit_folder(folder_name):
    idx = int(folder_name.replace("Sample", ""))
    return 1 <= idx <= 10

def predict_chars74k_image(img_path, sample_folder, visualize=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "ERROR", None

    try:
        if is_digit_folder(sample_folder):
            A, cleaned, binary_norm = preprocess_step1(img, visualize=visualize)
            A = (A > 0).astype(np.uint8)

            num, labels, stats, _ = cv2.connectedComponentsWithStats(A, connectivity=8)
            if num > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest = 1 + np.argmax(areas)
                A = (labels == largest).astype(np.uint8)

            pred = classify_with_blobs_from_A(A, visualize=False, debug=False)
            if pred is None:
                return "ERROR", None

            if isinstance(pred, tuple):
                pred_digit = pred[0]
                pred_group = pred[1] if len(pred) > 1 else None
            else:
                pred_digit = pred
                pred_group = None

            return str(pred_digit).strip(), pred_group

        else:
            A, cleaned, binary_norm = preprocess_letters(img, visualize=visualize)
            A_thin = thin(A)

            pred = classify_letter(A, A_thin)

            if pred is None:
                return "ERROR", None

            return str(pred).strip(), None

    except Exception as e:
        print(f"Error on {img_path}: {e}")
        return "ERROR", None

# build dataset
folders = sorted([
    f for f in os.listdir(ROOT)
    if os.path.isdir(os.path.join(ROOT, f))
])

records = []

for folder in folders:
    true_label = sample_folder_to_label(folder)
    if true_label is None:
        continue

    folder_path = os.path.join(ROOT, folder)
    image_paths = (
        glob.glob(os.path.join(folder_path, "*.png")) +
        glob.glob(os.path.join(folder_path, "*.jpg")) +
        glob.glob(os.path.join(folder_path, "*.jpeg")) +
        glob.glob(os.path.join(folder_path, "*.bmp"))
    )

    for img_path in image_paths:
        records.append({
            "sample_folder": folder,
            "true_label": true_label,
            "image_path": img_path
        })

df = pd.DataFrame(records)
print(f"Total images found: {len(df)}")

# run predictions
pred_labels = []
pred_groups = []

for i, row in df.iterrows():
    pred_label, pred_group = predict_chars74k_image(
        row["image_path"],
        row["sample_folder"],
        visualize=False
    )
    pred_labels.append(pred_label)
    pred_groups.append(pred_group)

    if (i + 1) % 200 == 0 or (i + 1) == len(df):
        print(f"Processed {i + 1}/{len(df)}")

df["pred_label"] = pred_labels
df["pred_group"] = pred_groups
df["correct"] = df["true_label"] == df["pred_label"]

df.to_csv(ALL_RESULTS_CSV, index=False)

digit_df = df[df["true_label"].isin(DIGIT_LABELS)].copy()
letter_df = df[df["true_label"].isin(LETTER_LABELS)].copy()

digit_df["true_label"] = digit_df["true_label"].astype(str).str.strip()
digit_df["pred_label"] = digit_df["pred_label"].astype(str).str.strip()

letter_df["true_label"] = letter_df["true_label"].astype(str).str.strip()
letter_df["pred_label"] = letter_df["pred_label"].astype(str).str.strip()

digit_valid = digit_df[digit_df["pred_label"].isin(DIGIT_LABELS)].copy()
letter_valid = letter_df[letter_df["pred_label"].isin(LETTER_LABELS)].copy()

digit_cm = confusion_matrix(
    digit_valid["true_label"],
    digit_valid["pred_label"],
    labels=DIGIT_LABELS
)
digit_cm_df = pd.DataFrame(digit_cm, index=DIGIT_LABELS, columns=DIGIT_LABELS)
digit_cm_df.to_csv(DIGIT_CM_CSV)

letter_cm = confusion_matrix(
    letter_valid["true_label"],
    letter_valid["pred_label"],
    labels=LETTER_LABELS
)
letter_cm_df = pd.DataFrame(letter_cm, index=LETTER_LABELS, columns=LETTER_LABELS)
letter_cm_df.to_csv(LETTER_CM_CSV)

def build_per_class_accuracy_table(sub_df, labels):
    rows = []

    for lbl in labels:
        class_df = sub_df[sub_df["true_label"] == lbl]
        total = len(class_df)
        correct = (class_df["pred_label"] == lbl).sum()
        errors = (class_df["pred_label"] == "ERROR").sum()
        acc = (correct / total * 100) if total > 0 else 0.0

        rows.append({
            "class": lbl,
            "total_samples": total,
            "correct_predictions": correct,
            "errors": errors,
            "accuracy_percent": round(acc, 2)
        })

    return pd.DataFrame(rows)

digit_per_class_df = build_per_class_accuracy_table(digit_df, DIGIT_LABELS)
letter_per_class_df = build_per_class_accuracy_table(letter_df, LETTER_LABELS)

digit_per_class_df.to_csv(DIGIT_PER_CLASS_CSV, index=False)
letter_per_class_df.to_csv(LETTER_PER_CLASS_CSV, index=False)

overall_acc = df["correct"].mean() * 100 if len(df) else 0
digit_acc = digit_df["correct"].mean() * 100 if len(digit_df) else 0
letter_acc = letter_df["correct"].mean() * 100 if len(letter_df) else 0

print("\n==============================")
print(f"Overall accuracy: {overall_acc:.2f}%")
print(f"Digit accuracy:   {digit_acc:.2f}%")
print(f"Letter accuracy:  {letter_acc:.2f}%")
print("==============================")