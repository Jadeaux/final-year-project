import os
import pandas as pd
import cv2
from sklearn.metrics import confusion_matrix

from pipeline.license_plate_cars import recognize_plate

GT_CSV = r"images/plates/plates_dataset/ground_truth.csv"
OUT_RESULTS = r"images/plates/plates_dataset/plates_results/results.csv"
OUT_CHAR_RESULTS = r"images/plates/plates_dataset/plates_results/char_results.csv"
OUT_FAILURES = r"images/plates/plates_dataset/plates_results/plates_failures"
OUT_DIR = r"images/plates/plates_dataset/plates_results"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_FAILURES, exist_ok=True)

df = pd.read_csv(GT_CSV)

plate_results = []
char_results = []

for _, row in df.iterrows():
    img_path = row["file_path"]
    true_label = str(row["true_label"]).strip()

    if not os.path.exists(img_path):
        plate_results.append({
            "file": img_path,
            "true": true_label,
            "pred": "MISSING",
            "correct": False,
            "error": "file_not_found"
        })
        continue

    img = cv2.imread(img_path)

    try:
        out = recognize_plate(img, debug=False)
        pred = str(out.get("plate_text", ""))
        error_msg = ""
    except Exception as e:
        out = None
        pred = "ERROR"
        error_msg = repr(e)

    correct = (pred == true_label)

    plate_results.append({
        "file": img_path,
        "true": true_label,
        "pred": pred,
        "correct": correct,
        "error": error_msg
    })

    if not correct and img is not None:
        cv2.imwrite(os.path.join(OUT_FAILURES, os.path.basename(img_path)), img)

    if out is not None:
        pred_chars = out.get("chars", [])

        max_len = max(len(true_label), len(pred_chars))
        for pos in range(max_len):
            true_char = true_label[pos] if pos < len(true_label) else "_"

            if pos < len(pred_chars):
                crow = pred_chars[pos]
                pred_char = str(crow.get("pred_char", "_"))
            else:
                crow = {}
                pred_char = "_"

            char_results.append({
                "file": img_path,
                "true_plate": true_label,
                "pred_plate": pred,
                "pos": pos,
                "char_type": "letter" if pos < 3 else "digit",
                "true_char": str(true_char),
                "pred_char": str(pred_char),
                "correct": str(true_char) == str(pred_char),
                "digit_group": crow.get("digit_group"),
                "n_blobs": crow.get("n_blobs"),
                "n_stems": crow.get("n_stems"),
            })

plate_df = pd.DataFrame(plate_results)
char_df = pd.DataFrame(char_results)

plate_df.to_csv(OUT_RESULTS, index=False)
char_df.to_csv(OUT_CHAR_RESULTS, index=False)

# -------------------------
# Accuracy metrics
# -------------------------
plate_acc = plate_df["correct"].mean() * 100 if len(plate_df) else 0
char_acc = char_df["correct"].mean() * 100 if len(char_df) else 0

letter_df = char_df[char_df["char_type"] == "letter"].copy()
digit_df = char_df[char_df["char_type"] == "digit"].copy()

letter_acc = letter_df["correct"].mean() * 100 if len(letter_df) else 0
digit_acc = digit_df["correct"].mean() * 100 if len(digit_df) else 0

print(f"Character Accuracy: {char_acc:.2f}%")
print(f"Letter Accuracy:    {letter_acc:.2f}%")
print(f"Digit Accuracy:     {digit_acc:.2f}%")


metrics_df = pd.DataFrame([{
    "character_accuracy": round(char_acc, 2),
    "letter_accuracy": round(letter_acc, 2),
    "digit_accuracy": round(digit_acc, 2)
}])

metrics_path = os.path.join(OUT_DIR, "summary_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print(f"\nSaved summary metrics to: {metrics_path}")


# -------------------------
# Per-position accuracy
# -------------------------
if len(char_df):
    pos_acc = (
        char_df.groupby("pos")["correct"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"correct": "accuracy_pct"})
    )
    pos_acc.to_csv(os.path.join(OUT_DIR, "per_position_accuracy.csv"), index=False)

# -------------------------
# Letter confusion matrix
# -------------------------
letter_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["?", "_", "UNK"]

if len(letter_df):
    letter_eval = letter_df.copy()
    letter_eval["true_char"] = letter_eval["true_char"].astype(str)
    letter_eval["pred_char"] = letter_eval["pred_char"].astype(str)

    cm_letters = confusion_matrix(
        letter_eval["true_char"],
        letter_eval["pred_char"],
        labels=letter_labels
    )

    cm_letters_df = pd.DataFrame(cm_letters, index=letter_labels, columns=letter_labels)
    cm_letters_df.to_csv(os.path.join(OUT_DIR, "letter_confusion_matrix.csv"))

# -------------------------
# Digit confusion matrix
# -------------------------
digit_labels = [str(i) for i in range(10)] + ["?", "_", "UNK"]

if len(digit_df):
    digit_eval = digit_df.copy()
    digit_eval["true_char"] = digit_eval["true_char"].astype(str)
    digit_eval["pred_char"] = digit_eval["pred_char"].astype(str)

    cm_digits = confusion_matrix(
        digit_eval["true_char"],
        digit_eval["pred_char"],
        labels=digit_labels
    )

    cm_digits_df = pd.DataFrame(cm_digits, index=digit_labels, columns=digit_labels)
    cm_digits_df.to_csv(os.path.join(OUT_DIR, "digit_confusion_matrix.csv"))

# -------------------------
# Digit blob-count distribution
# -------------------------
def blob_bucket(x):
    if pd.isna(x):
        return "UNK"
    x = int(x)
    if x >= 3:
        return "3+"
    return str(x)

if len(digit_df) and "n_blobs" in digit_df.columns:
    digit_feat = digit_df.copy()
    digit_feat["blob_bucket"] = digit_feat["n_blobs"].apply(blob_bucket)

    blob_table = pd.crosstab(digit_feat["true_char"], digit_feat["blob_bucket"])

    for col in ["0", "1", "2", "3+", "UNK"]:
        if col not in blob_table.columns:
            blob_table[col] = 0

    blob_table = blob_table[["0", "1", "2", "3+", "UNK"]]
    blob_table.to_csv(os.path.join(OUT_DIR, "digit_blobs_distribution.csv"))