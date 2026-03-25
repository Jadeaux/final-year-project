import cv2
import os
import glob
import pandas as pd
from pipeline.license_plate import localise_plate, segment_plate, recognize_plate, char_type_from_index

PLATES_DIR = "images/plates"
LABELS_XLSX = "images/license_plate_labels.xlsx"
OUT_CSV = "lpr_outputs/plate_predictions_datasetSurvey.csv"


def run_pipeline(img_path, debug=False):
    img = cv2.imread(img_path)
    if img is None:
        return None
    plate_roi = localise_plate(img, debug=debug)
    if plate_roi is None:
        return None
    normalized_chars = segment_plate(plate_roi, debug=debug)
    plate_text = recognize_plate(normalized_chars, debug=debug)
    return plate_text


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def compare_plate_strings(true_plate: str, pred_plate: str):
    """
    Returns detailed comparison stats:
    - total correct chars
    - total chars compared
    - correct digit chars
    - total digit chars
    - correct letter chars
    - total letter chars
    - exact match
    """

    true_plate = safe_str(true_plate).upper()
    pred_plate = safe_str(pred_plate).upper()

    max_len = max(len(true_plate), len(pred_plate))

    total_correct = 0
    total_count = 0

    digit_correct = 0
    digit_total = 0

    letter_correct = 0
    letter_total = 0

    for i in range(max_len):
        t = true_plate[i] if i < len(true_plate) else None
        p = pred_plate[i] if i < len(pred_plate) else None

        # if no true char at this position, skip from accuracy denominator
        if t is None:
            continue

        total_count += 1
        if p == t:
            total_correct += 1

        ctype = char_type_from_index(i)

        if ctype == "digit":
            digit_total += 1
            if p == t:
                digit_correct += 1
        else:
            letter_total += 1
            if p == t:
                letter_correct += 1

    exact_match = (true_plate == pred_plate)

    return {
        "total_correct_chars": total_correct,
        "total_true_chars": total_count,
        "digit_correct": digit_correct,
        "digit_total": digit_total,
        "letter_correct": letter_correct,
        "letter_total": letter_total,
        "exact_match": exact_match
    }


def run_batch_evaluation():
    os.makedirs("lpr_outputs", exist_ok=True)

    # Read labels
    df_labels = pd.read_excel(LABELS_XLSX)

    # normalize column names just in case
    df_labels.columns = [str(c).strip() for c in df_labels.columns]

    required_cols = {"image_name", "true_plate"}
    if not required_cols.issubset(df_labels.columns):
        raise ValueError(f"Excel must contain columns: {required_cols}")

    # Build lookup dict
    label_map = {}
    for _, row in df_labels.iterrows():
        image_name = safe_str(row["image_name"])
        true_plate = safe_str(row["true_plate"]).upper()
        if image_name:
            label_map[image_name] = true_plate

    image_paths = sorted(glob.glob(os.path.join(PLATES_DIR, "*.*")))

    results = []

    grand_total_correct = 0
    grand_total_chars = 0

    grand_digit_correct = 0
    grand_digit_total = 0

    grand_letter_correct = 0
    grand_letter_total = 0

    exact_match_count = 0
    evaluated_count = 0

    for img_path in image_paths:
        image_name = os.path.basename(img_path)

        if image_name not in label_map:
            print(f"[SKIP] No label found for {image_name}")
            continue

        true_plate = label_map[image_name]

        pred_plate = run_pipeline(img_path, debug=False)
        pred_plate = "" if pred_plate is None else str(pred_plate).strip().upper()

        stats = compare_plate_strings(true_plate, pred_plate)

        grand_total_correct += stats["total_correct_chars"]
        grand_total_chars += stats["total_true_chars"]

        grand_digit_correct += stats["digit_correct"]
        grand_digit_total += stats["digit_total"]

        grand_letter_correct += stats["letter_correct"]
        grand_letter_total += stats["letter_total"]

        exact_match_count += int(stats["exact_match"])
        evaluated_count += 1

        results.append({
            "image_name": image_name,
            "true_plate": true_plate,
            "pred_plate": pred_plate,
            "exact_match": stats["exact_match"],
            "correct_chars": stats["total_correct_chars"],
            "total_chars": stats["total_true_chars"],
            "digit_correct": stats["digit_correct"],
            "digit_total": stats["digit_total"],
            "letter_correct": stats["letter_correct"],
            "letter_total": stats["letter_total"]
        })

        print(
            f"{image_name} | true={true_plate} | pred={pred_plate} | "
            f"chars={stats['total_correct_chars']}/{stats['total_true_chars']} | "
            f"digits={stats['digit_correct']}/{stats['digit_total']} | "
            f"letters={stats['letter_correct']}/{stats['letter_total']} | "
            f"exact={stats['exact_match']}"
        )

    # Final metrics
    overall_char_acc = grand_total_correct / grand_total_chars if grand_total_chars > 0 else 0.0
    digit_acc = grand_digit_correct / grand_digit_total if grand_digit_total > 0 else 0.0
    letter_acc = grand_letter_correct / grand_letter_total if grand_letter_total > 0 else 0.0
    exact_plate_acc = exact_match_count / evaluated_count if evaluated_count > 0 else 0.0

    print("\n==============================")
    print(f"Images evaluated:         {evaluated_count}")
    print(f"Exact plate accuracy:     {exact_plate_acc * 100:.2f}%")
    print(f"Overall char accuracy:    {overall_char_acc * 100:.2f}%")
    print(f"Digit char accuracy:      {digit_acc * 100:.2f}%")
    print(f"Letter char accuracy:     {letter_acc * 100:.2f}%")
    print("==============================")

    # Add summary row to results
    results.append({
        "image_name": "SUMMARY",
        "true_plate": f"{evaluated_count} images",
        "pred_plate": "",
        "exact_match": f"{exact_plate_acc * 100:.2f}%",
        "correct_chars": f"{grand_total_correct}/{grand_total_chars} ({overall_char_acc*100:.2f}%)",
        "total_chars": grand_total_chars,
        "digit_correct": f"{grand_digit_correct}/{grand_digit_total} ({digit_acc*100:.2f}%)",
        "digit_total": grand_digit_total,
        "letter_correct": f"{grand_letter_correct}/{grand_letter_total} ({letter_acc*100:.2f}%)",
        "letter_total": grand_letter_total,
    })

    # Save CSV with summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"Saved CSV to: {OUT_CSV}")


if __name__ == "__main__":
    run_batch_evaluation()
