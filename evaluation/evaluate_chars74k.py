"""
Chars74K digit trace harness
Uses the SAME preprocessing/classification pipeline as the actual run.

Outputs:
- chars74k_digit_results.csv
- chars74k_digit_confusion.csv
- chars74k_digit_endpoints.csv
- chars74k_digit_blobs_thick.csv
- chars74k_digit_blobs_thin.csv
- chars74k_digit_feature_averages.csv
- chars74k_digit_rulepath_counts.csv
- chars74k_digit_traces_per_sample.csv
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd

from skimage.measure import label, regionprops

from pipeline.preprocessing_chars import preprocess_step1, thin
from pipeline.morphology_chars import find_blobs
from pipeline.features import get_stems, get_banded_points, draw_line
from pipeline.classification_kumar import classify_with_blobs_from_A


# =========================================================
# PATHS
# =========================================================
ROOT = r"E:\EnglishImg (1)\English\Img\GoodImg\Bmp"
OUT_DIR = r"E:\chars74k_digit_trace"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def prepare_digit_for_classification(img, visualize=False):
    """
    Match the current working Chars74K digit pipeline:
    - preprocess_step1 from preprocessing_chars
    - convert to 0/1
    - keep only largest connected component
    """
    A, cleaned, binary_norm = preprocess_step1(img, visualize=visualize)
    A = (A > 0).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(A, connectivity=8)

    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        A = (labels == largest).astype(np.uint8)

    return A, cleaned, binary_norm

def list_images(folder, exts=("png", "jpg", "jpeg", "bmp", "tif", "tiff")):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(folder, f"*.{e}"))
    return sorted(paths)

def sample_folder_to_digit(folder_name):
    idx = int(folder_name.replace("Sample", ""))
    if 1 <= idx <= 10:
        return idx - 1
    return None

def digit_sample_folders(root):
    folders = []
    for f in sorted(os.listdir(root)):
        full = os.path.join(root, f)
        if not os.path.isdir(full):
            continue
        d = sample_folder_to_digit(f)
        if d is not None:
            folders.append(f)
    return folders

def endpoint_count_8conn(skel01: np.ndarray) -> int:
    sk = (skel01 > 0).astype(np.uint8)
    if sk.sum() == 0:
        return 0
    K = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(sk, -1, K, borderType=cv2.BORDER_CONSTANT)
    endpoints = ((sk == 1) & (nb == 1)).sum()
    return int(endpoints)

def bin_endpoints(e: int) -> str:
    if e is None:
        return "UNK"
    if e >= 6:
        return "6+"
    return str(int(e))

def bin_blobs(n: int) -> str:
    if n is None:
        return "UNK"
    if n >= 3:
        return "3+"
    return str(int(n))

def blob_areas_from_mask(blobs01: np.ndarray):
    lab = label(blobs01, connectivity=2)
    props = regionprops(lab)
    return sorted([p.area for p in props], reverse=True)


# =========================================================
# TRACE CLASSIFIER PATH
# =========================================================
def trace_classifier_path(A: np.ndarray):
    """
    Mirrors the digit classifier logic for logging only.
    Uses the SAME preprocessed A that your actual classifier sees.
    """
    out = {}

    A01 = (A > 0).astype(np.uint8)
    A_thin = thin(A01)

    # thick blobs = actual group selector
    blobs_thick, n_blobs_thick = find_blobs(A01)
    out["n_blobs_thick"] = int(n_blobs_thick)

    # thin blobs = useful extra diagnostic
    blobs_thin, n_blobs_thin = find_blobs(A_thin)
    out["n_blobs_thin"] = int(n_blobs_thin)

    # endpoints
    out["endpoints"] = endpoint_count_8conn(A_thin)

    # decide path
    if n_blobs_thick > 0:
        out["group"] = "Group 1"

        # stems are computed from thin skeleton + thin blobs
        blobs_for_stems, nb_for_stems = find_blobs(A_thin)
        stems_img, n_stems, stem_cents = get_stems(A_thin, blobs_for_stems)

        out["g1_n_stems"] = int(n_stems)
        out["g1_nb_thin_for_stems"] = int(nb_for_stems)

        # group 2 fields left blank
        out["g2_pts_ok"] = None
        out["g2_nb1_tlbl"] = None
        out["g2_nb2_tlbr"] = None
        out["g2_nb3_trbr"] = None
        out["g2_n_stems_after_tlbl"] = None
        out["g2_case"] = None

    else:
        out["group"] = "Group 2"

        pts = get_banded_points(A_thin, split=0.5)
        if pts is None:
            out["g2_pts_ok"] = 0
            out["g2_nb1_tlbl"] = None
            out["g2_nb2_tlbr"] = None
            out["g2_nb3_trbr"] = None
            out["g2_n_stems_after_tlbl"] = None
            out["g2_case"] = "no_pts"
            out["g1_n_stems"] = None
            out["g1_nb_thin_for_stems"] = None
            return out

        out["g2_pts_ok"] = 1
        TL, BL, TR, BR = pts

        # TL -> BL
        A1 = draw_line(A_thin, TL, BL)
        _, nb1 = find_blobs(A1)
        out["g2_nb1_tlbl"] = int(nb1)

        if nb1 == 0:
            # TL -> BR
            A2 = draw_line(A_thin, TL, BR)
            _, nb2 = find_blobs(A2)
            out["g2_nb2_tlbr"] = int(nb2)

            out["g2_nb3_trbr"] = None
            out["g2_n_stems_after_tlbl"] = None
            out["g2_case"] = "nb1=0_then_tlbr"

        else:
            blobs1, _ = find_blobs(A1)
            _, n_stems, _ = get_stems(A1, blobs1)
            out["g2_n_stems_after_tlbl"] = int(n_stems)

            if n_stems == 2:
                out["g2_nb2_tlbr"] = None
                out["g2_nb3_trbr"] = None
                out["g2_case"] = "nb1>0_stems=2"

            elif n_stems > 0:
                out["g2_nb2_tlbr"] = None
                out["g2_nb3_trbr"] = None
                out["g2_case"] = "nb1>0_stems>0"

            else:
                # TR -> BR
                A3 = draw_line(A_thin, TR, BR)
                _, nb3 = find_blobs(A3, min_blob_area=30)
                out["g2_nb3_trbr"] = int(nb3)
                out["g2_nb2_tlbr"] = None
                out["g2_case"] = "nb1>0_stems=0_then_trbr"

        out["g1_n_stems"] = None
        out["g1_nb_thin_for_stems"] = None

    return out


# =========================================================
# EVALUATE ONE DIGIT FOLDER
# =========================================================
def eval_one_digit_folder(folder_path: str, true_digit: int):
    paths = list_images(folder_path)
    if not paths:
        return 0, 0, 0.0, {}, None, {}, {}, {}, []

    correct = 0
    confusion = {}

    ep_hist = {}
    thick_blob_hist = {}
    thin_blob_hist = {}

    feats = {
        "endpoints": [],
        "n_blobs_thick": [],
        "n_blobs_thin": [],
        "group1_n_stems": [],
        "g2_nb1_tlbl": [],
        "g2_nb2_tlbr": [],
        "g2_nb3_trbr": [],
        "g2_n_stems_after_tlbl": [],
    }

    traces = []

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # SAME preprocessing as your actual run
        # SAME preprocessing as your actual working run
        A, _, _ = prepare_digit_for_classification(img, visualize=False)
        pred = classify_with_blobs_from_A(A, visualize=False, debug=False)

        pred_digit = "UNK"
        pred_group = None

        if pred is not None:
            if isinstance(pred, tuple):
                pred_digit = str(int(pred[0])) if pred[0] is not None else "UNK"
                pred_group = pred[1] if len(pred) > 1 else None
            else:
                pred_digit = str(int(pred))

        if pred_digit == str(true_digit):
            correct += 1
        confusion[pred_digit] = confusion.get(pred_digit, 0) + 1

        tr = trace_classifier_path(A)
        tr["file"] = os.path.basename(p)
        tr["true"] = int(true_digit)
        tr["pred"] = pred_digit
        tr["pred_group_from_classifier"] = pred_group
        traces.append(tr)

        e = tr.get("endpoints", None)
        nb_thick = tr.get("n_blobs_thick", None)
        nb_thin = tr.get("n_blobs_thin", None)

        ep_hist[bin_endpoints(e)] = ep_hist.get(bin_endpoints(e), 0) + 1
        thick_blob_hist[bin_blobs(nb_thick)] = thick_blob_hist.get(bin_blobs(nb_thick), 0) + 1
        thin_blob_hist[bin_blobs(nb_thin)] = thin_blob_hist.get(bin_blobs(nb_thin), 0) + 1

        feats["endpoints"].append(e)
        feats["n_blobs_thick"].append(nb_thick)
        feats["n_blobs_thin"].append(nb_thin)

        feats["group1_n_stems"].append(tr.get("g1_n_stems", 0) or 0)
        feats["g2_nb1_tlbl"].append(tr.get("g2_nb1_tlbl", 0) or 0)
        feats["g2_nb2_tlbr"].append(tr.get("g2_nb2_tlbr", 0) or 0)
        feats["g2_nb3_trbr"].append(tr.get("g2_nb3_trbr", 0) or 0)
        feats["g2_n_stems_after_tlbl"].append(tr.get("g2_n_stems_after_tlbl", 0) or 0)

    n = len(paths)
    acc = (correct / n) if n > 0 else 0.0

    feat_means = {
        "Digit": int(true_digit),
        "Samples": int(n),
        "Accuracy_%": round(acc * 100, 2),
        "endpoints_mean": float(np.mean(feats["endpoints"])) if n else 0.0,
        "blobs_thick_mean": float(np.mean(feats["n_blobs_thick"])) if n else 0.0,
        "blobs_thin_mean": float(np.mean(feats["n_blobs_thin"])) if n else 0.0,
        "group1_stems_mean": float(np.mean(feats["group1_n_stems"])) if n else 0.0,
        "g2_nb1_tlbl_mean": float(np.mean(feats["g2_nb1_tlbl"])) if n else 0.0,
        "g2_nb2_tlbr_mean": float(np.mean(feats["g2_nb2_tlbr"])) if n else 0.0,
        "g2_nb3_trbr_mean": float(np.mean(feats["g2_nb3_trbr"])) if n else 0.0,
        "g2_stems_after_tlbl_mean": float(np.mean(feats["g2_n_stems_after_tlbl"])) if n else 0.0,
    }

    return n, correct, acc, confusion, feat_means, ep_hist, thick_blob_hist, thin_blob_hist, traces


# =========================================================
# AGGREGATE RULE PATHS
# =========================================================
def aggregate_rulepaths(traces_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for d in range(10):
        sub = traces_df[traces_df["true"] == d].copy()
        if len(sub) == 0:
            continue

        g1 = (sub["group"] == "Group 1").sum()
        g2 = (sub["group"] == "Group 2").sum()

        g2sub = sub[sub["group"] == "Group 2"]

        tlbl_nb1_pos = (g2sub["g2_nb1_tlbl"].fillna(0) > 0).sum()
        tlbl_nb1_zero = (g2sub["g2_nb1_tlbl"].fillna(0) == 0).sum()

        tlbr_pos = (g2sub["g2_nb2_tlbr"].fillna(0) > 0).sum()

        trbr_pos = (g2sub["g2_nb3_trbr"].fillna(0) > 0).sum()
        trbr_zero = (g2sub["g2_nb3_trbr"].fillna(0) == 0).sum()

        stems2 = (g2sub["g2_n_stems_after_tlbl"].fillna(0) == 2).sum()
        stems_pos = (g2sub["g2_n_stems_after_tlbl"].fillna(0) > 0).sum()
        stems_zero = (g2sub["g2_n_stems_after_tlbl"].fillna(0) == 0).sum()

        case_counts = g2sub["g2_case"].fillna("NA").value_counts().to_dict()

        row = {
            "Digit": d,
            "Samples": int(len(sub)),
            "Group1_count": int(g1),
            "Group2_count": int(g2),
            "G2_TLBL_nb1>0_count": int(tlbl_nb1_pos),
            "G2_TLBL_nb1==0_count": int(tlbl_nb1_zero),
            "G2_TLBR_nb2>0_count": int(tlbr_pos),
            "G2_TRBR_nb3>0_count": int(trbr_pos),
            "G2_TRBR_nb3==0_count": int(trbr_zero),
            "G2_stems==2_count": int(stems2),
            "G2_stems>0_count": int(stems_pos),
            "G2_stems==0_count": int(stems_zero),
        }

        for k, v in case_counts.items():
            row[f"case:{k}"] = int(v)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("Digit")


# =========================================================
# MAIN EVALUATION
# =========================================================
def eval_all_digits(root_dir: str):
    folders = digit_sample_folders(root_dir)

    rows = []
    feature_rows = []

    all_total = 0
    all_correct = 0

    confusion_matrix = {}
    endpoint_hists = {}
    thick_blob_hists = {}
    thin_blob_hists = {}
    all_traces = []

    for folder_name in folders:
        true_digit = sample_folder_to_digit(folder_name)
        folder_path = os.path.join(root_dir, folder_name)

        n, correct, acc, confusion, feat_means, ep_hist, thick_hist, thin_hist, traces = \
            eval_one_digit_folder(folder_path, true_digit)

        rows.append({
            "Digit": true_digit,
            "Samples": n,
            "Correct": correct,
            "Accuracy_%": round(acc * 100, 2),
        })

        feature_rows.append(feat_means)

        confusion_matrix[str(true_digit)] = confusion
        endpoint_hists[str(true_digit)] = ep_hist
        thick_blob_hists[str(true_digit)] = thick_hist
        thin_blob_hists[str(true_digit)] = thin_hist
        all_traces += traces

        all_total += n
        all_correct += correct

        print(f"{true_digit}: {correct}/{n} = {acc*100:.2f}%")

    overall_acc = (all_correct / all_total) if all_total > 0 else 0.0
    print("\n====================")
    print(f"OVERALL: {all_correct}/{all_total} = {overall_acc*100:.2f}%")
    print("====================\n")

    results_df = pd.DataFrame(rows).sort_values("Digit")
    features_df = pd.DataFrame(feature_rows).sort_values("Digit")

    all_preds = [str(i) for i in range(10)] + ["UNK"]
    cm_df = pd.DataFrame(0, index=[str(i) for i in range(10)], columns=all_preds)
    for t in [str(i) for i in range(10)]:
        for p, cnt in confusion_matrix.get(t, {}).items():
            if p not in cm_df.columns:
                cm_df[p] = 0
            cm_df.loc[t, p] = cnt

    ep_bins = ["0", "1", "2", "3", "4", "5", "6+", "UNK"]
    endpoint_df = pd.DataFrame(0, index=[str(i) for i in range(10)], columns=ep_bins)
    for t in [str(i) for i in range(10)]:
        for b, cnt in endpoint_hists.get(t, {}).items():
            endpoint_df.loc[t, b] = cnt

    blob_bins = ["0", "1", "2", "3+", "UNK"]

    blob_thick_df = pd.DataFrame(0, index=[str(i) for i in range(10)], columns=blob_bins)
    for t in [str(i) for i in range(10)]:
        for b, cnt in thick_blob_hists.get(t, {}).items():
            blob_thick_df.loc[t, b] = cnt

    blob_thin_df = pd.DataFrame(0, index=[str(i) for i in range(10)], columns=blob_bins)
    for t in [str(i) for i in range(10)]:
        for b, cnt in thin_blob_hists.get(t, {}).items():
            blob_thin_df.loc[t, b] = cnt

    traces_df = pd.DataFrame(all_traces)
    rulepath_df = aggregate_rulepaths(traces_df)

    return results_df, cm_df, features_df, endpoint_df, blob_thick_df, blob_thin_df, rulepath_df, traces_df


if __name__ == "__main__":
    (results_df, confusion_df, features_df,
     endpoint_df, blob_thick_df, blob_thin_df,
     rulepath_df, traces_df) = eval_all_digits(ROOT)

    results_df.to_csv(os.path.join(OUT_DIR, "chars74k_digit_results_kumar.csv"), index=False)
    confusion_df.to_csv(os.path.join(OUT_DIR, "chars74k_digit_confusion_kumar.csv"))
    endpoint_df.to_csv(os.path.join(OUT_DIR, "chars74k_digit_endpoints_kumar.csv"))
    blob_thick_df.to_csv(os.path.join(OUT_DIR, "chars74k_digit_blobs_thick_kumar.csv"))
    blob_thin_df.to_csv(os.path.join(OUT_DIR, "chars74k_digit_blobs_thin_kumar.csv"))
    features_df.to_csv(os.path.join(OUT_DIR, "chars74k_digit_feature_averages_kumar.csv"), index=False)
    rulepath_df.to_csv(os.path.join(OUT_DIR, "chars74k_digit_rulepath_counts_kumar.csv"), index=False)
    traces_df.to_csv(os.path.join(OUT_DIR, "chars74k_digit_traces_per_sample_kumar.csv"), index=False)

    print("Saved chars74k_digit_results_kumar.csv")
    print("Saved chars74k_digit_confusion_kumar.csv")
    print("Saved chars74k_digit_endpoints_kumar.csv")
    print("Saved chars74k_digit_blobs_thick_kumar.csv")
    print("Saved chars74k_digit_blobs_thin_kumar.csv")
    print("Saved chars74k_digit_feature_averages_kumar.csv")
    print("Saved chars74k_digit_rulepath_counts_kumar.csv")
    print("Saved chars74k_digit_traces_per_sample_kumar.csv")