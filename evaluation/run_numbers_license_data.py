"""
Digit evaluation harness (numbers 0–9) — same idea as your letters evaluator, plus
**rule-path instrumentation** for your blob + helper-line classifier.

Outputs:
- digit_eval_results.csv            (per-digit accuracy)
- digit_eval_confusion.csv          (confusion matrix: true x predicted)
- digit_eval_endpoints.csv          (endpoint-bin counts per true digit)
- digit_eval_blobs_thick.csv        (THICK blob-bin counts per true digit)
- digit_eval_blobs_thin.csv         (THIN blob-bin counts per true digit)
- digit_feature_averages.csv        (means of key features per digit)
- digit_rulepath_counts.csv         (counts of classifier branches / helper-line outcomes per digit)
"""

import os, glob
import cv2
import numpy as np
import pandas as pd

# ---- your pipeline imports (adjust if needed) ----
from pipeline.preprocessing import preprocess_step1, thin
from pipeline.morphology import find_blobs
from pipeline.features import get_stems, get_banded_points, draw_line
from pipeline.classification_kumar import classify_with_blobs_from_A  # (A -> digit, group)
from skimage.measure import label, regionprops


# =========================
# Helpers

def blob_areas_from_mask(blobs01: np.ndarray):
    """
    Return sorted blob areas from a 0/1 blob mask.
    """
    lab = label(blobs01, connectivity=2)
    props = regionprops(lab)
    return sorted([p.area for p in props], reverse=True)


def analyze_line_blob_sizes_for_digit_folder(folder_path: str, true_digit: int):
    """
    For a folder of one true digit, measure blob sizes after:
      A2 = TL->BR   (your 'second line')
      A3 = TR->BR   (your 'third line')

    Only meaningful for Group 2 style digits, but we compute whenever points exist.
    """
    paths = list_images(folder_path)

    rows = []

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # use the SAME preprocessing as your run file
        img = resize_to_height(img, target_h=240)
        A, _, _ = preprocess_step1(img, visualize=False)
        A = (A > 0).astype(np.uint8)

        A_thin = thin(A)

        pts = get_banded_points(A_thin, split=0.5)
        if pts is None:
            continue

        TL, BL, TR, BR = pts

        # second line in your wording: TL -> BR
        A2 = draw_line(A_thin, TL, BR)
        blobs2, nb2 = find_blobs(A2, min_blob_area=1)
        areas2 = blob_areas_from_mask(blobs2)

        # third line: TR -> BR
        A3 = draw_line(A_thin, TR, BR)
        blobs3, nb3 = find_blobs(A3, min_blob_area=1)
        areas3 = blob_areas_from_mask(blobs3)

        rows.append({
            "file": os.path.basename(p),
            "true_digit": true_digit,

            "A2_nb": nb2,
            "A2_largest_blob": float(areas2[0]) if len(areas2) > 0 else 0.0,
            "A2_total_blob_area": float(sum(areas2)) if len(areas2) > 0 else 0.0,

            "A3_nb": nb3,
            "A3_largest_blob": float(areas3[0]) if len(areas3) > 0 else 0.0,
            "A3_total_blob_area": float(sum(areas3)) if len(areas3) > 0 else 0.0,
        })

    return pd.DataFrame(rows)

def resize_to_height(gray, target_h=240):
    h, w = gray.shape[:2]
    if h == target_h:
        return gray
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_AREA)

def preprocess_plate_style_digit(img, visualize=False,
                                 blur_ksize=(3, 3),
                                 adapt_block=31,
                                 adapt_C=10,
                                 close_ksize=(2, 2),
                                 close_iters=1,
                                 min_area=40,
                                 fill_holes=True):
    """
    Plate-style preprocessing adapted for ONE isolated synthetic digit.

    Returns:
        A          : 0/1 foreground mask
        cleaned255 : 0/255 cleaned binary
        binary255  : initial adaptive-threshold binary
    """

    # 1) grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.astype(np.uint8)

    # 2) blur
    gray = cv2.GaussianBlur(gray, blur_ksize, 0)

    # 3) adaptive threshold like license plate chars
    binary255 = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adapt_block,
        adapt_C
    )

    # 4) small close to bridge cracks in strokes
    if close_ksize is not None:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, close_ksize)
        cleaned255 = cv2.morphologyEx(binary255, cv2.MORPH_CLOSE, k, iterations=close_iters)
    else:
        cleaned255 = binary255.copy()

    # 5) connected-component cleanup
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned255, connectivity=8)
    den = np.zeros_like(cleaned255)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            den[labels == i] = 255

    cleaned255 = den

    # 6) keep largest CC only (very important for isolated digits)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned255, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        keep = np.zeros_like(cleaned255)
        keep[labels == largest_idx] = 255
        cleaned255 = keep

    # 7) tight crop to ink
    ys, xs = np.where(cleaned255 > 0)
    if len(xs) > 0:
        pad = 2
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        x0 = max(0, x0 - pad)
        x1 = min(cleaned255.shape[1] - 1, x1 + pad)
        y0 = max(0, y0 - pad)
        y1 = min(cleaned255.shape[0] - 1, y1 + pad)

        cleaned255 = cleaned255[y0:y1+1, x0:x1+1]

    # 8) optional fill of tiny accidental holes
    if fill_holes:
        cleaned255 = fill_small_holes(cleaned255, max_hole_frac=0.02)

    # 9) convert to 0/1
    A = (cleaned255 > 0).astype(np.uint8)

    if visualize:
        plt.figure(figsize=(10, 2.5))
        plt.subplot(1, 4, 1)
        plt.imshow(gray, cmap="gray")
        plt.title("gray+blur")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(binary255, cmap="gray")
        plt.title("adaptive binary")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(cleaned255, cmap="gray")
        plt.title("cleaned")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(A, cmap="gray")
        plt.title("A (0/1)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return A, cleaned255, binary255

def fill_small_holes(bin255, max_hole_frac=0.05, max_hole_pixels=None):
    A = (bin255 > 0).astype(np.uint8)
    ys, xs = np.where(A == 1)
    if xs.size == 0:
        return bin255

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    roi = A[y0:y1+1, x0:x1+1]
    H, W = roi.shape
    box_area = H * W

    bg = (roi == 0).astype(np.uint8) * 255
    num, lab, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)

    hole_ids = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        touches_border = (x == 0) or (y == 0) or (x + w == W) or (y + h == H)
        if not touches_border:
            hole_ids.append((i, area))

    if not hole_ids:
        return bin255

    if max_hole_pixels is None:
        max_hole_pixels = int(max_hole_frac * box_area)

    roi_filled = roi.copy()
    for i, area in hole_ids:
        if area <= max_hole_pixels:
            roi_filled[lab == i] = 1

    out = A.copy()
    out[y0:y1+1, x0:x1+1] = roi_filled
    return (out * 255).astype(np.uint8)

def endpoint_count_8conn(skel01: np.ndarray) -> int:
    """Count endpoints in a 0/1 skeleton: pixels with exactly 1 neighbor (8-connected)."""
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
    return str(int(e))  # "0".."5"

def bin_blobs(n: int) -> str:
    if n is None:
        return "UNK"
    if n >= 3:
        return "3+"
    return str(int(n))  # "0","1","2"

def list_images(folder, exts=("png","jpg","jpeg","bmp","tif","tiff")):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(folder, f"*.{e}"))
    return sorted(paths)


# =========================
# Rule-path instrumentation
# (mirrors your classifier logic, but only for logging)
# =========================

def trace_classifier_path(A: np.ndarray, debug=False):
    """
    Returns a dict describing which branch the digit went through and the key intermediate counts.

    Mirrors:
      - Group select using THICK blobs on A
      - Group1: stems on THIN
      - Group2: helper lines TL->BL, TL->BR, TR->BR and blob/stem outcomes
    """
    out = {}

    A01 = (A > 0).astype(np.uint8)
    A_thin = thin(A01)

    # group select is based on THICK blobs (your classify_with_blobs_from_A)
    blobs_thick, n_blobs_thick = find_blobs(A01)
    out["n_blobs_thick"] = int(n_blobs_thick)

    # also log thin blobs (useful feature)
    blobs_thin, n_blobs_thin = find_blobs(A_thin)
    out["n_blobs_thin"] = int(n_blobs_thin)

    # endpoints on thin skeleton
    out["endpoints"] = endpoint_count_8conn(A_thin)

    if n_blobs_thick > 0:
        out["group"] = "Group 1"

        # In your code, group1 uses stems/topology on THIN but passes n_blobs_thick into classify_group1.
        # Log stems computed on THIN blobs.
        blobs_for_stems, nb_for_stems = find_blobs(A_thin)
        stems_img, n_stems, stem_cents = get_stems(A_thin, blobs_for_stems)
        out["g1_n_stems"] = int(n_stems)

        # Optional: log "1-blob vs 2-blob" seen on THIN too
        out["g1_nb_thin_for_stems"] = int(nb_for_stems)

    else:
        out["group"] = "Group 2"

        # helper line pipeline
        pts = get_banded_points(A_thin, split=0.5)
        if pts is None:
            out["g2_pts_ok"] = 0
            # mark as missing
            out["g2_nb1_tlbl"] = None
            out["g2_nb2_tlbr"] = None
            out["g2_nb3_trbr"] = None
            out["g2_n_stems_after_tlbl"] = None
            out["g2_case"] = "no_pts"
            return out

        out["g2_pts_ok"] = 1
        TL, BL, TR, BR = pts

        # TL->BL
        A1 = draw_line(A_thin, TL, BL)
        _, nb1 = find_blobs(A1)
        out["g2_nb1_tlbl"] = int(nb1)

        if nb1 == 0:
            # TL->BR (your code returns 4 regardless; we log nb2 anyway)
            A2 = draw_line(A_thin, TL, BR)
            _, nb2 = find_blobs(A2)
            out["g2_nb2_tlbr"] = int(nb2)
            out["g2_nb3_trbr"] = 0
            out["g2_n_stems_after_tlbl"] = 0
            out["g2_case"] = "nb1=0_then_tlbr"

        else:
            # stems after TL->BL to split {2,5} vs {3,7} / "1"
            blobs1, _ = find_blobs(A1)
            _, n_stems, _ = get_stems(A1, blobs1)
            out["g2_n_stems_after_tlbl"] = int(n_stems)

            if n_stems == 2:
                out["g2_case"] = "nb1>0_stems=2"
                out["g2_nb2_tlbr"] = 0
                out["g2_nb3_trbr"] = 0

            elif n_stems > 0:
                out["g2_case"] = "nb1>0_stems>0"
                out["g2_nb2_tlbr"] = 0
                out["g2_nb3_trbr"] = 0

            else:
                # TR->BR to split 3 vs 7
                A3 = draw_line(A_thin, TR, BR)
                _, nb3 = find_blobs(A3, min_blob_area=30)
                out["g2_nb3_trbr"] = int(nb3)
                out["g2_nb2_tlbr"] = 0
                out["g2_case"] = "nb1>0_stems=0_then_trbr"

    return out


# =========================
# Evaluation
# =========================

def eval_one_digit_folder(folder_path: str, true_digit: int):
    paths = list_images(folder_path)
    if not paths:
        return 0, 0, 0.0, {}, None, {}, {}, {}, []

    correct = 0
    confusion = {}

    # histograms
    ep_hist = {}
    thick_blob_hist = {}
    thin_blob_hist = {}

    # feature accumulators (means)
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

    # rule-path counts (categorical)
    # we’ll return a list of per-sample trace dicts and later aggregate
    traces = []

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = resize_to_height(img, target_h=240)
        if img is None:
            continue

        A, _, _ = preprocess_step1(img, visualize=False)

        A255 = fill_small_holes((A > 0).astype(np.uint8) * 255, max_hole_frac=0.02)
        A  = (A255 > 0).astype(np.uint8)   # back to 0/1

        digit_pred, group = classify_with_blobs_from_A(A, visualize=False, debug=False)

        if digit_pred is None:
            pred = "UNK"
        else:
            pred = str(int(digit_pred))

        if pred == str(true_digit):
            correct += 1
        confusion[pred] = confusion.get(pred, 0) + 1

        # trace / features
        tr = trace_classifier_path(A)
        tr["true"] = int(true_digit)
        tr["pred"] = pred
        traces.append(tr)

        e = tr.get("endpoints", None)
        nb_thick = tr.get("n_blobs_thick", None)
        nb_thin = tr.get("n_blobs_thin", None)

        # hists
        b = bin_endpoints(e)
        ep_hist[b] = ep_hist.get(b, 0) + 1

        bb = bin_blobs(nb_thick)
        thick_blob_hist[bb] = thick_blob_hist.get(bb, 0) + 1

        bb2 = bin_blobs(nb_thin)
        thin_blob_hist[bb2] = thin_blob_hist.get(bb2, 0) + 1

        # means
        feats["endpoints"].append(e)
        feats["n_blobs_thick"].append(nb_thick)
        feats["n_blobs_thin"].append(nb_thin)

        if tr.get("group") == "Group 1":
            feats["group1_n_stems"].append(tr.get("g1_n_stems", 0))
        else:
            feats["group1_n_stems"].append(0)

        # group2 helper-line stats (0 if not group2)
        feats["g2_nb1_tlbl"].append(tr.get("g2_nb1_tlbl", 0) or 0)
        feats["g2_nb2_tlbr"].append(tr.get("g2_nb2_tlbr", 0) or 0)
        feats["g2_nb3_trbr"].append(tr.get("g2_nb3_trbr", 0) or 0)
        feats["g2_n_stems_after_tlbl"].append(tr.get("g2_n_stems_after_tlbl", 0) or 0)

    n = len(paths)
    acc = (correct / n) if n > 0 else 0.0

    feat_means = None
    if n > 0 and len(feats["endpoints"]) > 0:
        feat_means = {
            "Digit": int(true_digit),
            "Samples": int(n),
            "Accuracy_%": round(acc * 100, 2),

            "endpoints_mean": float(np.mean(feats["endpoints"])),
            "blobs_thick_mean": float(np.mean(feats["n_blobs_thick"])),
            "blobs_thin_mean": float(np.mean(feats["n_blobs_thin"])),

            "group1_stems_mean": float(np.mean(feats["group1_n_stems"])),

            "g2_nb1_tlbl_mean": float(np.mean(feats["g2_nb1_tlbl"])),
            "g2_nb2_tlbr_mean": float(np.mean(feats["g2_nb2_tlbr"])),
            "g2_nb3_trbr_mean": float(np.mean(feats["g2_nb3_trbr"])),
            "g2_stems_after_tlbl_mean": float(np.mean(feats["g2_n_stems_after_tlbl"])),
        }

    return n, correct, acc, confusion, feat_means, ep_hist, thick_blob_hist, thin_blob_hist, traces


def aggregate_rulepaths(traces_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a per-TRUE-digit summary of the *paths* taken through the classifier.

    You asked things like:
      - "how many 7s: TLBL creates a blob AND achieves a blob"
      - "how many 7s: TRBR creates a blob"
    Those are direct counts below.
    """
    rows = []

    for d in range(10):
        sub = traces_df[traces_df["true"] == d].copy()
        if len(sub) == 0:
            continue

        # group counts
        g1 = (sub["group"] == "Group 1").sum()
        g2 = (sub["group"] == "Group 2").sum()

        # helper-line outcomes (only meaningful for group2)
        g2sub = sub[sub["group"] == "Group 2"]

        # TL->BL results
        tlbl_nb1_pos = (g2sub["g2_nb1_tlbl"].fillna(0) > 0).sum()
        tlbl_nb1_zero = (g2sub["g2_nb1_tlbl"].fillna(0) == 0).sum()

        # TL->BR results (only when nb1==0 in your logic, but we log it)
        tlbr_pos = (g2sub["g2_nb2_tlbr"].fillna(0) > 0).sum()

        # TR->BR results (only when stems==0 in your logic, but we log it)
        trbr_pos = (g2sub["g2_nb3_trbr"].fillna(0) > 0).sum()
        trbr_zero = (g2sub["g2_nb3_trbr"].fillna(0) == 0).sum()

        # stems after TLBL
        stems2 = (g2sub["g2_n_stems_after_tlbl"].fillna(0) == 2).sum()
        stems_pos = (g2sub["g2_n_stems_after_tlbl"].fillna(0) > 0).sum()
        stems_zero = (g2sub["g2_n_stems_after_tlbl"].fillna(0) == 0).sum()

        # case labels
        case_counts = g2sub["g2_case"].fillna("NA").value_counts().to_dict()

        row = {
            "Digit": d,
            "Samples": int(len(sub)),
            "Group1_count": int(g1),
            "Group2_count": int(g2),

            # Your “7 analysis” type questions live here:
            "G2_TLBL_nb1>0_count": int(tlbl_nb1_pos),
            "G2_TLBL_nb1==0_count": int(tlbl_nb1_zero),
            "G2_TLBR_nb2>0_count": int(tlbr_pos),
            "G2_TRBR_nb3>0_count": int(trbr_pos),
            "G2_TRBR_nb3==0_count": int(trbr_zero),

            "G2_stems==2_count": int(stems2),
            "G2_stems>0_count": int(stems_pos),
            "G2_stems==0_count": int(stems_zero),
        }

        # expand the case counts into columns (stable names)
        for k, v in case_counts.items():
            row[f"case:{k}"] = int(v)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("Digit")


def eval_all_digits(root_digits_dir: str):
    """
    Expects structure like:
      images/variations_digits/0/*.png
      images/variations_digits/1/*.png
      ...
      images/variations_digits/9/*.png
    Folder name is the TRUE digit.
    """
    subdirs = [d for d in os.listdir(root_digits_dir)
               if os.path.isdir(os.path.join(root_digits_dir, d))]
    subdirs = sorted([d for d in subdirs if d.startswith("number")])

    if not subdirs:
        raise FileNotFoundError(f"No digit subfolders (0–9) found in: {root_digits_dir}")

    rows = []
    feature_rows = []

    all_total = 0
    all_correct = 0

    confusion_matrix = {}
    endpoint_hists = {}
    thick_blob_hists = {}
    thin_blob_hists = {}
    all_traces = []

    for sd in subdirs:
        d = int(sd.replace("number", ""))
        folder = os.path.join(root_digits_dir, sd)

        n, correct, acc, confusion, feat_means, ep_hist, thick_hist, thin_hist, traces = \
            eval_one_digit_folder(folder, d)

        rows.append({
            "Digit": d,
            "Samples": n,
            "Correct": correct,
            "Accuracy_%": round(acc * 100, 2),
        })

        if feat_means is not None:
            feature_rows.append(feat_means)

        confusion_matrix[str(d)] = confusion
        endpoint_hists[str(d)] = ep_hist
        thick_blob_hists[str(d)] = thick_hist
        thin_blob_hists[str(d)] = thin_hist
        all_traces += traces

        all_total += n
        all_correct += correct
        print(f"{d}: {correct}/{n} = {acc*100:.2f}%")

    overall_acc = (all_correct / all_total) if all_total > 0 else 0.0
    print("\n====================")
    print(f"OVERALL: {all_correct}/{all_total} = {overall_acc*100:.2f}%")
    print("====================\n")

    results_df = pd.DataFrame(rows).sort_values("Digit")
    features_df = pd.DataFrame(feature_rows).sort_values("Digit")

    # confusion matrix dataframe: true rows x predicted cols
    all_preds = sorted({pred for cm in confusion_matrix.values() for pred in cm.keys()},
                       key=lambda x: (x == "UNK", x))  # keep UNK at end

    cm_df = pd.DataFrame(0, index=[str(i) for i in range(10)], columns=all_preds)
    for t in [str(i) for i in range(10)]:
        for p, cnt in confusion_matrix.get(t, {}).items():
            if p not in cm_df.columns:
                cm_df[p] = 0
            cm_df.loc[t, p] = cnt

    # endpoint bin matrix
    ep_bins = ["0","1","2","3","4","5","6+","UNK"]
    endpoint_df = pd.DataFrame(0, index=[str(i) for i in range(10)], columns=ep_bins)
    for t in [str(i) for i in range(10)]:
        hist = endpoint_hists.get(t, {})
        for b, cnt in hist.items():
            if b not in endpoint_df.columns:
                endpoint_df[b] = 0
            endpoint_df.loc[t, b] = cnt

    # blob bin matrices (thick + thin)
    blob_bins = ["0","1","2","3+","UNK"]

    blob_thick_df = pd.DataFrame(0, index=[str(i) for i in range(10)], columns=blob_bins)
    for t in [str(i) for i in range(10)]:
        hist = thick_blob_hists.get(t, {})
        for b, cnt in hist.items():
            if b not in blob_thick_df.columns:
                blob_thick_df[b] = 0
            blob_thick_df.loc[t, b] = cnt

    blob_thin_df = pd.DataFrame(0, index=[str(i) for i in range(10)], columns=blob_bins)
    for t in [str(i) for i in range(10)]:
        hist = thin_blob_hists.get(t, {})
        for b, cnt in hist.items():
            if b not in blob_thin_df.columns:
                blob_thin_df[b] = 0
            blob_thin_df.loc[t, b] = cnt

    # rule-path counts
    traces_df = pd.DataFrame(all_traces)
    rulepath_df = aggregate_rulepaths(traces_df)

    return results_df, cm_df, features_df, endpoint_df, blob_thick_df, blob_thin_df, rulepath_df, traces_df

def summarize_line_blob_sizes(root_digits_dir: str):
    """
    Compare average blob sizes for digits 3 and 7 after:
      - A2 = TL->BR
      - A3 = TR->BR
    """
    all_dfs = []

    for d in [3, 7]:
        folder = os.path.join(root_digits_dir, f"number{d}")
        df = analyze_line_blob_sizes_for_digit_folder(folder, d)
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        return None, None

    full_df = pd.concat(all_dfs, ignore_index=True)

    summary = full_df.groupby("true_digit").agg({
        "A2_nb": "mean",
        "A2_largest_blob": "mean",
        "A2_total_blob_area": "mean",
        "A3_nb": "mean",
        "A3_largest_blob": "mean",
        "A3_total_blob_area": "mean",
    }).reset_index()

    return full_df, summary
if __name__ == "__main__":
    ROOT = r"C:\Users\jadsa\GitRepos\final-year-project-jad\images\variations_numbers"

    # ---------------------------------
    # Create experiment folder
    # ---------------------------------
    EXP_NAME = "original_implementation"
    OUT_DIR = os.path.join(ROOT, EXP_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)

    (results_df, confusion_df, features_df,
     endpoint_df, blob_thick_df, blob_thin_df,
     rulepath_df, traces_df) = eval_all_digits(ROOT)

    # Save outputs
    results_df.to_csv(os.path.join(ROOT, "digit_eval_results_original.csv"), index=False)
    confusion_df.to_csv(os.path.join(ROOT, "digit_eval_confusion_original.csv"))
    endpoint_df.to_csv(os.path.join(ROOT, "digit_eval_endpoints_original.csv"))

    blob_thick_df.to_csv(os.path.join(ROOT, "digit_eval_blobs_thick_original.csv"))
    blob_thin_df.to_csv(os.path.join(ROOT, "digit_eval_blobs_thin_original.csv"))

    features_df.to_csv(os.path.join(ROOT, "digit_feature_averages_original.csv"), index=False)
    rulepath_df.to_csv(os.path.join(ROOT, "digit_rulepath_counts_original.csv"), index=False)

    # OPTIONAL: full per-sample trace dump (very useful for debugging)
    traces_df.to_csv(os.path.join(ROOT, "digit_traces_per_sample_original.csv"), index=False)

    line_blob_df, line_blob_summary_df = summarize_line_blob_sizes(ROOT)

    if line_blob_df is not None:
        line_blob_df.to_csv(os.path.join(ROOT, "digit_line_blob_sizes_3_7_original.csv"), index=False)
        line_blob_summary_df.to_csv(os.path.join(ROOT, "digit_line_blob_sizes_3_7_summary_original.csv"), index=False)

        print("Saved digit_line_blob_sizes_3_7.csv")
        print("Saved digit_line_blob_sizes_3_7_summary.csv")
        print(line_blob_summary_df)

    print("Saved digit_eval_results.csv")
    print("Saved digit_eval_confusion.csv")
    print("Saved digit_eval_endpoints.csv")
    print("Saved digit_eval_blobs_thick.csv")
    print("Saved digit_eval_blobs_thin.csv")
    print("Saved digit_feature_averages.csv")
    print("Saved digit_rulepath_counts.csv")
    print("Saved digit_traces_per_sample.csv (optional)")