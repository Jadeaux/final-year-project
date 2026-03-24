import os
import glob
import cv2
import numpy as np
import pandas as pd

# ---- import YOUR pipeline ----
# Adjust these imports to match your project structure.
# If you're running this from inside "modules/", you might need:
# from modules.preprocess_for_segmented import preprocess_letters
# from modules.preprocessing import thin
# from modules.classification_letters import classify_letter

from modules.preprocess_for_segmented import preprocess_letters
from modules.preprocessing import thin
from modules.classification_letters_data import classify_letter
from modules.features_letters import (
    prune_spurs,
    count_endpoints,
    hole_count_and_largest_pct,
    vertical_symmetry_lr_balance,
    horizontal_symmetry_tb_balance,
    bottom_width_ratio,
    center_density_ratio,
    concavity_tb_strength,
    concavity_lr_strength,
    count_vertical_strokes,
    count_horizontal_strokes,
    side_open_score,
    count_holes
)


import cv2
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def debug_show_bbox_and_lr_symmetry(A_bbox: np.ndarray, sym_score: float, title=""):
    """
    Visualize the bbox mask and how LR symmetry compares left vs right.
    Shows:
      - A_bbox
      - Left half
      - Right half (flipped to align with left)
      - Overlap (agreement) map
      - Difference map
    """
    A = (A_bbox > 0).astype(np.uint8)
    h, w = A.shape

    mid = w // 2
    left  = A[:, :mid]
    right = A[:, w - mid:]          # take same width as left
    right_flipped = np.fliplr(right)

    # agreement / difference
    overlap = (left & right_flipped).astype(np.uint8)
    diff = (left ^ right_flipped).astype(np.uint8)

    fig, axs = plt.subplots(1, 5, figsize=(14, 3))
    axs[0].imshow(A, cmap="gray")
    axs[0].set_title(f"BBox {title}\nshape={A.shape}")

    axs[1].imshow(left, cmap="gray")
    axs[1].set_title("Left half")

    axs[2].imshow(right_flipped, cmap="gray")
    axs[2].set_title("Right half (flipped)")

    axs[3].imshow(overlap, cmap="gray")
    axs[3].set_title("Overlap (AND)")

    axs[4].imshow(diff, cmap="gray")
    axs[4].set_title("Difference (XOR)")

    for ax in axs:
        ax.axis("off")

    fig.suptitle(f"vertical_symmetry_lr_balance(A_bbox) = {sym_score:.3f}", y=1.05)
    plt.tight_layout()
    plt.show()

def endpoint_bin(e: int):
    if e is None:
        return "UNK"
    if e >= 5:
        return "5+"
    return str(int(e))  # "0","1","2","3","4"

def preclean_synthetic_gray(img_gray: np.ndarray,
                            min_area: int = 60,
                            close_ksize: int = 3,
                            close_iters: int = 1,
                            open_ksize: int = 0,
                            open_iters: int = 0) -> np.ndarray:
    """
    Light cleanup for synthetic chars BEFORE preprocess_letters().
    - Binarize with Otsu
    - Remove tiny CC speckles
    - Small morphological close (helps stop endpoint-splitting at feet)
    - Optional small open

    Returns a GRAYSCALE image (0/255) that still looks like a char mask.
    """
    g = img_gray.copy()

    # Otsu binarize
    blur = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ensure foreground is white, background black
    # (if it's inverted, flip)
    if np.mean(bw) > 127:
        bw = cv2.bitwise_not(bw)

    # Remove tiny connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    den = np.zeros_like(bw)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            den[labels == i] = 255

    # Small close to remove tiny notches/jaggies
    if close_ksize > 0 and close_iters > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        den = cv2.morphologyEx(den, cv2.MORPH_CLOSE, k_close, iterations=close_iters)

    # Optional open to remove micro spurs
    if open_ksize > 0 and open_iters > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        den = cv2.morphologyEx(den, cv2.MORPH_OPEN, k_open, iterations=open_iters)

    return den  # still grayscale 0/255
def eval_one_folder(folder_path: str, true_label: str, exts=("png", "jpg", "jpeg")):
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder_path, f"*.{e}")))
    paths = sorted(paths)

    if not paths:
        return 0, 0, 0.0, {}, None

    correct = 0
    confusion = {}

    # accumulate features
    feats = {
        "endpoints": [],
        "v_sym": [],
        "h_sym": [],
        "hole_count": [],
        "hole_pct": [],
        "bw": [],
        "cd": [],
        "north": [],
        "south": [],
        "east": [],
        "west": [],
        "vstroke": [],
        "hstroke": [],
        "endpoints": [],
        "endpoints_bin": [], 
        "left_ratio": [],
        "right_ratio": [],
        "blobs": []
    }

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = preclean_synthetic_gray(img, min_area=60, close_ksize=3, close_iters=1)

        A, cleaned, bin_norm = preprocess_letters(img, visualize=False)
        skel = thin(A)

        pred = classify_letter(A, skel)
        if pred is None:
            pred = "UNK"

        if pred == true_label:
            correct += 1
        confusion[pred] = confusion.get(pred, 0) + 1

        # ---- FEATURE EXTRACTION (no change to classifier) ----
        sk_pruned = prune_spurs(skel, max_length=2)

        ys, xs = np.where(A > 0)

        if xs.size == 0:
            continue  # empty mask edge-case

        pad = 2  # try 2–4
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        y0 = max(0, y0 - pad); y1 = min(A.shape[0]-1, y1 + pad)
        x0 = max(0, x0 - pad); x1 = min(A.shape[1]-1, x1 + pad)

        A_bbox = A[y0:y1+1, x0:x1+1]
        sk_bbox = sk_pruned[y0:y1+1, x0:x1+1]
        

        e  = count_endpoints(sk_bbox)

        vs = vertical_symmetry_lr_balance(A_bbox)
        hs = horizontal_symmetry_tb_balance(A_bbox)
        

        hc, hpct = hole_count_and_largest_pct(A_bbox)#
        blobs = hc
        bw = bottom_width_ratio(A_bbox, band_frac=0.20)
        cd = center_density_ratio(A_bbox, frac=0.35)

        north, south = concavity_tb_strength(A_bbox)
        left_ratio, right_ratio = side_open_score(A_bbox)

        # IMPORTANT: your function returns (west, east)
        west, east = concavity_lr_strength(A_bbox)

        vstroke = count_vertical_strokes(
        A_bbox,
        min_frac=0.6,
        gap_allow=2,
        support_cols=3,
        orient_thresh=0.12
    )
        hstroke = count_horizontal_strokes(
            A_bbox,
            min_frac=0.8,
            gap_allow=2,
            band=5,
            support_rows=3,
            orient_thresh=0.12,
            rel_width_keep=0.5
        )


        feats["v_sym"].append(vs)
        feats["h_sym"].append(hs)
        feats["hole_count"].append(hc)
        feats["hole_pct"].append(hpct)
        feats["bw"].append(bw)
        feats["cd"].append(cd)
        feats["north"].append(north)
        feats["south"].append(south)
        feats["east"].append(east)
        feats["west"].append(west)
        feats["vstroke"].append(vstroke)
        feats["hstroke"].append(hstroke)
        feats["endpoints"].append(e)
        feats["endpoints_bin"].append(endpoint_bin(e))
        feats["left_ratio"].append(left_ratio)
        feats["right_ratio"].append(right_ratio)
        feats["blobs"].append(blobs)

    n = len(paths)
    acc = (correct / n) if n > 0 else 0.0

    # compute means (guard empty)
    feat_means = None
    if n > 0 and len(feats["endpoints"]) > 0:
        feat_means = {
            "Label": true_label,
            "Samples": n,
            "Accuracy_%": round(acc * 100, 2),
            "endpoints_mean": float(np.mean(feats["endpoints"])),
            "v_sym_mean": float(np.mean(feats["v_sym"])),
            "h_sym_mean": float(np.mean(feats["h_sym"])),
            "hole_count_mean": float(np.mean(feats["hole_count"])),
            "hole_pct_mean": float(np.mean(feats["hole_pct"])),
            "bw_mean": float(np.mean(feats["bw"])),
            "cd_mean": float(np.mean(feats["cd"])),
            "north_mean": float(np.mean(feats["north"])),
            "south_mean": float(np.mean(feats["south"])),
            "vstroke_mean": float(np.mean(feats["vstroke"])),
            "hstroke_mean": float(np.mean(feats["hstroke"])),
            "east_mean": float(np.mean(feats["east"])),
            "west_mean": float(np.mean(feats["west"])),
            "left_ratio_mean": float(np.mean(feats["left_ratio"])),
            "right_ratio_mean": float(np.mean(feats["right_ratio"])),
            "blobs_mean": float(np.mean(feats["blobs"]))
        }

    ep_hist = {}
    for b in feats["endpoints_bin"]:
        ep_hist[b] = ep_hist.get(b, 0) + 1

    blob_hist = {}
    for b in feats["blobs"]:
        key = "UNK" if b is None else ("3+" if b >= 3 else str(int(b)))  # 0,1,2,3+
        blob_hist[key] = blob_hist.get(key, 0) + 1

    return n, correct, acc, confusion, feat_means, ep_hist, blob_hist

def eval_all_letters(root_variations_dir: str):
    """
    Expects structure like:
      images/variations/A/*.png
      images/variations/B/*.png
      ...
    Folder name is the TRUE label.
    """
    # Find subfolders like A, B, C...
    subdirs = [d for d in os.listdir(root_variations_dir)
               if os.path.isdir(os.path.join(root_variations_dir, d))]

    # Keep only single-char folders (A-Z, 0-9 etc). Adjust if needed.
    subdirs = sorted([d for d in subdirs if len(d) == 1])

    if not subdirs:
        raise FileNotFoundError(f"No label subfolders found in: {root_variations_dir}")

    rows = []
    all_total = 0
    all_correct = 0

    # Build a confusion matrix dict: true -> pred -> count
    confusion_matrix = {}
    endpoint_hists = {}  # true_label -> endpoint_bin -> count
    blob_hists = {}  # true_label -> blob_bin -> count
    feature_rows = []

    for label in subdirs:
        folder = os.path.join(root_variations_dir, label)
        n, correct, acc, confusion, feat_means, ep_hist, blob_hist = eval_one_folder(folder, label)
        endpoint_hists[label] = ep_hist
        blob_hists[label] = blob_hist

        

        rows.append({
            "Label": label,
            "Samples": n,
            "Correct": correct,
            "Accuracy_%": round(acc * 100, 2),
        })

        if feat_means is not None:
            feature_rows.append(feat_means)

        all_total += n
        all_correct += correct
        confusion_matrix[label] = confusion

        print(f"{label}: {correct}/{n} = {acc*100:.2f}%")

    overall_acc = (all_correct / all_total) if all_total > 0 else 0.0
    print("\n====================")
    print(f"OVERALL: {all_correct}/{all_total} = {overall_acc*100:.2f}%")
    print("====================\n")

    df = pd.DataFrame(rows).sort_values("Label")

    # Turn confusion dict into a DataFrame (true rows, pred cols)
    # collect all predicted labels that appeared
    all_preds = sorted({pred for cm in confusion_matrix.values() for pred in cm.keys()})
    cm_df = pd.DataFrame(0, index=subdirs, columns=all_preds)
    for t in subdirs:
        for p, cnt in confusion_matrix[t].items():
            cm_df.loc[t, p] = cnt
    
    #create endpoint matrix
    features_df = pd.DataFrame(feature_rows).sort_values("Label")

    bins = ["0", "1", "2", "3", "4", "5+", "UNK"]  # choose what you want
    endpoint_cm = pd.DataFrame(0, index=subdirs, columns=bins)

    for lbl in subdirs:
        hist = endpoint_hists.get(lbl, {})
        for b, cnt in hist.items():
            if b not in endpoint_cm.columns:
                endpoint_cm[b] = 0
            endpoint_cm.loc[lbl, b] = cnt
    
    blob_bins = ["0", "1", "2", "3+", "UNK"]
    blob_df = pd.DataFrame(0, index=subdirs, columns=blob_bins)

    for lbl in subdirs:
        hist = blob_hists.get(lbl, {})
        for b, cnt in hist.items():
            if b not in blob_df.columns:
                blob_df[b] = 0
            blob_df.loc[lbl, b] = cnt

    
    return df, cm_df, features_df, endpoint_cm, blob_df



if __name__ == "__main__":
    ROOT = r"C:\Users\jadsa\GitRepos\final-year-project-jad\images\variations"

    results_df, confusion_df, features_df, endpoint_df, blob_df = eval_all_letters(ROOT)
    # Save outputs
    out_csv = os.path.join(ROOT, "synthetic_eval_results_bbox.csv")
    out_cm  = os.path.join(ROOT, "synthetic_eval_confusion_bbox.csv")
    out_ep  = os.path.join(ROOT, "synthetic_eval_endpoint_bbox.csv")
    out_blob = os.path.join(ROOT, "synthetic_eval_blobs_bbox.csv")

    results_df.to_csv(out_csv, index=False)
    confusion_df.to_csv(out_cm)
    endpoint_df.to_csv(out_ep)
    blob_df.to_csv(out_blob, index=False)

    out_feat = os.path.join(ROOT, "synthetic_feature_averages_bbox.csv")
    features_df.to_csv(out_feat, index=False)
    print("Saved:", out_feat)
    print("Saved:", out_ep)
    print("Saved:", out_csv)
    print("Saved:", out_cm)
    print("Saved:", out_blob)
    