"""
license_plate_batch.py  (keep EVERYTHING in this file)

Batch pipeline:
1) localize_plate_roi (blackhat blob)
2) segment_word_mask_from_roi (threshold -> denoise -> keep_main_plate_chars -> word_mask)
3) segment_chars_from_word_mask (CC boxes -> normalize)
4) classify plate (digits via classify_with_blobs_from_A, letters via classify_letter)

Run:
python -m modules.license_plate_batch --plates_dir images/plates --out_csv lpr_outputs/predictions.csv --debug_dir lpr_outputs/debug_batch
"""

import os, glob, csv, argparse
import cv2
import numpy as np

# ---- your project imports (KEEP these) ----
from .classification_letters import classify_letter
from .preprocess_for_segmented import preprocess_letters as preprocess_segmented
from .preprocessing import thin
from .features_letters import prune_spurs, count_endpoints
from .classification import classify_with_blobs_from_A


# ============================================================
# Small utilities
# ============================================================

def pad_bbox(x, y, w, h, H, W, pad_x_frac=0.06, pad_y_frac=0.20):
    pad_x = int(pad_x_frac * w)
    pad_y = int(pad_y_frac * h)
    x2 = max(0, x - pad_x)
    y2 = max(0, y - pad_y)
    x3 = min(W, x + w + pad_x)
    y3 = min(H, y + h + pad_y)
    return x2, y2, (x3 - x2), (y3 - y2)


def normalize_plate_size(plate_roi, min_ok_w=400, target_w=600, max_ok_w=600):
    h, w = plate_roi.shape[:2]
    if min_ok_w <= w <= max_ok_w:
        return plate_roi, 1.0
    scale = target_w / float(w)
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    plate_resized = cv2.resize(plate_roi, None, fx=scale, fy=scale, interpolation=interp)
    return plate_resized, scale


def bridge_gaps(bin255):
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    out = cv2.morphologyEx(bin255, cv2.MORPH_CLOSE, k_close, iterations=1)
    out = cv2.dilate(out, k_close, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=1)
    return out


def fill_small_holes(bin255, max_hole_frac=0.05, max_hole_pixels=None):
    """Fill small enclosed background holes inside ONE character (0/255)."""
    A = (bin255 > 0).astype(np.uint8)
    ys, xs = np.where(A == 1)
    if xs.size == 0:
        return bin255

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    roi = A[y0:y1 + 1, x0:x1 + 1]
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
    out[y0:y1 + 1, x0:x1 + 1] = roi_filled
    return (out * 255).astype(np.uint8)


def char_type_from_index(i: int) -> str:
    # California style: D L L L D D D
    if i == 0:
        return "digit"
    if 1 <= i <= 3:
        return "letter"
    return "digit"


# ============================================================
# keep_main_plate_chars (yours, unchanged)
# ============================================================

def keep_main_plate_chars(bin255,
                          min_area=120,
                          h_lo=0.55, h_hi=0.98,
                          ar_lo=0.12, ar_hi=1.10,
                          border_margin=1,
                          border_tall_frac=0.60,
                          max_gap_factor=0.60):
    """
    bin255: 0/255 image, white = foreground
    Returns: cleaned 0/255 mask with only the main character group.
    """
    A = (bin255 > 0).astype(np.uint8) * 255
    H, W = A.shape

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(A, connectivity=8)

    comps = []
    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]

        if area < min_area:
            continue

        if h < h_lo * H or h > h_hi * H:
            continue

        ar = w / (h + 1e-6)
        if ar < ar_lo or ar > ar_hi:
            continue

        touches_left = x <= border_margin
        touches_right = (x + w) >= (W - border_margin)
        if (touches_left or touches_right) and (h >= border_tall_frac * H):
            continue

        comps.append((i, x, y, w, h, area, cx, cy))

    if len(comps) == 0:
        return A

    comps.sort(key=lambda t: t[1])

    widths = np.array([c[3] for c in comps], dtype=np.float32)
    w_med = float(np.median(widths))

    best_run = []
    run = [comps[0]]
    for prev, cur in zip(comps[:-1], comps[1:]):
        prev_x, prev_w = prev[1], prev[3]
        cur_x = cur[1]
        gap = cur_x - (prev_x + prev_w)

        if gap <= max_gap_factor * w_med:
            run.append(cur)
        else:
            if len(run) > len(best_run):
                best_run = run
            run = [cur]
    if len(run) > len(best_run):
        best_run = run

    keep_ids = set([c[0] for c in best_run])

    out = np.zeros_like(A)
    for i in keep_ids:
        out[labels == i] = 255

    return out


# ============================================================
# 1) Plate localisation (your blackhat blob)
# ============================================================

def localize_plate_roi(img_bgr: np.ndarray):
    H, W = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    k_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_bh)

    _, bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=2)

    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    bw = cv2.dilate(bw, k_dil, iterations=1)

    margin = 10
    bw[:margin, :] = 0
    bw[-margin:, :] = 0
    bw[:, :margin] = 0
    bw[:, -margin:] = 0

    mask_roi = np.zeros_like(bw)
    y_start = int(0.35 * H)
    y_end = int(0.90 * H)
    mask_roi[y_start:y_end, :] = 255
    bw = cv2.bitwise_and(bw, mask_roi)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h + 1e-6)
        area = w * h

        if ar < 1.6 or ar > 8.0:
            continue
        if area < 0.01 * W * H or area > 0.80 * W * H:
            continue
        if y < 0.20 * H or y > 0.80 * H:
            continue

        cx = x + w / 2
        cy = y + h / 2
        center_dist = abs(cx - W / 2) / W + abs(cy - H / 2) / H
        score = (w * h) / (1.0 + 5.0 * center_dist)

        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        return None

    x, y, w, h = best
    x, y, w, h = pad_bbox(x, y, w, h, H, W)
    return img_bgr[y:y + h, x:x + w].copy()


# ============================================================
# 2) Build a clean word_mask from plate ROI
#    (this is where your W/M “gaps” were coming from)
# ============================================================

def segment_word_mask_from_roi(plate_roi: np.ndarray, debug_dir: str = None, stem: str = ""):
    # --- binarize
    pgray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    pgray = cv2.GaussianBlur(pgray, (3, 3), 0)

    chars = cv2.adaptiveThreshold(
        pgray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    # --- main row band by row projection
    proj = np.sum(chars > 0, axis=1).astype(np.float32)
    proj_s = cv2.GaussianBlur(proj.reshape(-1, 1), (1, 31), 0).flatten()
    proj_n = proj_s / (proj_s.max() + 1e-6)

    pos = proj_n[proj_n > 0]
    thr = 0.10 if pos.size == 0 else max(0.10, 0.5 * np.median(pos))
    mask_rows = (proj_n > thr).astype(np.uint8)

    runs = []
    start = None
    for i, v in enumerate(mask_rows):
        if v == 1 and start is None:
            start = i
        if v == 0 and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask_rows) - 1))

    if runs:
        top, bottom = max(runs, key=lambda r: r[1] - r[0])
        pad = 3
        top = max(0, top - pad)
        bottom = min(chars.shape[0] - 1, bottom + pad)
        chars = chars[top:bottom + 1, :]
        plate_roi = plate_roi[top:bottom + 1, :]

    # --- denoise speckles FIRST
    num, labels, stats, _ = cv2.connectedComponentsWithStats(chars, connectivity=8)
    min_area = 80
    chars_denoised = np.zeros_like(chars)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            chars_denoised[labels == i] = 255

    # optional light close (don’t overdo)
    chars_denoised = cv2.morphologyEx(
        chars_denoised, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1
    )

    # --- IMPORTANT: keep_main_plate_chars on the DENOISED mask (your fix)
    chars_clean = keep_main_plate_chars(chars_denoised, min_area=120)
    if np.count_nonzero(chars_clean) == 0:
        chars_clean = chars_denoised

    # --- tight crop by ink bbox (now safe)
    ys, xs = np.where(chars_clean > 0)
    if xs.size == 0:
        return None, None, None

    pad = 6
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); x1 = min(chars_clean.shape[1] - 1, x1 + pad)
    y0 = max(0, y0 - pad); y1 = min(chars_clean.shape[0] - 1, y1 + pad)

    chars_clean = chars_clean[y0:y1 + 1, x0:x1 + 1]
    plate_roi = plate_roi[y0:y1 + 1, x0:x1 + 1]

    # --- (optional) bridge gaps lightly
    chars_clean = bridge_gaps(chars_clean)

    # --- save debug if requested
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_roi.png"), plate_roi)
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_chars_clean.png"), chars_clean)

    return plate_roi, chars_clean, chars_clean  # word_mask == chars_clean in this pipeline


# ============================================================
# 3) Segment characters from word_mask (CC boxes -> normalized squares)
# ============================================================

def segment_chars_from_word_mask(word_mask: np.ndarray, target_size=96, debug_dir: str = None, stem: str = ""):
    img_bin = word_mask.copy()
    h, w = img_bin.shape

    num, labels, stats, _ = cv2.connectedComponentsWithStats(img_bin, connectivity=8)

    boxes = []
    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 120:
            continue
        if ch < 0.45 * h:
            continue
        if cw < 5:
            continue

        boxes.append((x, y, cw, ch, area))

    # cap to 7 largest
    MAX_CHARS = 7
    if len(boxes) > MAX_CHARS:
        boxes.sort(key=lambda b: b[4])
        boxes = boxes[-MAX_CHARS:]

    boxes.sort(key=lambda b: b[0])

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    normalized_chars = []
    for idx, (x, y, cw, ch, _) in enumerate(boxes):
        char_img = img_bin[y:y + ch, x:x + cw].copy()

        # fix tiny “holes” + light stroke clean
        char_img = fill_small_holes(char_img, max_hole_frac=0.05)
        char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, k1, iterations=1)
        char_img = cv2.dilate(char_img, k1, iterations=1)
        char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, k2, iterations=1)

        ys, xs = np.where(char_img > 0)
        if xs.size == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        char_img = char_img[y0:y1 + 1, x0:x1 + 1]

        hh, ww = char_img.shape
        size = max(hh, ww)
        square = np.zeros((size, size), dtype=np.uint8)
        y_off = (size - hh) // 2
        x_off = (size - ww) // 2
        square[y_off:y_off + hh, x_off:x_off + ww] = char_img

        resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        normalized_chars.append(resized)

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"{stem}_char{idx:02d}.png"), resized)

    return normalized_chars


# ============================================================
# 4) Classify one plate
# ============================================================

def predict_plate_text(img_bgr: np.ndarray, debug_dir: str = None, stem: str = ""):
    plate_roi = localize_plate_roi(img_bgr)
    if plate_roi is None:
        return "", "NO_PLATE"

    plate_roi, _ = normalize_plate_size(plate_roi, min_ok_w=400, target_w=600, max_ok_w=600)

    plate_roi2, chars_clean, word_mask = segment_word_mask_from_roi(plate_roi, debug_dir=debug_dir, stem=stem)
    if word_mask is None:
        return "", "NO_CHARS"

    normalized_chars = segment_chars_from_word_mask(word_mask, target_size=96, debug_dir=debug_dir, stem=stem)
    if not normalized_chars:
        return "", "NO_CHARS"

    plate_text = ""
    for idx, char_img in enumerate(normalized_chars):
        ctype = char_type_from_index(idx)

        # ONE preprocess for both digit/letter
        A, cleaned, bin_norm = preprocess_segmented(char_img, visualize=False, plate_mode=True)

        # skeleton for endpoint logic (letters), harmless for digits
        skel = thin(A)
        sk_pruned = prune_spurs(skel, max_length=2)
        _ = count_endpoints(sk_pruned)

        if ctype == "digit":
            digit, group = classify_with_blobs_from_A(A, debug=False)
            plate_text += str(digit) if digit is not None else "?"
        else:
            pred = classify_letter(A, skel)
            plate_text += pred if pred is not None else "?"

    return plate_text, "OK"


# ============================================================
# Batch over images/plates
# ============================================================

def run_batch(plates_dir="images/plates", out_csv="lpr_outputs/predictions.csv", exts="png,jpg,jpeg,bmp", debug_dir=None):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    exts = [e.strip().lower() for e in exts.split(",")]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(plates_dir, f"*.{e}")))
        paths.extend(glob.glob(os.path.join(plates_dir, f"*.{e.upper()}")))
    paths = sorted(set(paths))

    if not paths:
        print(f"No images found in: {plates_dir}")
        return

    rows = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            rows.append((os.path.basename(p), "", "READ_FAIL"))
            continue

        stem = os.path.splitext(os.path.basename(p))[0]
        pred, status = predict_plate_text(img, debug_dir=debug_dir, stem=stem)

        print(f"{os.path.basename(p)} -> {pred} [{status}]")
        rows.append((os.path.basename(p), pred, status))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "prediction", "status"])
        w.writerows(rows)

    print(f"\nSaved: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates_dir", default="images/plates")
    ap.add_argument("--out_csv", default="lpr_outputs/predictions.csv")
    ap.add_argument("--exts", default="png,jpg,jpeg,bmp")
    ap.add_argument("--debug_dir", default="", help="If set, saves intermediate masks/chars here")
    args = ap.parse_args()

    debug_dir = args.debug_dir.strip() or None
    run_batch(args.plates_dir, args.out_csv, args.exts, debug_dir=debug_dir)


if __name__ == "__main__":
    main()