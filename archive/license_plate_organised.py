import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .classification_letters import classify_letter
from .preprocess_for_segmented import preprocess_letters as preprocess_segmented
from .classification import classify_with_blobs_from_A
from .morphology import find_blobs
from .features import get_stems, get_banded_points, draw_line
from .preprocessing import thin
from .features_letters import (
    concavity_tb_strength,
    concavity_lr_strength,
    count_horizontal_strokes,
    debug_endpoints,
    debug_horizontal_strokes,
    horizontal_symmetry_tb_balance,
    prune_spurs,
    count_endpoints,
    hole_count_and_largest_pct,
    vertical_symmetry_lr_balance,
    count_vertical_strokes,
    bottom_width_ratio,
    center_density_ratio,
    debug_vertical_strokes,
    side_open_score,
)





def pad_bbox(x, y, w, h, H, W, pad_x_frac=0.06, pad_y_frac=0.20):
    pad_x = int(pad_x_frac * w)
    pad_y = int(pad_y_frac * h)

    x2 = max(0, x - pad_x)
    y2 = max(0, y - pad_y)
    x3 = min(W, x + w + pad_x)
    y3 = min(H, y + h + pad_y)

    return x2, y2, (x3 - x2), (y3 - y2)


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

        # character-like height
        if h < h_lo * H or h > h_hi * H:
            continue

        ar = w / (h + 1e-6)
        if ar < ar_lo or ar > ar_hi:
            continue

        # kill border strips (your thick left/right columns)
        touches_left  = x <= border_margin
        touches_right = (x + w) >= (W - border_margin)
        if (touches_left or touches_right) and (h >= border_tall_frac * H):
            continue

        comps.append((i, x, y, w, h, area, cx, cy))

    if len(comps) == 0:
        return A  # nothing filtered

    # sort left-to-right
    comps.sort(key=lambda t: t[1])

    # estimate typical char width from survivors
    widths = np.array([c[3] for c in comps], dtype=np.float32)
    w_med = float(np.median(widths))

    # build longest run with reasonable gaps
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

#------for sclaing the license plate
def normalize_plate_size(plate_roi,
                         min_ok_w=600,
                         target_w=700,
                         max_ok_w=800):
    """
    Normalize plate ROI width around ~700 px.

    - If width < 600  → upscale to ~700
    - If width > 800  → downscale to ~700
    - If 600–800      → leave unchanged
    """
    h, w = plate_roi.shape[:2]

    if min_ok_w <= w <= max_ok_w:
        return plate_roi, 1.0

    scale = target_w / float(w)

    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    plate_resized = cv2.resize(
        plate_roi,
        None,
        fx=scale,
        fy=scale,
        interpolation=interp
    )

    return plate_resized, scale

# -----------------------------
# Character classification hook
# -----------------------------
def classify_segmented_char(char32: np.ndarray, debug=False) -> str:
    """
    Run a segmented 32x32 char through your exact
    preprocess -> thin -> classify_letter pipeline.
    """
    if char32.ndim == 3:
        char32 = cv2.cvtColor(char32, cv2.COLOR_BGR2GRAY)

    char32 = char32.astype(np.uint8)

    # IMPORTANT: ensure white foreground
    # Uncomment if predictions look inverted:
    # char32 = 255 - char32

    A, cleaned, bin_norm = preprocess_letters(char32, visualize=False)
    skel = thin(A)
    pred = classify_letter(A, skel)

    return pred

def bridge_gaps(bin255):
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    out = cv2.morphologyEx(bin255, cv2.MORPH_CLOSE, k_close, iterations=1)
    out = cv2.dilate(out, k_close, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=1)
    return out

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

def localise_plate(img, debug=False):
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        return None, {"blackhat": blackhat, "bw": bw, "morph": morph}

    x, y, w, h = best
    x, y, w, h = pad_bbox(x, y, w, h, H, W)

    plate_roi = img[y:y+h, x:x+w].copy()
    plate_roi, scale = normalize_plate_size(
        plate_roi,
        min_ok_w=400,
        target_w=600,
        max_ok_w=600
    )

    debug_data = {
        "blackhat": blackhat,
        "bw": bw,
        "morph": morph,
        "bbox": (x, y, w, h),
        "scale": scale
    }

    if debug:
        return plate_roi, debug_data
    return plate_roi, None

def preprocess_plate(plate_roi, debug=False):
    pgray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    pgray = cv2.GaussianBlur(pgray, (3, 3), 0)

    chars = cv2.adaptiveThreshold(
        pgray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    ys, xs = np.where(chars > 0)
    if len(xs) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        pad = 8
        x0 = max(0, x0 - pad)
        x1 = min(chars.shape[1] - 1, x1 + pad)
        y0 = max(0, y0 - pad)
        y1 = min(chars.shape[0] - 1, y1 + pad)

        plate_roi = plate_roi[y0:y1+1, x0:x1+1]
        chars = chars[y0:y1+1, x0:x1+1]

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    chars = cv2.morphologyEx(chars, cv2.MORPH_CLOSE, k, iterations=1)

    if debug:
        return plate_roi, chars, {"gray": pgray}
    return plate_roi, chars, None

def refine_plate_band(plate_roi, chars, debug=False):
    proj = np.sum(chars, axis=1).astype(np.float32)
    proj_s = cv2.GaussianBlur(proj.reshape(-1, 1), (1, 31), 0).flatten()
    proj_n = proj_s / (proj_s.max() + 1e-6)

    nonzero = proj_n[proj_n > 0]
    thr = max(0.10, 0.5 * np.median(nonzero)) if len(nonzero) > 0 else 0.10
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

    if len(runs) == 0:
        top, bottom = 0, chars.shape[0] - 1
    else:
        top, bottom = max(runs, key=lambda r: r[1] - r[0])

    pad = 3
    top = max(0, top - pad)
    bottom = min(chars.shape[0] - 1, bottom + pad)

    plate_refined = plate_roi[top:bottom+1, :]
    chars_refined = chars[top:bottom+1, :]

    if debug:
        return plate_refined, chars_refined, {"row_threshold": thr}
    return plate_refined, chars_refined, None

def keep_main_plate_chars(bin255,
                          min_area=120,
                          h_lo=0.55, h_hi=0.98,
                          ar_lo=0.12, ar_hi=1.10,
                          border_margin=1,
                          border_tall_frac=0.60,
                          max_gap_factor=0.60):
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

def clean_plate(chars_refined, debug=False):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(chars_refined, connectivity=8)

    chars_denoised = np.zeros_like(chars_refined)
    min_area = 80

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            chars_denoised[labels == i] = 255

    chars_clean = keep_main_plate_chars(chars_denoised)
    chars_clean = bridge_gaps(chars_clean)

    col_sum = np.sum(chars_clean > 0, axis=0).astype(np.float32)
    col_sum_s = cv2.GaussianBlur(col_sum.reshape(1, -1), (31, 1), 0).flatten()
    col_sum_n = col_sum_s / (np.max(col_sum_s) + 1e-6)

    cols = np.where(col_sum_n > 0.15)[0]
    if len(cols) > 0:
        left = max(0, cols[0] - 5)
        right = min(chars_clean.shape[1] - 1, cols[-1] + 5)
        chars_clean = chars_clean[:, left:right+1]

    if debug:
        return chars_clean, {"chars_denoised": chars_denoised}
    return chars_clean, None

def segment_characters(word_mask, max_chars=7, debug=False):
    h, w = word_mask.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(word_mask, connectivity=8)

    chars = []
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

        chars.append((x, y, cw, ch, area))

    if len(chars) > max_chars:
        chars = sorted(chars, key=lambda c: c[4])
        chars = chars[len(chars) - max_chars:]

    chars = sorted(chars, key=lambda c: c[0])

    char_images = []
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    for x, y, cw, ch, _ in chars:
        char_img = word_mask[y:y+ch, x:x+cw].copy()
        char_img = fill_small_holes(char_img, max_hole_frac=0.05)
        char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, k1, iterations=1)
        char_img = cv2.dilate(char_img, k1, iterations=1)
        char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, k2, iterations=1)
        char_images.append(char_img)

    if debug:
        return char_images, chars
    return char_images, None


def normalize_characters(char_images, target_size=96):
    normalized_chars = []

    for char_img in char_images:
        ys, xs = np.where(char_img > 0)
        if xs.size == 0:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        char_img = char_img[y0:y1+1, x0:x1+1]

        h, w = char_img.shape
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)

        y_off = (size - h) // 2
        x_off = (size - w) // 2
        square[y_off:y_off+h, x_off:x_off+w] = char_img

        resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        normalized_chars.append(resized)

    return normalized_chars

def char_type_from_index(i: int) -> str:
    if i == 0:
        return "digit"
    if 1 <= i <= 3:
        return "letter"
    return "digit"

def classify_plate_chars(normalized_chars, debug=False):
    plate_text = ""

    for idx, char_img in enumerate(normalized_chars):
        ctype = char_type_from_index(idx)

        A, cleaned, bin_norm = preprocess_segmented(
            char_img,
            visualize=debug,
            plate_mode=True
        )

        skel = thin(A)
        sk_pruned = prune_spurs(skel, max_length=2)

        if ctype == "digit":
            digit, group = classify_with_blobs_from_A(A, debug=debug)
            pred = str(digit) if digit is not None else "?"
        else:
            pred = classify_letter(A, skel)
            pred = pred if pred is not None else "?"

        plate_text += pred

    return plate_text

def run_plate_pipeline(img_path, debug=False):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {img_path}")

    plate_roi, loc_debug = localise_plate(img, debug=debug)
    if plate_roi is None:
        raise RuntimeError("No plate found")

    plate_roi, chars, prep_debug = preprocess_plate(plate_roi, debug=debug)
    plate_refined, chars_refined, band_debug = refine_plate_band(plate_roi, chars, debug=debug)
    chars_clean, clean_debug = clean_plate(chars_refined, debug=debug)
    char_images, boxes = segment_characters(chars_clean, max_chars=7, debug=debug)
    normalized_chars = normalize_characters(char_images, target_size=96)

    plate_text = classify_plate_chars(normalized_chars, debug=debug)

    return {
        "plate_text": plate_text,
        "plate_roi": plate_roi,
        "plate_refined": plate_refined,
        "chars_clean": chars_clean,
        "char_images": char_images,
        "normalized_chars": normalized_chars,
    }

if __name__ == "__main__":
    IMG_PATH = "images/plates/plate33.png"

    result = run_plate_pipeline(IMG_PATH, debug=True)
    print("FINAL PLATE:", result["plate_text"])