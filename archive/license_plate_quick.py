import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from .classification_letters import classify_letter
from .preprocess_for_segmented import preprocess_letters as preprocess_segmented
from .preprocessing import thin
from .features_letters import concavity_tb_strength, count_horizontal_strokes, debug_endpoints, debug_horizontal_strokes, horizontal_symmetry_tb_balance, prune_spurs, count_endpoints, hole_count_and_largest_pct, vertical_symmetry_lr_balance, count_vertical_strokes, bottom_width_ratio, center_density_ratio, debug_vertical_strokes, debug_bottom_width_ratio, side_open_score

def pad_bbox(x, y, w, h, H, W, pad_x_frac=0.06, pad_y_frac=0.20):
    pad_x = int(pad_x_frac * w)
    pad_y = int(pad_y_frac * h)

    x2 = max(0, x - pad_x)
    y2 = max(0, y - pad_y)
    x3 = min(W, x + w + pad_x)
    y3 = min(H, y + h + pad_y)

    return x2, y2, (x3 - x2), (y3 - y2)

def char_type_from_index(i: int) -> str:
    if i == 0:
        return "digit"
    if 1 <= i <= 3:
        return "letter"
    return "digit"   # i >= 4

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
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
        out = cv2.morphologyEx(bin255, cv2.MORPH_CLOSE, k_close, iterations=1)
        out = cv2.dilate(out, k_close, iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=1)
        return out

PLATE_DIR = "images/plates"
OUT_DIR = "lpr_outputs/debug"
OUT_PLATES_DIR = "lpr_outputs/plates"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_PLATES_DIR, exist_ok=True)

plate_files = sorted([
    f for f in os.listdir(PLATE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

results = {}

for plate_file in plate_files:
    IMG_PATH = os.path.join(PLATE_DIR, plate_file)
    stem = os.path.splitext(plate_file)[0]

    print("\n===================================")
    print("Processing:", IMG_PATH)
    print("===================================")

    img = cv2.imread(IMG_PATH)
    if img is None:
        print("Failed to load:", IMG_PATH)
        continue

    H, W = img.shape[:2]

    # --- Build a plate-like blob from intensity structure ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    k_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_bh)

    _, bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=2)

    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    bw = cv2.dilate(bw, k_dil, iterations=1)

    cv2.imwrite(f"{OUT_DIR}/{stem}_03b_bw.png", bw)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imwrite(f"{OUT_DIR}/{stem}_04_morph.png", morph)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1

    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        ar = ww / float(hh + 1e-6)
        area = ww * hh

        if ar < 1.6 or ar > 8.0:
            continue
        if area < 0.01 * W * H or area > 0.80 * W * H:
            continue
        if y < 0.20 * H or y > 0.80 * H:
            continue

        cx = x + ww / 2
        cy = y + hh / 2
        center_dist = abs(cx - W / 2) / W + abs(cy - H / 2) / H
        score = (ww * hh) / (1.0 + 5.0 * center_dist)

        if score > best_score:
            best_score = score
            best = (x, y, ww, hh)

    if best is None:
        print("No plate candidate found -> skipping", plate_file)
        results[plate_file] = "NO_PLATE"
        continue

    x, y, ww, hh = best
    x, y, ww, hh = pad_bbox(x, y, ww, hh, H, W)

    plate_roi = img[y:y+hh, x:x+ww].copy()

    plate_roi, scale = normalize_plate_size(
        plate_roi,
        min_ok_w=400,
        target_w=600,
        max_ok_w=600
    )

    cv2.imwrite(f"{OUT_PLATES_DIR}/{stem}_roi.png", plate_roi)

    # -------------------------
    # SEGMENTATION + OCR (yours)
    # -------------------------

    pgray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    pgray = cv2.GaussianBlur(pgray, (3, 3), 0)

    chars = cv2.adaptiveThreshold(
        pgray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    ys, xs = np.where(chars > 0)
    if len(xs) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        pad = 8
        x0 = max(0, x0 - pad); x1 = min(chars.shape[1] - 1, x1 + pad)
        y0 = max(0, y0 - pad); y1 = min(chars.shape[0] - 1, y1 + pad)

        plate_roi = plate_roi[y0:y1+1, x0:x1+1]
        chars = chars[y0:y1+1, x0:x1+1]

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    chars = cv2.morphologyEx(chars, cv2.MORPH_CLOSE, k, iterations=1)

    cv2.imwrite(f"{OUT_DIR}/{stem}_chars.png", chars)

    # --- Robustly find the main character band (big letters) ---

    

    proj = np.sum(chars, axis=1).astype(np.float32)
    proj_s = cv2.GaussianBlur(proj.reshape(-1,1), (1,31), 0).flatten()

    # normalize
    proj_n = proj_s / (proj_s.max() + 1e-6)

    # adaptive threshold based on the profile (much safer)
    thr = max(0.10, 0.5 * np.median(proj_n[proj_n > 0]))   # usually ~0.10–0.20
    mask_rows = (proj_n > thr).astype(np.uint8)

    # find contiguous runs of 1s and pick the longest run
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
        # choose the longest run (main text band)
        top, bottom = max(runs, key=lambda r: r[1] - r[0])

    # pad a bit (so we don't cut off tops/bottoms of letters)
    pad = 3
    top = max(0, top - pad)
    bottom = min(chars.shape[0] - 1, bottom + pad)

    plate_refined = plate_roi[top:bottom+1, :]
    chars_refined = chars[top:bottom+1, :]

    #########horizonal

    # --- Robust left/right crop but DON'T cut off first/last character ---

    # --- Left/right crop using connected components (robust) ---
    hR, wR = chars_refined.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(chars_refined, connectivity=8)

    # find tallest component height (likely a real letter)
    max_h = 0
    for i in range(1, num):
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if ch > max_h:
            max_h = ch

    min_h = int(0.70 * max_h)   # keep big letters; if it drops letters, use 0.65

    xs = []
    xe = []

    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # filter noise
        if area < 50:
            continue
        if ch < min_h:
            continue

        xs.append(x)
        xe.append(x + cw)

    if len(xs) == 0:
        # fallback: don't crop left/right
        left, right = 0, wR - 1
    else:
        left = min(xs)
        right = max(xe)

    # padding so we never cut letters
    pad_lr = int(0.03 * wR)  # 3% width
    left = max(0, left - pad_lr)
    right = min(wR - 1, right + pad_lr)

    plate_final = plate_refined[:, left:right]
    chars_final = chars_refined[:, left:right]

    # --- Left/right crop using CC on chars_refined ---
    hR, wR = chars_refined.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(chars_refined, connectivity=8)

    # tallest component height (likely a real character)
    max_h = max(stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num)) if num > 1 else 0
    min_h = int(0.65 * max_h)   # 0.65 keeps thinner letters like L

    xs, xe = [], []
    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 50:
            continue
        if ch < min_h:
            continue

        xs.append(x)
        xe.append(x + cw)

    if len(xs) == 0:
        left, right = 0, wR - 1
    else:
        left, right = min(xs), max(xe)

    pad_lr = int(0.04 * wR)  # 4% width padding
    left = max(0, left - pad_lr)
    right = min(wR - 1, right + pad_lr)

    chars_final = chars_refined[:, left:right]
    plate_final = plate_refined[:, left:right]

    # ---- 0) Start from your binary (0/255) characters mask
    bin255 = chars_final.copy()

    # ---- 1) Remove tiny speckles (optional but usually helps)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8)
    min_area = 80
    den = np.zeros_like(bin255)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            den[labels == i] = 255

    # ---- 2) Create ONE BIG WORD BLOB (horizontal close)
    # Wide kernel connects characters across gaps but doesn't prefer vertical merging much.
    k_word = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))   # try (25-55, 3-7)
    blob = cv2.morphologyEx(den, cv2.MORPH_CLOSE, k_word, iterations=2)

    # Optional: small dilation makes it more solid
    blob = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_RECT, (5,3)), iterations=1)

    # ---- 3) Pick best blob rectangle (like localisation) and crop
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = den.shape
    best = None
    best_score = -1

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ar = w / (h + 1e-6)

        # filters: wide and reasonably tall (word band)
        if ar < 2.0:
            continue
        if h < 0.45 * H:
            continue
        if area < 0.15 * W * H:
            continue

        # score using ink density on the ORIGINAL den mask (not blob)
        roi = den[y:y+h, x:x+w]
        ink = np.sum(roi > 0)
        density = ink / (area + 1e-6)

        # penalty if it’s basically the frame touching both sides
        touches_left  = x <= int(0.02 * W)
        touches_right = (x + w) >= int(0.98 * W)
        if touches_left and touches_right:
            continue

        score = density * area
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        word_crop = den
    else:
        x, y, w, h = best
        pad = 6
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        word_crop = den[y0:y1, x0:x1]

    # 1) DENOISE first: keep only CCs above area threshold
    h, w = chars_final.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(chars_final, connectivity=8)

    min_area = 80
    chars_denoised = np.zeros_like(chars_final)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            chars_denoised[labels == i] = 255

    bin255 = chars_denoised  # <-- THIS is the mask you want to blob

    # 2) Create "word blob" by connecting letters horizontally
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    blob = cv2.morphologyEx(bin255, cv2.MORPH_CLOSE, k, iterations=1)
    blob = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_RECT, (7,3)), iterations=1)

    # 3) Find best rectangle candidate on the blob
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = bin255.shape
    best = None
    best_score = -1

    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        ar = ww / (hh + 1e-6)

        # word-like geometry filters
        if ar < 2.0:
            continue
        if area < 0.05 * W * H:
            continue
        if hh < 0.40 * H:
            continue

        # density measured on ORIGINAL letters mask (bin255)
        roi = bin255[y:y+hh, x:x+ww]
        ink = np.sum(roi > 0)
        density = ink / (area + 1e-6)

        # avoid plate frame boxes spanning both sides
        touches_left  = x <= int(0.02 * W)
        touches_right = (x + ww) >= int(0.98 * W)
        if touches_left and touches_right:
            continue

        score = density * area
        if score > best_score:
            best_score = score
            best = (x, y, ww, hh)

    # 4) Build word_mask OUTSIDE the loop
    if best is None:
        word_mask = bin255.copy()
    else:
        x, y, ww, hh = best
        pad = 4
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + ww + pad); y1 = min(H, y + hh + pad)

        word_mask = np.zeros_like(bin255)
        word_mask[y0:y1, x0:x1] = bin255[y0:y1, x0:x1]

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            chars_denoised[labels == i] = 255

    chars_denoised = bridge_gaps(chars_denoised)

    chars_clean = keep_main_plate_chars(chars_denoised)  # or chars_final / chars_denoised

    img_bin = chars_clean.copy()
    h, w = img_bin.shape

    col_sum = np.sum(img_bin > 0, axis=0).astype(np.float32)

    # Smooth heavily to remove micro-gaps
    col_sum_s = cv2.GaussianBlur(col_sum.reshape(1, -1), (31,1), 0).flatten()

    # Normalize
    col_sum_n = col_sum_s / (np.max(col_sum_s) + 1e-6)

    thr = 0.15  # lower than before; we want sensitivity
    cols = np.where(col_sum_n > thr)[0]

    left = cols[0]
    right = cols[-1]

    pad = 5
    left = max(0, left - pad)
    right = min(w-1, right + pad)

    clean_word = img_bin[:, left:right+1]

    img_bin = clean_word.copy()
    h, w = img_bin.shape

    num, labels, stats, _ = cv2.connectedComponentsWithStats(img_bin, connectivity=8)

    out = np.zeros_like(img_bin)

    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        ar = cw / float(ch + 1e-6)

        # Character-like filters (tune lightly)
        if ch < 0.4 * h:          # must be tall enough
            continue
        if cw < 0.02 * w:         # must not be super thin
            continue
        if ar < 0.10 or ar > 1.20: # characters aren't ultra skinny or ultra wide
            continue
        if area < 80:
            continue

        out[labels == i] = 255

    # out = chars_only_rect_filter result (0/255)
    mask = (out > 0).astype(np.uint8) * 255

    # 1) Tight crop to ink bounding box (removes lots of frame automatically)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Mask is empty after chars_only_rect_filter")

    pad = 3
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); x1 = min(mask.shape[1]-1, x1 + pad)
    y0 = max(0, y0 - pad); y1 = min(mask.shape[0]-1, y1 + pad)

    mask_tight = mask[y0:y1+1, x0:x1+1]

    # 2) "Rectangle again" but DONE RIGHT: merge letters slightly then pick best contour rectangle
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    merged = cv2.dilate(mask_tight, k, iterations=1)

    # After merging step
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    H, W = mask_tight.shape

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ar = w / float(h + 1e-6)

        if area < 0.03 * W * H:
            continue

        boxes.append((x, y, w, h))

    if len(boxes) == 0:
        word_mask = mask_tight
    else:
        # sort left to right
        boxes = sorted(boxes, key=lambda b: b[0])

        # merge all boxes into one big horizontal region
        xs = [b[0] for b in boxes]
        xe = [b[0] + b[2] for b in boxes]
        ys = [b[1] for b in boxes]
        ye = [b[1] + b[3] for b in boxes]

        x0 = min(xs)
        x1 = max(xe)
        y0 = min(ys)
        y1 = max(ye)

        pad = 3
        x0 = max(0, x0 - pad)
        x1 = min(W - 1, x1 + pad)
        y0 = max(0, y0 - pad)
        y1 = min(H - 1, y1 + pad)

        word_mask = np.zeros_like(mask_tight)
        word_mask[y0:y1, x0:x1] = mask_tight[y0:y1, x0:x1]

    import cv2
    import numpy as np

    def fill_small_holes(bin255, max_hole_frac=0.05, max_hole_pixels=None):
        """
        Fill small enclosed background holes in a single-character binary image.
        - max_hole_frac is fraction of bounding-box area (NOT whole image area).
        """
        A = (bin255 > 0).astype(np.uint8)  # 0/1
        ys, xs = np.where(A == 1)
        if xs.size == 0:
            return bin255

        # bbox of ink
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        roi = A[y0:y1+1, x0:x1+1]
        H, W = roi.shape
        box_area = H * W

        # background inside ROI
        bg = (roi == 0).astype(np.uint8) * 255

        # label background CCs
        num, lab, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)

        # any bg component touching border is NOT a hole
        hole_ids = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            touches_border = (x == 0) or (y == 0) or (x + w == W) or (y + h == H)
            if not touches_border:
                hole_ids.append((i, area))

        if not hole_ids:
            return bin255

        # threshold
        if max_hole_pixels is None:
            max_hole_pixels = int(max_hole_frac * box_area)

        # fill small holes
        roi_filled = roi.copy()
        for i, area in hole_ids:
            if area <= max_hole_pixels:
                roi_filled[lab == i] = 1

        out = A.copy()
        out[y0:y1+1, x0:x1+1] = roi_filled
        return (out * 255).astype(np.uint8)


    # 3) Character segmentation from word_mask
    img_bin = word_mask
    h, w = img_bin.shape

    num, labels, stats, _ = cv2.connectedComponentsWithStats(img_bin, connectivity=8)

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

        # store area so we can remove the smallest properly
        chars.append((x, y, cw, ch, area))

    # ---- cap to max 7 by removing smallest area ----
    MAX_CHARS = 7
    if len(chars) > MAX_CHARS:
        # sort by area ascending, remove smallest until only 7 remain
        chars = sorted(chars, key=lambda c: c[4])   # smallest first
        chars = chars[len(chars) - MAX_CHARS:]      # keep largest 7

    # ---- reorder left->right ----
    chars = sorted(chars, key=lambda c: c[0])

    chars = [(x, y, cw, ch) for (x, y, cw, ch, area) in chars]

    # ---------------------------------------
    # PROCESS CHARACTERS PROPERLY
    # ---------------------------------------

    processed_chars = []

    for i, (x, y, cw, ch) in enumerate(chars):
        char_img = img_bin[y:y+ch, x:x+cw].copy()

        # hole fixing + light cleanup
        char_img = fill_small_holes(char_img, max_hole_frac=0.05)
        char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, k1, iterations=1)
        char_img = cv2.dilate(char_img, k1, iterations=1)
        char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, k2, iterations=1)

        processed_chars.append(char_img)

    # ---------------------------------------
    # NORMALIZE PROCESSED CHARS
    # ---------------------------------------

    normalized_chars = []
    target_size = 96

    for i, char_img in enumerate(processed_chars):

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

        resized = cv2.resize(square, (target_size, target_size),
                            interpolation=cv2.INTER_NEAREST)

        normalized_chars.append(resized)

    #-------------COMMENT---------------
    from .morphology import find_blobs
    from .features import get_stems, get_banded_points, draw_line

    def debug_digit_steps(A, title="digit-debug"):
        """
        A is 0/1 binary (foreground=1). Shows how digit pipeline manipulates it:
        thin -> blobs -> stems, and if Group2: helper lines + blobs after each line.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # Step 1: thin
        A_thin = thin(A)

        # Step 2: blobs on thinned (NO FIND BLOBS ON THICK IMAGE, it’s too merged; we want to see the effect of thinning + pruning on blobs)
        blobs0, n_blobs0 = find_blobs(A)

        # Base view (always)
        #plt.figure(figsize=(10, 3))
        #plt.suptitle(f"{title} | start blobs={n_blobs0}")
        #plt.subplot(1,3,1); #plt.imshow(A, cmap="gray");      #plt.title("A (binary)"); #plt.axis("off")
        #plt.subplot(1,3,2); #plt.imshow(A_thin, cmap="gray"); #plt.title("A_thin");     #plt.axis("off")
        #plt.subplot(1,3,3); #plt.imshow(blobs0, cmap="gray"); #plt.title(f"blobs0 (n={n_blobs0})"); #plt.axis("off")
        #plt.tight_layout()

        # ---- Group 1 debug: stems ----
        if n_blobs0 > 0:
            stems_img, n_stems, _ = get_stems(A, blobs0)

            #plt.figure(figsize=(8, 3))
            #plt.suptitle(f"{title} | Group1 stems={n_stems}")
            #plt.subplot(1,2,1); #plt.imshow(A_thin, cmap="gray");    #plt.title("A_thin"); #plt.axis("off")
            #plt.subplot(1,2,2); #plt.imshow(stems_img, cmap="gray"); #plt.title("stems_img"); #plt.axis("off")
            #plt.tight_layout()
            return  # done

        # ---- Group 2 debug: helper lines ----
        pts = get_banded_points(A_thin, split=0.5)
        if pts is None:
            return

        TL, BL, TR, BR = pts

        # A1: TL->BL
        A1 = draw_line(A_thin, TL, BL)
        blobs1, nb1 = find_blobs(A1, min_blob_area=1)

        if nb1 == 0:
            # A2: TL->BR (1 vs 4)
            A2 = draw_line(A_thin, TL, BR)
            blobs2, nb2 = find_blobs(A2, min_blob_area=1)
            return

        # stems on A1 (2 vs 5)
        stems_img, n_stems, _ = get_stems(A1, blobs1)

        if n_stems == 0:
            # A3: TR->BR (3 vs 7)
            A3 = draw_line(A_thin, TR, BR)
            blobs3, nb3 = find_blobs(A3, min_blob_area=1)
            return

    from .classification import classify_with_blobs_from_A

    def char_type_from_index(i: int) -> str:
        if i == 0:
            return "digit"
        if 1 <= i <= 3:
            return "letter"
        return "digit"

    plate_text = ""

    for idx, char_img in enumerate(normalized_chars):

        ctype = char_type_from_index(idx)

        # 1) ONE preprocess for both digit/letter
        A, cleaned, bin_norm = preprocess_segmented(char_img, visualize=True, plate_mode=True)

        # skeleton always computed (you want debug visuals anyway)
        skel = thin(A)
        sk_pruned = prune_spurs(skel, max_length=2)
        e = count_endpoints(sk_pruned)

        if ctype == "digit":
            # DIGIT CLASSIFICATION
            digit, group = classify_with_blobs_from_A(A, debug=True, visualize=False)
            #debug_digit_steps(A, title=f"plate idx={idx} pred={digit} group={group}")
            pred = str(digit) if digit is not None else "?"
            plate_text += pred

        else:
            # LETTER CLASSIFICATION
            pred = classify_letter(A, skel)
            pred = pred if pred is not None else "?"
            plate_text += pred

            hc, hpct = hole_count_and_largest_pct(A)
            vs = vertical_symmetry_lr_balance(A)
            hs = horizontal_symmetry_tb_balance(A)

            vstrokes = count_vertical_strokes(A, min_frac=0.7)
            hstrokes = count_horizontal_strokes(
                A, min_frac=0.8, gap_allow=2, band=5, support_rows=3,
                orient_thresh=0.12, rel_width_keep=0.80
            )

            north, south = concavity_tb_strength(A)
            bw = bottom_width_ratio(A, band_frac=0.20)
            ratio = center_density_ratio(A, frac=0.35)
            left_ratio, right_ratio = side_open_score(A)

            # --- Your debug visuals ---
            # debug_endpoints(sk_pruned)
            # debug_vertical_strokes(A, min_frac=0.7, gap_allow=2, support_cols=3, orient_thresh=0.12)
            # debug_horizontal_strokes(A, min_frac=0.8, gap_allow=2, band=5, support_rows=3,
            #                         orient_thresh=0.12, rel_width_keep=0.80)

        # always compute skeletons (both digit & letter)
        _ = (skel * 255).astype(np.uint8)
        _ = (sk_pruned * 255).astype(np.uint8)

        results[plate_file] = plate_text
        print(f"Predicted ({plate_file}): {plate_text}")



print("\nFINAL RESULTS")
for k,v in results.items():
    print(k, "->", v)

        #plt.close("all")

# final text available in plate_text

