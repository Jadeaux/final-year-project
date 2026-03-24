import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from .classification_kumar import classify_letter
from .preprocess_for_segmented import preprocess_letters as preprocess_segmented
from .preprocessing import thin
from .features_letters import (
    concavity_tb_strength, concavity_lr_strength,
    count_horizontal_strokes, debug_endpoints, debug_horizontal_strokes,
    horizontal_symmetry_tb_balance, prune_spurs, count_endpoints,
    hole_count_and_largest_pct, vertical_symmetry_lr_balance,
    count_vertical_strokes, bottom_width_ratio, center_density_ratio,
    debug_vertical_strokes, debug_bottom_width_ratio, side_open_score
)

IMG_PATH = "images/plates/image2.png"
OUT_DIR = "lpr_outputs/debug"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("lpr_outputs/plates", exist_ok=True)


def keep_main_plate_chars(bin255,
                          min_area=40,
                          h_lo=0.30, h_hi=0.98,
                          ar_lo=0.05, ar_hi=1.50,
                          border_margin=1,
                          border_tall_frac=0.75,
                          max_gap_factor=3.0,
                          debug=False):
    """
    Keep the main left-to-right run of character-like connected components.
    Input:  0/255 binary image, white = foreground
    Output: 0/255 binary image
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

        ar = w / float(h + 1e-6)

        if debug:
            print(f"[keep_main_plate_chars] CC#{i}: x={x} y={y} w={w} h={h} area={area} ar={ar:.2f}")

        if area < min_area:
            continue
        if h < h_lo * H or h > h_hi * H:
            continue
        if ar < ar_lo or ar > ar_hi:
            continue

        touches_left = x <= border_margin
        touches_right = (x + w) >= (W - border_margin)

        # reject very tall border junk
        if (touches_left or touches_right) and (h >= border_tall_frac * H):
            continue

        comps.append((i, x, y, w, h, area, cx, cy))

    if debug:
        print("[keep_main_plate_chars] survivors:", len(comps))

    # fallback: if nothing survives, return input instead of crashing
    if len(comps) == 0:
        return A.copy()

    comps.sort(key=lambda t: t[1])

    widths = np.array([c[3] for c in comps], dtype=np.float32)
    w_med = float(np.median(widths)) if len(widths) > 0 else 1.0
    if w_med < 1:
        w_med = 1.0

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

    keep_ids = {c[0] for c in best_run}

    out = np.zeros_like(A)
    for i in keep_ids:
        out[labels == i] = 255

    return out

def pad_bbox(x, y, w, h, H, W, pad_x_frac=0.06, pad_y_frac=0.20):
    pad_x = int(pad_x_frac * w)
    pad_y = int(pad_y_frac * h)

    x2 = max(0, x - pad_x)
    y2 = max(0, y - pad_y)
    x3 = min(W, x + w + pad_x)
    y3 = min(H, y + h + pad_y)

    return x2, y2, (x3 - x2), (y3 - y2)


def normalize_plate_size(plate_roi,
                         min_ok_w=400,
                         target_w=600,
                         max_ok_w=600):
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


def get_vehicle_search_roi(img):
    """
    Restrict search to the central road region where the lead vehicle usually is.
    Returns:
        search_roi
        (x_off, y_off)
    """
    H, W = img.shape[:2]

    x0 = int(0.30 * W)
    x1 = int(0.70 * W)
    y0 = int(0.28 * H)
    y1 = int(0.72 * H)

    return img[y0:y1, x0:x1].copy(), (x0, y0)


def localise_plate_small_far(img, debug=False):
    """
    Plate localisation for distant vehicles.
    Searches only in a central vehicle ROI and uses smaller morphology kernels.
    Returns:
        plate_roi, bbox_full, debug_dict
        bbox_full = (x, y, w, h) in full image coords
    """
    H, W = img.shape[:2]

    search_roi, (x_off, y_off) = get_vehicle_search_roi(img)
    hs, ws = search_roi.shape[:2]

    gray = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # blackhat tuned for smaller/farther plate text
    k_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_bh)

    # Otsu threshold inside search ROI
    _, bw = cv2.threshold(
        blackhat, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Keep lower-middle region only, but not too aggressively
    mask_lower = np.zeros_like(bw)
    mask_lower[int(0.38 * hs):, :] = 255
    bw = cv2.bitwise_and(bw, mask_lower)

    # connect plate fragments more strongly
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    bw_close = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=1)

    # light dilation to make candidate more solid
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    bw_blob = cv2.dilate(bw_close, k_dil, iterations=1)

    # Remove ROI borders
    margin = 4
    bw_blob[:margin, :] = 0
    bw_blob[-margin:, :] = 0
    bw_blob[:, :margin] = 0
    bw_blob[:, -margin:] = 0

    contours, _ = cv2.findContours(bw_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("Total contours found:", len(contours))

    # visualize all candidate boxes in search ROI
    vis_cands = search_roi.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(vis_cands, (x, y), (x + w, y + h), (0, 255, 0), 1)

    best = None
    best_score = -1.0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h + 1e-6)
        area = w * h

        print(f"\nCandidate: x={x} y={y} w={w} h={h} ar={ar:.2f} area={area}")

        # relaxed geometry for distant / broken plate blobs
        if ar < 1.4 or ar > 7.0:
            print("  rejected: aspect ratio")
            continue

        if area < 0.00025 * (W * H):
            print("  rejected: area too small")
            continue

        if area > 0.030 * (W * H):
            print("  rejected: area too large")
            continue

        # avoid things glued to ROI borders
        if x <= 2 or y <= 2 or (x + w) >= (ws - 2) or (y + h) >= (hs - 2):
            print("  rejected: touches ROI border")
            continue

        # prefer lower half of search ROI
        cx = x + w / 2.0
        cy = y + h / 2.0
        cx_n = cx / ws
        cy_n = cy / hs

        if cy_n < 0.45:
            print("  rejected: too high in ROI")
            continue

        # score based on blackhat strength inside candidate
        bh_roi = blackhat[y:y+h, x:x+w]
        ink = float(np.sum(bh_roi)) / 255.0
        density = ink / (area + 1e-6)

        center_pen = abs(cx_n - 0.50)
        y_pen = abs(cy_n - 0.58)
        ar_pen = abs(ar - 3.5)

        score = (
            2.5 * density +
            0.8 * np.log(area + 1.0) -
            2.2 * center_pen -
            1.8 * y_pen -
            0.25 * ar_pen
        )

        print(f"  PASSED | density={density:.4f} score={score:.4f}")

        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    debug_dict = {
        "search_roi": search_roi,
        "gray": gray,
        "blackhat": blackhat,
        "bw": bw,
        "bw_close": bw_close,
        "bw_blob": bw_blob,
        "all_candidate_boxes": vis_cands
    }

    if best is None:
        return None, None, debug_dict

    x, y, w, h = best
    x_full = x + x_off
    y_full = y + y_off

    x_full, y_full, w_full, h_full = pad_bbox(
        x_full, y_full, w, h, H, W,
        pad_x_frac=0.08,
        pad_y_frac=0.18
    )

    plate_roi = img[y_full:y_full+h_full, x_full:x_full+w_full].copy()
    plate_roi, scale = normalize_plate_size(
        plate_roi,
        min_ok_w=400,
        target_w=600,
        max_ok_w=600
    )

    debug_dict["bbox_full"] = (x_full, y_full, w_full, h_full)
    debug_dict["scale"] = scale
    debug_dict["best_score"] = best_score

    return plate_roi, (x_full, y_full, w_full, h_full), debug_dict


# -------------------------------------------------
# RUN
# -------------------------------------------------
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load {IMG_PATH}")

H, W = img.shape[:2]
print("Loaded:", IMG_PATH, "shape:", img.shape)

plate_roi, plate_bbox, dbg = localise_plate_small_far(img, debug=True)

if plate_roi is None:
    print("No plate chosen")

    cv2.imshow("search_roi", dbg["search_roi"])
    cv2.imshow("blackhat", dbg["blackhat"])
    cv2.imshow("bw", dbg["bw"])
    cv2.imshow("bw_close", dbg["bw_close"])
    cv2.imshow("bw_blob", dbg["bw_blob"])
    cv2.imshow("all_candidate_boxes", dbg["all_candidate_boxes"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    x, y, w, h = plate_bbox
    print("CHOSEN PLATE:", plate_bbox)
    print("Best score:", dbg["best_score"])

    vis2 = img.copy()
    cv2.rectangle(vis2, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow("search_roi", dbg["search_roi"])
    cv2.imshow("blackhat", dbg["blackhat"])
    cv2.imshow("bw", dbg["bw"])
    cv2.imshow("bw_close", dbg["bw_close"])
    cv2.imshow("bw_blob", dbg["bw_blob"])
    cv2.imshow("all_candidate_boxes", dbg["all_candidate_boxes"])
    cv2.imshow("chosen_plate_fullimg", vis2)
    cv2.imshow("plate_roi", plate_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("lpr_outputs/plates/plate1_roi.png", plate_roi)

#-------------------------UP UNTIL HERE LOCALISATION WORKS---------------------------

pgray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
pgray = cv2.GaussianBlur(pgray, (3,3), 0)

chars = cv2.adaptiveThreshold(
    pgray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31, 10
)

# chars is 0/255, white = foreground (letters)
k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

# 1) close to bridge small cracks in strokes

# --- Tighten plate ROI using the chars mask (remove purple surround/frame) ---
ys, xs = np.where(chars > 0)

if len(xs) > 0:
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    pad = 8  # small safety margin
    x0 = max(0, x0 - pad); x1 = min(chars.shape[1]-1, x1 + pad)
    y0 = max(0, y0 - pad); y1 = min(chars.shape[0]-1, y1 + pad)

    # crop BOTH the ROI and the mask consistently
    plate_roi = plate_roi[y0:y1+1, x0:x1+1]
    chars     = chars[y0:y1+1, x0:x1+1]

cv2.imshow("plate_roi_tight", plate_roi)
cv2.imshow("chars_mask_tight", chars)
cv2.waitKey(0)
cv2.destroyAllWindows()

k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
chars = cv2.morphologyEx(chars, cv2.MORPH_CLOSE, k, iterations=1)


cv2.imwrite("lpr_outputs/debug/plate1_chars.png", chars)

cv2.imshow("08_chars_mask", chars)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Robustly find the main character band (big letters) ---

def bridge_gaps(bin255):
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    out = cv2.morphologyEx(bin255, cv2.MORPH_CLOSE, k_close, iterations=1)
    out = cv2.dilate(out, k_close, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=1)
    return out

proj = np.sum(chars, axis=1).astype(np.float32)
proj_s = cv2.GaussianBlur(proj.reshape(-1,1), (1,31), 0).flatten()

# normalize
proj_n = proj_s / (proj_s.max() + 1e-6)

# adaptive threshold based on the profile (much safer)
thr = max(0.10, 0.5 * np.median(proj_n[proj_n > 0]))   # usually ~0.10–0.20
mask_rows = (proj_n > thr).astype(np.uint8)
print("row thr:", thr)

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
    print("No strong character band found; keeping original.")
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

cv2.imshow("refined_roi", plate_refined)
cv2.imshow("refined_chars", chars_refined)
cv2.waitKey(0)
cv2.destroyAllWindows()


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

print("CC left/right:", left, right, "min_h:", min_h, "max_h:", max_h)

cv2.imshow("final_zoom_roi", plate_final)
cv2.imshow("final_zoom_chars", chars_final)
cv2.waitKey(0)
cv2.destroyAllWindows()

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


cv2.imshow("den", den); cv2.waitKey(0)

# ---- 2) Create ONE BIG WORD BLOB (horizontal close)
# Wide kernel connects characters across gaps but doesn't prefer vertical merging much.
k_word = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))   # try (25-55, 3-7)
blob = cv2.morphologyEx(den, cv2.MORPH_CLOSE, k_word, iterations=2)

# Optional: small dilation makes it more solid
blob = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_RECT, (5,3)), iterations=1)

cv2.imshow("blob", blob); cv2.waitKey(0)

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
    print("No good blob found -> using den as-is")
    word_crop = den
else:
    x, y, w, h = best
    pad = 6
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    word_crop = den[y0:y1, x0:x1]

cv2.imshow("word_crop", word_crop); cv2.waitKey(0)


print("CC left/right:", left, right, "min_h:", min_h, "max_h:", max_h)

cv2.imshow("final_zoom_chars", chars_final)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

cv2.imshow("chars_denoised", bin255); cv2.waitKey(0)

# 2) Create "word blob" by connecting letters horizontally
k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
blob = cv2.morphologyEx(bin255, cv2.MORPH_CLOSE, k, iterations=1)
blob = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_RECT, (7,3)), iterations=1)

cv2.imshow("word_blob_candidates", blob); cv2.waitKey(0)

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

cv2.imshow("word_mask_kept", word_mask); cv2.waitKey(0)

for i in range(1, num):
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= min_area:
        chars_denoised[labels == i] = 255


cv2.imshow("chars_denoised", chars_denoised)
cv2.waitKey(0)
cv2.destroyAllWindows()
#------removed the lines by calculating the distances between neighbors

chars_denoised = cv2.morphologyEx(chars_denoised, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), iterations=2)
cv2.imshow("closed char denoised", chars_denoised); cv2.waitKey(0)
chars_clean = keep_main_plate_chars(chars_denoised)  # or chars_final / chars_denoised
cv2.imshow("chars_clean", chars_clean); cv2.waitKey(0)

chars_clean = bridge_gaps(chars_clean)
cv2.imshow("chars_clean_bridged", chars_clean); cv2.waitKey(0)

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

cv2.imshow("clean_word_only", clean_word)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

cv2.imshow("chars_only_rect_filter", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

cv2.imshow("chars_mask_tight", mask_tight)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2) "Rectangle again" but DONE RIGHT: merge letters slightly then pick best contour rectangle
# (This finds the center region and kills left/right junk.)
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

cv2.imshow("word_mask", word_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

print("Detected characters (capped):", len(chars))

chars = [(x, y, cw, ch) for (x, y, cw, ch, area) in chars]

# ---------------------------------------
# PROCESS CHARACTERS PROPERLY
# ---------------------------------------

processed_chars = []

processed_chars = []

for i, (x, y, cw, ch) in enumerate(chars):
    char_img = img_bin[y:y+ch, x:x+cw].copy()

    # -----------------------------
    # 1) crop tightly first
    # -----------------------------
    ys, xs = np.where(char_img > 0)
    if xs.size == 0:
        continue

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    char_img = char_img[y0:y1+1, x0:x1+1]

    # -----------------------------
    # 2) upscale first
    # makes morphology smoother
    # -----------------------------
    char_img = cv2.resize(
        char_img, None, fx=3, fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # ensure binary again
    _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)

    # -----------------------------
    # 3) fill tiny holes
    # -----------------------------
    char_img = fill_small_holes(char_img, max_hole_frac=0.01)

    # -----------------------------
    # 4) smooth / connect bumps
    # -----------------------------
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, k_close, iterations=1)

    # -----------------------------
    # 5) thicken strokes
    # -----------------------------
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    char_img = cv2.dilate(char_img, k_dilate, iterations=1)

    # optional extra thickening if still too thin
    # char_img = cv2.dilate(char_img, k_dilate, iterations=1)

    # -----------------------------
    # 6) one final close
    # -----------------------------
    char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, k_close, iterations=1)

    processed_chars.append(char_img)
    cv2.imshow(f"char_processed_{i}", char_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


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
    cv2.imshow(f"char_norm_{i}", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()

#-----------------------------------UNCOMMENT AFTER HERE

# # ---------------------------------------
# # CLASSIFICATION (NOW CORRECT)
# # ---------------------------------------
# plate_text = ""

# for idx, char_img in enumerate(normalized_chars):

#     print("\n===================================")
#     print("Processing plate char index:", idx)
#     print("===================================")

#     # Show exact input to classifier
#     cv2.imshow("INPUT_TO_CLASSIFIER", char_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # preprocess -> A (0/1)
#     A, cleaned, bin_norm = preprocess_segmented(
#         char_img,
#         visualize=True,
#         plate_mode=True
#     )

#     # skeleton + prune (ONLY for endpoint logic)
#     skel = thin(A)
#     sk_pruned = prune_spurs(skel, max_length=2)

#     # predict
#     pred = classify_letter(A, skel)
#     pred = pred if pred is not None else "?"
#     plate_text += pred

#     # -----------------------
#     # Feature prints (BINARY ONLY)
#     # -----------------------
#     hc, hpct = hole_count_and_largest_pct(A)
#     vs = vertical_symmetry_lr_balance(A)
#     hs = horizontal_symmetry_tb_balance(A)

#     vstrokes = count_vertical_strokes(A, min_frac=0.6)
#     hstrokes = count_horizontal_strokes(
#         A,
#         min_frac=0.8,
#         gap_allow=2,
#         band=5,
#         support_rows=3,
#         orient_thresh=0.12,
#         rel_width_keep=0.80
#     )

#     north, south = concavity_tb_strength(A)
#     bw = bottom_width_ratio(A, band_frac=0.20)
#     ratio = center_density_ratio(A, frac=0.35)
#     left_ratio, right_ratio = side_open_score(A)

#     # -----------------------
#     # Feature prints (SKELETON ONLY)
#     # -----------------------
#     e = count_endpoints(sk_pruned)

#     print("Predicted:", pred)
#     print("Ratios (left, right):", round(left_ratio, 3), round(right_ratio, 3))
#     print("Center density ratio:", ratio)
#     print("Bottom width ratio:", round(bw, 3))
#     print("Endpoints(after prune):", e)
#     print("Vertical symmetry score:", round(vs, 3))
#     print("Horizontal symmetry score:", round(hs, 3))
#     print("vstrokes:", vstrokes, "| hstrokes:", hstrokes)
#     print("hole count:", hc, "| hole %:", round(hpct, 2))
#     print(f"north={north:.3f} | south={south:.3f}")

#     # -----------------------
#     # Debug visuals
#     # -----------------------

#     # 1) Skeleton endpoints (uses skeleton)
#     debug_endpoints(sk_pruned)

#     # 2) Stroke debug overlays (use A only)
#     debug_vertical_strokes(A, min_frac=0.6, gap_allow=2, support_cols=3, orient_thresh=0.12)
#     debug_horizontal_strokes(A, min_frac=0.8, gap_allow=2, band=5, support_rows=3, orient_thresh=0.12, rel_width_keep=0.80)

#     # 3) Optional: show skeleton image itself
#     cv2.imshow("SKEL", (skel * 255).astype(np.uint8))
#     cv2.imshow("SKEL_PRUNED", (sk_pruned * 255).astype(np.uint8))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     import matplotlib.pyplot as plt
#     plt.close("all")

# print("\nFINAL PLATE:", plate_text)


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
    print("number of Blobs on A (before thinning):", n_blobs0)

    # Base view (always)
    plt.figure(figsize=(10, 3))
    plt.suptitle(f"{title} | start blobs={n_blobs0}")
    plt.subplot(1,3,1); plt.imshow(A, cmap="gray");      plt.title("A (binary)"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(A_thin, cmap="gray"); plt.title("A_thin");     plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(blobs0, cmap="gray"); plt.title(f"blobs0 (n={n_blobs0})"); plt.axis("off")
    plt.tight_layout(); plt.show()

    # ---- Group 1 debug: stems ----
    if n_blobs0 > 0:
        stems_img, n_stems, _ = get_stems(A, blobs0)

        plt.figure(figsize=(8, 3))
        plt.suptitle(f"{title} | Group1 stems={n_stems}")
        plt.subplot(1,2,1); plt.imshow(A_thin, cmap="gray");    plt.title("A_thin"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(stems_img, cmap="gray"); plt.title("stems_img"); plt.axis("off")
        plt.tight_layout(); plt.show()
        return  # done

    # ---- Group 2 debug: helper lines ----
    pts = get_banded_points(A_thin, split=0.5)
    if pts is None:
        print("[debug_digit_steps] No banded points found.")
        return

    TL, BL, TR, BR = pts

    # show points
    plt.figure(figsize=(6, 3))
    plt.title(f"{title} | Group2 points (TL BL TR BR)")
    plt.imshow(A_thin, cmap="gray")
    plt.scatter([TL[0], BL[0], TR[0], BR[0]],
                [TL[1], BL[1], TR[1], BR[1]], c="red", s=25)
    plt.axis("off")
    plt.show()

    # A1: TL->BL
    A1 = draw_line(A_thin, TL, BL)
    blobs1, nb1 = find_blobs(A1, min_blob_area=1)

    plt.figure(figsize=(10, 3))
    plt.suptitle(f"{title} | Step A1 TL→BL | nb1={nb1}")
    plt.subplot(1,2,1); plt.imshow(A1, cmap="gray");     plt.title("A1"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(blobs1, cmap="gray"); plt.title("blobs1"); plt.axis("off")
    plt.tight_layout(); plt.show()

    if nb1 == 0:
        # A2: TL->BR (1 vs 4)
        A2 = draw_line(A_thin, TL, BR)
        blobs2, nb2 = find_blobs(A2, min_blob_area=1)

        plt.figure(figsize=(10, 3))
        plt.suptitle(f"{title} | Step A2 TL→BR | nb2={nb2} (1 if 0 else 4)")
        plt.subplot(1,2,1); plt.imshow(A2, cmap="gray");     plt.title("A2"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(blobs2, cmap="gray"); plt.title("blobs2"); plt.axis("off")
        plt.tight_layout(); plt.show()
        return

    # stems on A1 (2 vs 5)
    stems_img, n_stems, _ = get_stems(A1, blobs1)
    plt.figure(figsize=(10, 3))
    plt.suptitle(f"{title} | stems on A1 (2 vs 5) | n_stems={n_stems}")
    plt.subplot(1,2,1); plt.imshow(A1, cmap="gray");       plt.title("A1"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(stems_img, cmap="gray"); plt.title("stems_img"); plt.axis("off")
    plt.tight_layout(); plt.show()

    if n_stems == 0:
        # A3: TR->BR (3 vs 7)
        A3 = draw_line(A_thin, TR, BR)
        blobs3, nb3 = find_blobs(A3, min_blob_area=1)

        plt.figure(figsize=(10, 3))
        plt.suptitle(f"{title} | Step A3 TR→BR | nb3={nb3} (3 if >0 else 7)")
        plt.subplot(1,2,1); plt.imshow(A3, cmap="gray");     plt.title("A3"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(blobs3, cmap="gray"); plt.title("blobs3"); plt.axis("off")
        plt.tight_layout(); plt.show()

from .classification import classify_with_blobs_from_A

def char_type_from_index(i: int) -> str:
    if i < 3:
        return "letter"
    else:
        return "digit"

plate_text = ""

for idx, char_img in enumerate(normalized_chars):

    ctype = char_type_from_index(idx)

    print("\n===================================")
    print(f"Processing plate char index: {idx} | type={ctype}")
    print("===================================")

    cv2.imshow("INPUT_TO_CLASSIFIER", char_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 1) ONE preprocess for both digit/letter
    A, cleaned, bin_norm = preprocess_segmented(char_img, visualize=True, plate_mode=True)

    # skeleton always computed (you want debug visuals anyway)
    skel = thin(A)
    sk_pruned = prune_spurs(skel, max_length=2)
    e = count_endpoints(sk_pruned)

    if ctype == "digit":
        # -----------------------
        # DIGIT CLASSIFICATION
        # -----------------------
        digit, group = classify_with_blobs_from_A(A, debug=True)
        if ctype == "digit":
            debug_digit_steps(A, title=f"plate idx={idx} pred={digit} group={group}")
        pred = str(digit) if digit is not None else "?"
        plate_text += pred

        print("Predicted DIGIT:", pred, "|", group)

        # Optional: show digit-specific debug visuals (blobs/stems)
        # blobs, n_blobs, stems_img, n_stems = summarize_blobs_and_stems(thin(A), tag=f"DIGIT idx={idx}")

    else:
        # -----------------------
        # LETTER CLASSIFICATION
        # -----------------------
        pred = classify_letter(A, skel)
        pred = pred if pred is not None else "?"
        plate_text += pred

        # --- Your existing letter feature prints ---
        hc, hpct = hole_count_and_largest_pct(A)
        vs = vertical_symmetry_lr_balance(A)
        hs = horizontal_symmetry_tb_balance(A)

        vstrokes = count_vertical_strokes(A, min_frac=0.6)
        hstrokes = count_horizontal_strokes(
            A, min_frac=0.8, gap_allow=2, band=5, support_rows=3,
            orient_thresh=0.12, rel_width_keep=0.80
        )

        north, south = concavity_tb_strength(A)
        bw = bottom_width_ratio(A, band_frac=0.20)
        ratio = center_density_ratio(A, frac=0.35)
        left_ratio, right_ratio = side_open_score(A)
        west, east = concavity_lr_strength(A)

        print("Predicted LETTER:", pred)
        print("Ratios (left, right):", round(left_ratio, 3), round(right_ratio, 3))
        print("Center density ratio:", ratio)
        print("Bottom width ratio:", round(bw, 3))
        print("Endpoints(after prune):", e)
        print("Vertical symmetry score:", round(vs, 3))
        print("Horizontal symmetry score:", round(hs, 3))
        print("vstrokes:", vstrokes, "| hstrokes:", hstrokes)
        print("hole count:", hc, "| hole %:", round(hpct, 2))
        print(f"north={north:.3f} | south={south:.3f}")
        print(f"west={west:.3f} | east={east:.3f}")

        # --- Your debug visuals ---
        debug_endpoints(sk_pruned)
        debug_vertical_strokes(
        A,
        min_frac=0.6,
        gap_allow=2,
        band=5,
        support_cols=3,
        orient_thresh=0.12,
        rel_height_keep=0.80,
        rel_width_keep=0.30
        )
        debug_horizontal_strokes(A, min_frac=0.8, gap_allow=2, band=5, support_rows=3,
                                 orient_thresh=0.12, rel_width_keep=0.80)

    # always show skeletons (both digit & letter)
    cv2.imshow("SKEL", (skel * 255).astype(np.uint8))
    cv2.imshow("SKEL_PRUNED", (sk_pruned * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    import matplotlib.pyplot as plt
    plt.close("all")

print("\nFINAL PLATE:", plate_text)
def char_type_from_index(i: int) -> str:
    if i < 3:
        return "letter"
    return "digit"   # i >= 4