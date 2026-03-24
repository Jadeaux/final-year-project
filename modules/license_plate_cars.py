import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from .classification_kumar import classify_letter
from .preprocess_for_segmented import preprocess_letters as preprocess_segmented
from .preprocessing import thin
from .features_letters import concavity_tb_strength, concavity_lr_strength, count_horizontal_strokes, debug_endpoints, debug_horizontal_strokes, horizontal_symmetry_tb_balance, prune_spurs, count_endpoints, hole_count_and_largest_pct, vertical_symmetry_lr_balance, count_vertical_strokes, bottom_width_ratio, center_density_ratio, debug_vertical_strokes, debug_bottom_width_ratio, side_open_score
from .classification_original import classify_with_blobs_from_A
OUT_DIR = "lpr_outputs/debug"
os.makedirs(OUT_DIR, exist_ok=True)

def remove_plate_frame_and_symbol(bin255):
    H, W = bin255.shape
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin255, connectivity=8)

    out = np.zeros_like(bin255)

    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]

        ar = w / float(h + 1e-6)

        # remove tall border-touching frame pieces
        touches_left = x <= 1
        touches_right = (x + w) >= (W - 1)
        touches_top = y <= 1
        touches_bottom = (y + h) >= (H - 1)

        if (touches_left or touches_right or touches_top or touches_bottom):
            if area > 0.01 * H * W:
                continue
            if h > 0.60 * H or w > 0.60 * W:
                continue

        # remove small central symbol blobs
        if (0.35 * W <= cx <= 0.65 * W) and area < 0.04 * H * W and h < 0.35 * H:
            continue

        box_area = w * h
        density = area / float(box_area + 1e-6)

        # remove frame-like components:
        # large bbox but low fill density
        if box_area > 0.20 * H * W and density < 0.35:
            continue

        out[labels == i] = 255

    return out

import cv2
import numpy as np

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


def count_char_like_ccs(bin255, min_char_h=8):
    """
    Count character-like connected components in a binary candidate region.
    Input: 0/255 image, white = foreground
    """
    H, W = bin255.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8)

    cnt = 0
    boxes = []

    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 20:
            continue
        if h < max(min_char_h, int(0.25 * H)):
            continue
        if h > 0.95 * H:
            continue
        if w < 2:
            continue

        ar = w / float(h + 1e-6)
        if ar < 0.08 or ar > 1.2:
            continue

        cnt += 1
        boxes.append((x, y, w, h))

    return cnt, boxes


def localize_plate_morphology(img, debug=False, out_dir="lpr_outputs/debug"):
    """
    Morphology-based plate localisation inspired by:
    'License Plate Localisation based on Morphological Operations'

    Returns:
        plate_roi, (x0, y0, x1, y1), debug_vis
    """
    H, W = img.shape[:2]

    # --------------------------------------------------
    # 1) grayscale
    # --------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # light blur just to stabilize noise a bit
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # --------------------------------------------------
    # 2) first opening: estimate smooth background
    # paper uses rectangular SE 4x30 on 768x288 images
    # here we scale approximately with image size
    # --------------------------------------------------
    se1_h = max(3, int(round(H * 4 / 288.0)))
    se1_w = max(15, int(round(W * 30 / 768.0)))
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (se1_w, se1_h))

    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se1)

    # highlight plate-texture region
    highlight = cv2.subtract(gray, bg)

    # --------------------------------------------------
    # 3) binarise highlighted image
    # paper uses threshold from max/min formula;
    # Otsu is more practical here and easier to tune
    # --------------------------------------------------
    _, bw = cv2.threshold(
        highlight, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------
    # 4) second opening + closing
    # paper uses two openings and one closing overall
    # --------------------------------------------------
    se2_h = max(2, int(round(H * 2 / 288.0)))
    se2_w = max(8, int(round(W * 12 / 768.0)))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (se2_w, se2_h))

    bw_open = bw.copy()

    se3_h = max(3, int(round(H * 6 / 288.0)))
    se3_w = max(12, int(round(W * 20 / 768.0)))
    se3 = cv2.getStructuringElement(cv2.MORPH_RECT, (se3_w, se3_h))

    bw_clean = cv2.morphologyEx(bw_open, cv2.MORPH_CLOSE, se3)

    # optional lower-half prior for front-facing car shots
    band = np.zeros_like(bw_clean)
    y0_band = int(0.40 * H)
    y1_band = int(0.95 * H)
    band[y0_band:y1_band, :] = 255
    bw_clean = cv2.bitwise_and(bw_clean, band)

    # remove borders
    m = 10
    bw_clean[:m, :] = 0
    bw_clean[-m:, :] = 0
    bw_clean[:, :m] = 0
    bw_clean[:, -m:] = 0

    # --------------------------------------------------
    # 5) candidate regions from connected components
    # --------------------------------------------------
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw_clean, connectivity=8)

    best = None
    best_score = -1
    vis = img.copy()

    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 100:
            continue

        ar = w / float(h + 1e-6)

        # plate-like geometry
                # plate-like geometry (slightly more tolerant)
        if ar < 1.6 or ar > 10.0:
            continue
        if w < 0.06 * W or w > 0.75 * W:
            continue
        if h < 0.025 * H or h > 0.30 * H:
            continue

        # extract candidate from original grayscale-highlight binary
        pad_x = int(0.10 * w)
        pad_y = int(0.10* h)

        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)

        cand_gray = gray[y0:y1, x0:x1]

        # threshold candidate so we can validate using character-like CCs
        cand_bin = cv2.adaptiveThreshold(
            cand_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 10
        )

        cc_count, boxes = count_char_like_ccs(cand_bin)

        # similar idea to paper: real plate should contain a plausible
        # number of character-like connected components
        if cc_count < 1 or cc_count > 20:
            continue

        row_centers = [yy + hh / 2.0 for (_, yy, _, hh) in boxes] if boxes else [0]
        row_spread = np.std(row_centers) if len(row_centers) > 1 else 0.0

        score = (
            25.0 * cc_count +
            0.05 * w -
            6.0 * row_spread
        )

        if score > best_score:
            best_score = score
            best = (x0, y0, x1, y1, cand_bin, cc_count)

        if debug:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)

    if best is None:
        raise RuntimeError("No valid plate candidate found with morphology localisation.")

    x0, y0, x1, y1, cand_bin, cc_count = best
    plate_roi = img[y0:y1, x0:x1].copy()

    # normalize ROI width for downstream segmentation
    plate_roi, scale = normalize_plate_size(
        plate_roi,
        min_ok_w=400,
        target_w=600,
        max_ok_w=600
    )

    if debug:
        final_vis = img.copy()
        cv2.rectangle(final_vis, (x0, y0), (x1, y1), (0, 0, 255), 3)

        cv2.imshow("01_gray", gray)
        cv2.imshow("02_bg_open", bg)
        cv2.imshow("03_highlight", highlight)
        cv2.imshow("04_bw", bw)
        cv2.imshow("05_bw_clean", bw_clean)
        cv2.imshow("06_candidates", vis)
        cv2.imshow("07_final_plate_box", final_vis)
        cv2.imshow("08_plate_roi", plate_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return plate_roi, (x0, y0, x1, y1), vis

def bridge_gaps(bin255):
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    out = cv2.morphologyEx(bin255, cv2.MORPH_CLOSE, k_close, iterations=1)
    out = cv2.dilate(out, k_close, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=1)
    return out


def extract_character_components(bin255, max_chars=7, debug=False):
    """
    Extract character-like connected components from a plate mask.
    Input: 0/255 binary image, white = foreground
    Returns:
        chars_kept : cleaned mask of kept components
        boxes      : list of (x, y, w, h) sorted left->right
    """
    H, W = bin255.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8)

    candidates = []
    chars_kept = np.zeros_like(bin255)

    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        ar = w / float(h + 1e-6)
        cx = x + w / 2.0
        cy = y + h / 2.0

        # 1) tiny noise
        if area < 80:
            continue

        # 2) too short to be a main plate char
        if h < 0.38 * H:
            continue

        
        if area < 140 and w < 0.22 * H:
            continue

        # 4) side borders: tall, thin, touching left/right edge
        touches_left = x <= 1
        touches_right = (x + w) >= (W - 1)
        if (touches_left or touches_right) and h > 0.70 * H and w < 0.08 * W:
            continue

        # 5) ultra-thin vertical junk
        if ar < 0.05 and h > 0.65 * H:
            continue

        # 6) tiny symbol near middle (your dot/circle problem)
        # small area + not tall enough + near center horizontally

        # 7) reject absurd aspect ratios
        if ar < 0.08 or ar > 1.6:
            continue

        candidates.append((x, y, w, h, area, i))

    # sort left->right
    candidates = sorted(candidates, key=lambda c: c[0])

    # if too many, keep the biggest max_chars by area, then sort again left->right
    if len(candidates) > max_chars:
        candidates = sorted(candidates, key=lambda c: c[4], reverse=True)[:max_chars]
        candidates = sorted(candidates, key=lambda c: c[0])

    boxes = []
    for x, y, w, h, area, i in candidates:
        chars_kept[labels == i] = 255
        boxes.append((x, y, w, h))

    if debug:
        print("Kept character-like CCs:", len(boxes))
        print("Boxes:", boxes)

    return chars_kept, boxes


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


def char_type_from_index(i: int) -> str:
    if i < 3:
        return "letter"
    return "digit"

def show_debug(title, image, debug=False, wait=True):
    if not debug:
        return
    cv2.imshow(title, image)
    if wait:
        cv2.waitKey(0)

def close_debug(debug=False):
    if debug:
        cv2.destroyAllWindows()


def recognize_plate(img_or_path, debug=False):
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            raise FileNotFoundError(f"Could not load {img_or_path}")
        img_path = img_or_path
    else:
        img = img_or_path.copy()
        img_path = None

    H, W = img.shape[:2]

    if debug:
        print("Loaded:", img_path if img_path else "<array>", "shape:", img.shape)
        show_debug("00_input", img, debug=debug)
        close_debug(debug)

    # --------------------------------------------------
    # 1) plate localisation
    # --------------------------------------------------
    plate_roi, plate_box, _ = localize_plate_morphology(img, debug=debug)

    if debug:
        print("Chosen plate box:", plate_box)
        print("ROI shape:", plate_roi.shape)

    # optional save only in debug
    if debug:
        os.makedirs("lpr_outputs/plates", exist_ok=True)
        cv2.imwrite("lpr_outputs/plates/plate1_roi.png", plate_roi)

    # --------------------------------------------------
    # 2) threshold plate
    # --------------------------------------------------
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

    show_debug("plate_roi_tight", plate_roi, debug=debug)
    show_debug("chars_mask_tight", chars, debug=debug)
    close_debug(debug)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    chars = cv2.morphologyEx(chars, cv2.MORPH_CLOSE, k, iterations=1)

    if debug:
        cv2.imwrite("lpr_outputs/debug/plate1_chars.png", chars)

    show_debug("08_chars_mask", chars, debug=debug)
    close_debug(debug)

    # --------------------------------------------------
    # 3) horizontal band crop
    # --------------------------------------------------
    proj = np.sum(chars, axis=1).astype(np.float32)
    proj_s = cv2.GaussianBlur(proj.reshape(-1, 1), (1, 31), 0).flatten()
    proj_n = proj_s / (proj_s.max() + 1e-6)

    nz = proj_n[proj_n > 0]
    thr = max(0.10, 0.5 * np.median(nz)) if len(nz) else 0.10
    if debug:
        print("row thr:", thr)

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

    show_debug("refined_roi", plate_refined, debug=debug)
    show_debug("refined_chars", chars_refined, debug=debug)
    close_debug(debug)

    # --------------------------------------------------
    # 4) left-right crop
    # --------------------------------------------------
    hR, wR = chars_refined.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(chars_refined, connectivity=8)

    max_h = 0
    for i in range(1, num):
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if ch > max_h:
            max_h = ch

    min_h = int(0.70 * max_h) if max_h > 0 else 0

    xs = []
    xe = []

    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
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
        left = min(xs)
        right = max(xe)

    pad_lr = int(0.03 * wR)
    left = max(0, left - pad_lr)
    right = min(wR - 1, right + pad_lr)

    plate_final = plate_refined[:, left:right]
    chars_final = chars_refined[:, left:right]

    if debug:
        print("CC left/right:", left, right, "min_h:", min_h, "max_h:", max_h)

    show_debug("final_zoom_roi", plate_final, debug=debug)
    show_debug("final_zoom_chars", chars_final, debug=debug)
    close_debug(debug)

    # --------------------------------------------------
    # 5) remove frame + extract character CCs
    # --------------------------------------------------
    chars_noframe = remove_plate_frame_and_symbol(chars_final)
    show_debug("chars_noframe", chars_noframe, debug=debug)
    close_debug(debug)

    chars_kept, boxes = extract_character_components(chars_noframe, max_chars=7, debug=debug)

    show_debug("chars_kept", chars_kept, debug=debug)
    close_debug(debug)

    if debug:
        print("Detected characters:", len(boxes))

    # optional: strict length check
    if len(boxes) == 0:
        return ""

    # --------------------------------------------------
    # 6) process chars
    # --------------------------------------------------
    processed_chars = []
    for i, (x, y, w, h) in enumerate(boxes):
        char_img = chars_kept[y:y+h, x:x+w].copy()
        char_img = fill_small_holes(char_img, max_hole_frac=0.05)
        processed_chars.append(char_img)
        show_debug(f"char_processed_{i}", char_img, debug=debug, wait=False)
    close_debug(debug)

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
        pad = int(0.15 * size)

        square = np.zeros((size + 2 * pad, size + 2 * pad), dtype=np.uint8)
        y_off = (square.shape[0] - h) // 2
        x_off = (square.shape[1] - w) // 2
        square[y_off:y_off+h, x_off:x_off+w] = char_img

        resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        normalized_chars.append(resized)
        show_debug(f"char_norm_{i}", resized, debug=debug, wait=False)
    close_debug(debug)

    if debug:
        print("Normalized chars:", len(normalized_chars))

    # --------------------------------------------------
    # 7) classify
    # --------------------------------------------------
    plate_text = ""

    for idx, char_img in enumerate(normalized_chars):
        ctype = char_type_from_index(idx)

        if debug:
            print("\n===================================")
            print(f"Processing plate char index: {idx} | type={ctype}")
            print("===================================")
            show_debug("INPUT_TO_CLASSIFIER", char_img, debug=debug)
            close_debug(debug)

        A, cleaned, bin_norm = preprocess_segmented(char_img, visualize=debug, plate_mode=True)
        skel = thin(A)
        sk_pruned = prune_spurs(skel, max_length=2)

        if ctype == "digit":
            if debug:
                print(f"[DIGIT] entering idx={idx}")

            digit, group = classify_with_blobs_from_A(A, debug=False)

            if debug:
                print(f"[DIGIT] returned idx={idx} digit={digit} group={group}")
                # comment this out for now while debugging the pipeline flow
                # debug_digit_steps(A, title=f"plate idx={idx} pred={digit} group={group}")
                print("Predicted DIGIT:", digit, "|", group)

            pred = str(digit) if digit is not None else "?"
            plate_text += pred
        else:
            pred = classify_letter(A, skel)
            pred = pred if pred is not None else "?"
            plate_text += pred

            if debug:
                e = count_endpoints(sk_pruned)
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

                debug_endpoints(sk_pruned)
                debug_vertical_strokes(A, min_frac=0.7, gap_allow=2, support_cols=3, orient_thresh=0.12)
                debug_horizontal_strokes(
                    A, min_frac=0.8, gap_allow=2, band=5, support_rows=3,
                    orient_thresh=0.12, rel_width_keep=0.80
                )

        show_debug("SKEL", (skel * 255).astype(np.uint8), debug=debug, wait=False)
        show_debug("SKEL_PRUNED", (sk_pruned * 255).astype(np.uint8), debug=debug, wait=False)
        close_debug(debug)
        plt.close("all")

    if debug:
        print("\nFINAL PLATE:", plate_text)

    return plate_text


if __name__ == "__main__":
    test_path = "E:/plates_dataset/1-10/DSCN0415.jpg"
    pred = recognize_plate(test_path, debug=True)
    print("Prediction:", pred)