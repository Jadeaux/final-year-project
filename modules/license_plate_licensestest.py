import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from .classification_letters import classify_letter
from .preprocess_for_segmented import preprocess_letters as preprocess_segmented
from .preprocessing import thin
from .features_letters import concavity_tb_strength, concavity_lr_strength, count_horizontal_strokes, debug_endpoints, debug_horizontal_strokes, horizontal_symmetry_tb_balance, prune_spurs, count_endpoints, hole_count_and_largest_pct, vertical_symmetry_lr_balance, count_vertical_strokes, bottom_width_ratio, center_density_ratio, debug_vertical_strokes, debug_bottom_width_ratio, side_open_score
import glob
import cv2
import numpy as np
from .classification import classify_with_blobs_from_A

from .morphology import find_blobs
from .features import get_stems, get_banded_points, draw_line

OUT_DIR = "lpr_outputs/debug_plate"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("lpr_outputs/plates", exist_ok=True)



def debug_digit_steps(A, title="digit-debug"):
    if not debug:
        return
    
    import numpy as np
    import matplotlib.pyplot as plt

    A_thin = thin(A)

    blobs0, n_blobs0 = find_blobs(A)
    print("number of Blobs on A (before thinning):", n_blobs0)

    plt.figure(figsize=(10, 3))
    plt.suptitle(f"{title} | start blobs={n_blobs0}")
    plt.subplot(1,3,1); plt.imshow(A, cmap="gray");      plt.title("A (binary)"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(A_thin, cmap="gray"); plt.title("A_thin");     plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(blobs0, cmap="gray"); plt.title(f"blobs0 (n={n_blobs0})"); plt.axis("off")
    plt.tight_layout(); plt.show()

    if n_blobs0 > 0:
        stems_img, n_stems, _ = get_stems(A, blobs0)

        plt.figure(figsize=(8, 3))
        plt.suptitle(f"{title} | Group1 stems={n_stems}")
        plt.subplot(1,2,1); plt.imshow(A_thin, cmap="gray");    plt.title("A_thin"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(stems_img, cmap="gray"); plt.title("stems_img"); plt.axis("off")
        plt.tight_layout(); plt.show()
        return

    pts = get_banded_points(A_thin, split=0.5)
    if pts is None:
        print("[debug_digit_steps] No banded points found.")
        return

    TL, BL, TR, BR = pts

    plt.figure(figsize=(6, 3))
    plt.title(f"{title} | Group2 points (TL BL TR BR)")
    plt.imshow(A_thin, cmap="gray")
    plt.scatter([TL[0], BL[0], TR[0], BR[0]],
                [TL[1], BL[1], TR[1], BR[1]], c="red", s=25)
    plt.axis("off")
    plt.show()

    A1 = draw_line(A_thin, TL, BL)
    blobs1, nb1 = find_blobs(A1, min_blob_area=1)

    plt.figure(figsize=(10, 3))
    plt.suptitle(f"{title} | Step A1 TL→BL | nb1={nb1}")
    plt.subplot(1,2,1); plt.imshow(A1, cmap="gray");     plt.title("A1"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(blobs1, cmap="gray"); plt.title("blobs1"); plt.axis("off")
    plt.tight_layout(); plt.show()

    if nb1 == 0:
        A2 = draw_line(A_thin, TL, BR)
        blobs2, nb2 = find_blobs(A2, min_blob_area=1)

        plt.figure(figsize=(10, 3))
        plt.suptitle(f"{title} | Step A2 TL→BR | nb2={nb2} (1 if 0 else 4)")
        plt.subplot(1,2,1); plt.imshow(A2, cmap="gray");     plt.title("A2"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(blobs2, cmap="gray"); plt.title("blobs2"); plt.axis("off")
        plt.tight_layout(); plt.show()
        return

    stems_img, n_stems, _ = get_stems(A1, blobs1)
    plt.figure(figsize=(10, 3))
    plt.suptitle(f"{title} | stems on A1 (2 vs 5) | n_stems={n_stems}")
    plt.subplot(1,2,1); plt.imshow(A1, cmap="gray");       plt.title("A1"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(stems_img, cmap="gray"); plt.title("stems_img"); plt.axis("off")
    plt.tight_layout(); plt.show()

    if n_stems == 0:
        A3 = draw_line(A_thin, TR, BR)
        blobs3, nb3 = find_blobs(A3, min_blob_area=1)

        plt.figure(figsize=(10, 3))
        plt.suptitle(f"{title} | Step A3 TR→BR | nb3={nb3} (3 if >0 else 7)")
        plt.subplot(1,2,1); plt.imshow(A3, cmap="gray");     plt.title("A3"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(blobs3, cmap="gray"); plt.title("blobs3"); plt.axis("off")
        plt.tight_layout(); plt.show()



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
                         min_ok_w=500,
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

def recognize_plate(img_path, debug=False):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {img_path}")

    H, W = img.shape[:2]
    print("Loaded:", img_path, "shape:", img.shape)

    if debug:
        cv2.imshow("00_input", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    k_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_bh)

    if debug:
        cv2.imshow("bh", blackhat)
        cv2.waitKey(0)

    _, bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if debug:
        cv2.imshow("bw_bh", bw)
        cv2.waitKey(0)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=2)

    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    bw = cv2.dilate(bw, k_dil, iterations=1)

    if debug:
        cv2.imshow("bw_plate_blob", bw)
        cv2.waitKey(0)

    margin = 10
    bw[:margin, :] = 0
    bw[-margin:, :] = 0
    bw[:, :margin] = 0
    bw[:, -margin:] = 0

    mask_roi = np.zeros_like(bw)
    y_start = int(0.35 * H)
    y_end   = int(0.90 * H)
    mask_roi[y_start:y_end, :] = 255
    bw = cv2.bitwise_and(bw, mask_roi)

    if debug:
        cv2.imshow("03c_bw_band", bw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(f"{OUT_DIR}/03b_bw_noborder.png", bw)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imwrite(f"{OUT_DIR}/04_morph.png", morph)

    if debug:
        cv2.imshow("04_morph", morph)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = img.copy()

    print("\nContour candidates:\n")

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        area = w * h
        print(f"x={x}, y={y}, w={w}, h={h}, ar={ar:.2f}, area={area}")
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)

    if debug:
        cv2.imshow("05_all_contours", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

        cx = x + w/2
        cy = y + h/2
        center_dist = abs(cx - W/2) / W + abs(cy - H/2) / H

        score = (w * h) / (1.0 + 5.0 * center_dist)

        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        print("No plate chosen")
    else:
        x, y, w, h = best
        print("CHOSEN PLATE:", best)

        x, y, w, h = pad_bbox(x, y, w, h, H, W)

        plate_roi = img[y:y+h, x:x+w].copy()
        print("ROI shape (pre-norm):", plate_roi.shape)

        plate_roi, scale = normalize_plate_size(
            plate_roi,
            min_ok_w=400,
            target_w=600,
            max_ok_w=600
        )

        print("ROI shape (post-norm):", plate_roi.shape, "| scale:", scale)
        cv2.imwrite("lpr_outputs/plates/plate1_roi.png", plate_roi)

        vis2 = img.copy()
        cv2.rectangle(vis2, (x,y), (x+w, y+h), (0,0,255), 3)

        if debug:
            cv2.imshow("07_chosen_plate", vis2)
            cv2.imshow("07_plate_roi", plate_roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    pgray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    pgray = cv2.GaussianBlur(pgray, (3,3), 0)

    chars = cv2.adaptiveThreshold(
        pgray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

    ys, xs = np.where(chars > 0)

    if len(xs) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        pad = 8
        x0 = max(0, x0 - pad); x1 = min(chars.shape[1]-1, x1 + pad)
        y0 = max(0, y0 - pad); y1 = min(chars.shape[0]-1, y1 + pad)

        plate_roi = plate_roi[y0:y1+1, x0:x1+1]
        chars     = chars[y0:y1+1, x0:x1+1]

    if debug:
        cv2.imshow("plate_roi_tight", plate_roi)
        cv2.imshow("chars_mask_tight", chars)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    chars = cv2.morphologyEx(chars, cv2.MORPH_CLOSE, k, iterations=1)

    cv2.imwrite("lpr_outputs/debug/plate1_chars.png", chars)

    if debug:
        cv2.imshow("08_chars_mask", chars)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def bridge_gaps(bin255):
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
        out = cv2.morphologyEx(bin255, cv2.MORPH_CLOSE, k_close, iterations=1)
        out = cv2.dilate(out, k_close, iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=1)
        return out

    proj = np.sum(chars, axis=1).astype(np.float32)
    proj_s = cv2.GaussianBlur(proj.reshape(-1,1), (1,31), 0).flatten()

    proj_n = proj_s / (proj_s.max() + 1e-6)

    thr = max(0.10, 0.5 * np.median(proj_n[proj_n > 0]))
    mask_rows = (proj_n > thr).astype(np.uint8)
    print("row thr:", thr)

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
        top, bottom = max(runs, key=lambda r: r[1] - r[0])

    pad = 3
    top = max(0, top - pad)
    bottom = min(chars.shape[0] - 1, bottom + pad)

    plate_refined = plate_roi[top:bottom+1, :]
    chars_refined = chars[top:bottom+1, :]

    if debug:
        cv2.imshow("refined_roi", plate_refined)
        cv2.imshow("refined_chars", chars_refined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    hR, wR = chars_refined.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(chars_refined, connectivity=8)

    max_h = 0
    for i in range(1, num):
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if ch > max_h:
            max_h = ch

    min_h = int(0.70 * max_h)

    xs = []
    xe = []

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
        left = min(xs)
        right = max(xe)

    pad_lr = int(0.03 * wR)
    left = max(0, left - pad_lr)
    right = min(wR - 1, right + pad_lr)

    plate_final = plate_refined[:, left:right]
    chars_final = chars_refined[:, left:right]

    print("CC left/right:", left, right, "min_h:", min_h, "max_h:", max_h)

    if debug:
        cv2.imshow("final_zoom_roi", plate_final)
        cv2.imshow("final_zoom_chars", chars_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    hR, wR = chars_refined.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(chars_refined, connectivity=8)

    max_h = max(stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num)) if num > 1 else 0
    min_h = int(0.65 * max_h)

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

    pad_lr = int(0.04 * wR)
    left = max(0, left - pad_lr)
    right = min(wR - 1, right + pad_lr)

    chars_final = chars_refined[:, left:right]
    plate_final = plate_refined[:, left:right]

    bin255 = chars_final.copy()

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8)
    min_area = 80
    den = np.zeros_like(bin255)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            den[labels == i] = 255

    if debug:
        cv2.imshow("den", den)
        cv2.waitKey(0)

    k_word = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    blob = cv2.morphologyEx(den, cv2.MORPH_CLOSE, k_word, iterations=2)
    blob = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_RECT, (5,3)), iterations=1)

    if debug:
        cv2.imshow("blob", blob)
        cv2.waitKey(0)

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = den.shape
    best = None
    best_score = -1

    print("img shape:", img.shape)
    print("bw shape:", bw.shape)
    print("morph shape:", morph.shape)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ar = w / (h + 1e-6)

        if ar < 2.0:
            continue
        if h < 0.45 * H:
            continue
        if area < 0.15 * W * H:
            continue

        roi = den[y:y+h, x:x+w]
        ink = np.sum(roi > 0)
        density = ink / (area + 1e-6)

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

    if debug:
        cv2.imshow("word_crop", word_crop)
        cv2.waitKey(0)

    print("CC left/right:", left, right, "min_h:", min_h, "max_h:", max_h)

    if debug:
        cv2.imshow("final_zoom_chars", chars_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    h, w = chars_final.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(chars_final, connectivity=8)

    min_area = 80
    chars_denoised = np.zeros_like(chars_final)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            chars_denoised[labels == i] = 255

    bin255 = chars_denoised

    if debug:
        cv2.imshow("chars_denoised", bin255)
        cv2.waitKey(0)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    blob = cv2.morphologyEx(bin255, cv2.MORPH_CLOSE, k, iterations=1)
    blob = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_RECT, (7,3)), iterations=1)

    if debug:
        cv2.imshow("word_blob_candidates", blob)
        cv2.waitKey(0)

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = bin255.shape
    best = None
    best_score = -1

    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        ar = ww / (hh + 1e-6)

        if ar < 2.0:
            continue
        if area < 0.05 * W * H:
            continue
        if hh < 0.40 * H:
            continue

        roi = bin255[y:y+hh, x:x+ww]
        ink = np.sum(roi > 0)
        density = ink / (area + 1e-6)

        touches_left  = x <= int(0.02 * W)
        touches_right = (x + ww) >= int(0.98 * W)
        if touches_left and touches_right:
            continue

        score = density * area
        if score > best_score:
            best_score = score
            best = (x, y, ww, hh)

    if best is None:
        word_mask = bin255.copy()
    else:
        x, y, ww, hh = best
        pad = 4
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + ww + pad); y1 = min(H, y + hh + pad)

        word_mask = np.zeros_like(bin255)
        word_mask[y0:y1, x0:x1] = bin255[y0:y1, x0:x1]

    if debug:
        cv2.imshow("word_mask_kept", word_mask)
        cv2.waitKey(0)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            chars_denoised[labels == i] = 255

    if debug:
        cv2.imshow("chars_denoised", chars_denoised)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    chars_clean = keep_main_plate_chars(chars_denoised)
    if debug:
        cv2.imshow("chars_clean", chars_clean)
        cv2.waitKey(0)

    chars_clean = bridge_gaps(chars_clean)
    if debug:
        cv2.imshow("chars_clean_bridged", chars_clean)
        cv2.waitKey(0)

    img_bin = chars_clean.copy()
    h, w = img_bin.shape

    col_sum = np.sum(img_bin > 0, axis=0).astype(np.float32)
    col_sum_s = cv2.GaussianBlur(col_sum.reshape(1, -1), (31,1), 0).flatten()
    col_sum_n = col_sum_s / (np.max(col_sum_s) + 1e-6)

    thr = 0.15
    cols = np.where(col_sum_n > thr)[0]

    left = cols[0]
    right = cols[-1]

    pad = 5
    left = max(0, left - pad)
    right = min(w-1, right + pad)

    clean_word = img_bin[:, left:right+1]

    if debug:
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

        if ch < 0.4 * h:
            continue
        if cw < 0.02 * w:
            continue
        if ar < 0.10 or ar > 1.20:
            continue
        if area < 80:
            continue

        out[labels == i] = 255

    if debug:
        cv2.imshow("chars_only_rect_filter", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    mask = (out > 0).astype(np.uint8) * 255

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Mask is empty after chars_only_rect_filter")

    pad = 3
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); x1 = min(mask.shape[1]-1, x1 + pad)
    y0 = max(0, y0 - pad); y1 = min(mask.shape[0]-1, y1 + pad)

    mask_tight = mask[y0:y1+1, x0:x1+1]

    if debug:
        cv2.imshow("chars_mask_tight", mask_tight)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    merged = cv2.dilate(mask_tight, k, iterations=1)

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
        boxes = sorted(boxes, key=lambda b: b[0])

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

    if debug:
        cv2.imshow("word_mask", word_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

        chars.append((x, y, cw, ch, area))

    MAX_CHARS = 7
    if len(chars) > MAX_CHARS:
        chars = sorted(chars, key=lambda c: c[4])
        chars = chars[len(chars) - MAX_CHARS:]

    chars = sorted(chars, key=lambda c: c[0])

    print("Detected characters (capped):", len(chars))

    chars = [(x, y, cw, ch) for (x, y, cw, ch, area) in chars]

    processed_chars = []

    for i, (x, y, cw, ch) in enumerate(chars):
        char_img = img_bin[y:y+ch, x:x+cw].copy()

        char_img = fill_small_holes(char_img, max_hole_frac=0.05)
        char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, k1, iterations=1)
        char_img = cv2.dilate(char_img, k1, iterations=1)
        char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, k2, iterations=1)

        processed_chars.append(char_img)
        if debug:
            cv2.imshow(f"char_processed_{i}", char_img)

    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        if debug:
            cv2.imshow(f"char_norm_{i}", resized)

    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    plate_text = ""

    for idx, char_img in enumerate(normalized_chars):

        ctype = char_type_from_index(idx)

        print("\n===================================")
        print(f"Processing plate char index: {idx} | type={ctype}")
        print("===================================")

        if debug:
            cv2.imshow("INPUT_TO_CLASSIFIER", char_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        A, cleaned, bin_norm = preprocess_segmented(char_img, visualize=debug, plate_mode=True)

        skel = thin(A)
        sk_pruned = prune_spurs(skel, max_length=2)
        e = count_endpoints(sk_pruned)

        if ctype == "digit":
            digit, group = classify_with_blobs_from_A(A, debug=debug)
            if debug and ctype == "digit":
                debug_digit_steps(A, title=f"plate idx={idx} pred={digit} group={group}")
            pred = str(digit) if digit is not None else "?"
            plate_text += pred

            print("Predicted DIGIT:", pred, "|", group)

        else:
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

            if debug:
                debug_endpoints(sk_pruned)
                debug_vertical_strokes(A, min_frac=0.7, gap_allow=2, support_cols=3, orient_thresh=0.12)
                debug_horizontal_strokes(A, min_frac=0.8, gap_allow=2, band=5, support_rows=3,
                                        orient_thresh=0.12, rel_width_keep=0.80)

        if debug:
            cv2.imshow("SKEL", (skel * 255).astype(np.uint8))
            cv2.imshow("SKEL_PRUNED", (sk_pruned * 255).astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            import matplotlib.pyplot as plt
            plt.close("all")

    print("\nFINAL PLATE:", plate_text)
    return plate_text

PLATES_DIR = "images/plates"
LABELS_XLSX = "images/license_plate_labels.xlsx"   # adjust if needed
OUT_CSV = "lpr_outputs/plate_predictions_datasetSurvey.csv"

# =========================================================
# HELPERS
# =========================================================
def char_type_from_index(i: int) -> str:
    if i == 0:
        return "digit"
    if 1 <= i <= 3:
        return "letter"
    return "digit"


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


import pandas as pd
# =========================================================
# YOUR PIPELINE WRAPPER
# =========================================================

# =========================================================
# BATCH EVALUATION
# =========================================================
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

        pred_plate = recognize_plate(img_path, debug=False)
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

    # Save CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_CSV, index=False)

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
    print(f"Saved CSV to: {OUT_CSV}")


if __name__ == "__main__":
    run_batch_evaluation()