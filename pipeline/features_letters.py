import numpy as np
import cv2
import matplotlib.pyplot as plt
from .features import get_banded_points, draw_line

# -------------------------
# CORE FEATURES (no plots)
# -------------------------

def count_holes(A01: np.ndarray, min_hole_area: int = 30) -> int:
    """
    Count holes in a binary image A01 (foreground=1).
    Filters out tiny specks using min_hole_area.
    """
    A01 = (A01 > 0).astype(np.uint8)
    inv = (1 - A01).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num_labels <= 1:
        return 0

    h, w = inv.shape
    border_labels = set(np.unique(np.concatenate([
        labels[0, :], labels[h-1, :], labels[:, 0], labels[:, w-1]
    ])))

    holes = 0
    for lab in range(1, num_labels):
        if lab in border_labels:
            continue
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_hole_area:
            holes += 1

    return holes

def count_endpoints(skel01: np.ndarray, merge_dist: int = 3) -> int:
    """
    Endpoint = skeleton pixel with exactly 1 neighbor (8-connected).
    Nearby endpoints are merged into one.

    merge_dist: distance threshold (in pixels) to merge endpoints
    """

    skel = (skel01 > 0).astype(np.uint8)

    kernel = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ], dtype=np.uint8)

    neigh = cv2.filter2D(skel, -1, kernel)
    endpoints_mask = np.logical_and(skel == 1, neigh == 1)

    # get coordinates
    ys, xs = np.where(endpoints_mask)
    points = list(zip(xs, ys))

    if len(points) == 0:
        return 0

    # -------------------------------------------------
    # cluster nearby endpoints (simple greedy clustering)
    # -------------------------------------------------
    clusters = []

    for p in points:
        px, py = p
        assigned = False

        for cluster in clusters:
            for cx, cy in cluster:
                if (px - cx)**2 + (py - cy)**2 <= merge_dist**2:
                    cluster.append(p)
                    assigned = True
                    break
            if assigned:
                break

        if not assigned:
            clusters.append([p])

    return len(clusters)

def endpoints_xy(skel01: np.ndarray):
    """Return (ys, xs) of endpoints in a skeleton (8-connected)."""
    sk = (skel01 > 0).astype(np.uint8)

    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)

    neigh = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    ep = np.logical_and(sk == 1, neigh == 1)

    ys, xs = np.where(ep)
    return ys, xs


def endpoint_top_bottom_counts(skel01: np.ndarray, margin_frac: float = 0.05):
    """
    Split endpoints by skeleton bbox center (y).
    margin_frac makes a dead-zone around center to avoid borderline jitter.

    Returns:
      e_top, e_bot, e_mid, e_total, y_center
    """
    sk = (skel01 > 0).astype(np.uint8)
    ys_fg, xs_fg = np.where(sk == 1)
    if xs_fg.size == 0:
        return 0, 0, 0, 0, None

    y0, y1 = ys_fg.min(), ys_fg.max() + 1
    h = (y1 - y0)

    y_center = y0 + h / 2.0
    margin = margin_frac * h

    ys_ep, xs_ep = endpoints_xy(sk)
    e_total = int(len(xs_ep))

    e_top = int(np.sum(ys_ep < (y_center - margin)))
    e_bot = int(np.sum(ys_ep > (y_center + margin)))
    e_mid = int(e_total - e_top - e_bot)

    return e_top, e_bot, e_mid, e_total, y_center

def prune_spurs(skel01: np.ndarray, max_length: int = 2) -> np.ndarray:
    """
    Remove small spurs by repeatedly deleting endpoints.
    Use ~3-5 for 120x120. Too high will delete true endpoints.
    """
    skel = (skel01 > 0).astype(np.uint8)

    h, w = skel.shape

    # 8-neighbour offsets
    neighbors = [(-1,-1),(-1,0),(-1,1),
                 (0,-1),        (0,1),
                 (1,-1),(1,0),(1,1)]

    def get_neighbors(y,x):
        pts = []
        for dy,dx in neighbors:
            ny,nx = y+dy, x+dx
            if 0<=ny<h and 0<=nx<w and skel[ny,nx]==1:
                pts.append((ny,nx))
        return pts

    changed = True
    while changed:
        changed = False

        # find endpoints
        endpoints = []
        for y,x in zip(*np.where(skel==1)):
            if len(get_neighbors(y,x)) == 1:
                endpoints.append((y,x))

        for ey,ex in endpoints:
            path = [(ey,ex)]
            visited = {(ey, ex)}
            current = (ey,ex)

            while True:
                neigh = [n for n in get_neighbors(*current) if n not in visited]
                if len(neigh) != 1:
                    break
                current = neigh[0]
                path.append(current)
                visited.add(current)

                if len(path) > max_length:
                    break

            # if branch is short → remove it
            if len(path) <= max_length:
                for py,px in path:
                    skel[py,px] = 0
                changed = True

    return skel


###CRAMM

def hole_count_and_largest_pct(A01: np.ndarray, min_hole_area: int = 30):
    A01 = (A01 > 0).astype(np.uint8)

    fg_area = int(A01.sum())
    if fg_area == 0:
        return 0, 0.0

    inv = (1 - A01).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num <= 1:
        return 0, 0.0

    H, W = inv.shape
    border_labels = set(np.unique(np.concatenate([
        lab[0, :], lab[H-1, :], lab[:, 0], lab[:, W-1]
    ])))

    hole_areas = []
    for k in range(1, num):
        if k in border_labels:
            continue
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area >= min_hole_area:
            hole_areas.append(area)

    hole_count = len(hole_areas)
    largest = max(hole_areas) if hole_areas else 0

    hole_pct_fg = 100.0 * (largest / fg_area)
    return hole_count, hole_pct_fg

def vertical_symmetry_score(A01: np.ndarray):
    """
    Symmetry about vertical axis of the letter's bounding box.
    score ~1 means very symmetric.
    """
    A01 = (A01 > 0).astype(np.uint8)

    # crop to bbox of foreground
    ys, xs = np.where(A01 == 1)
    if xs.size == 0:
        return 0.0
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A01[y0:y1, x0:x1]

    flip = np.fliplr(crop)

    # compare overlap to union (IoU)
    inter = np.logical_and(crop == 1, flip == 1).sum()
    union = np.logical_or(crop == 1, flip == 1).sum()
    return float(inter / union) if union > 0 else 0.0



def concavity_tb_strength(A01: np.ndarray):
    """
    Detect CRAMM-style 'concavity direction' using boundary profiles.

    Returns:
      north_strength: how much the TOP boundary varies across x
                      (large => top has a dip => M-like)
      south_strength: how much the BOTTOM boundary varies across x
                      (large => bottom has a dip => U/V/W-like)

    Works on the binary (not skeleton).
    """
    A = (A01 > 0).astype(np.uint8)

    ys, xs = np.where(A == 1)
    if xs.size == 0:
        return 0.0, 0.0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A[y0:y1, x0:x1]

    h, w = crop.shape

    top_profile = np.full(w, -1, dtype=np.int32)
    bot_profile = np.full(w, -1, dtype=np.int32)

    for x in range(w):
        col = crop[:, x]
        yy = np.where(col == 1)[0]
        if yy.size == 0:
            continue
        top_profile[x] = int(yy[0])
        bot_profile[x] = int(yy[-1])

    # keep only columns that actually contain foreground
    valid = top_profile >= 0
    if valid.sum() < 3:
        return 0.0, 0.0

    top = top_profile[valid]
    bot = bot_profile[valid]

    # "strength" = range of boundary y-values
    north_strength = float(top.max() - top.min())
    south_strength = float(bot.max() - bot.min())

    # normalize by height so thresholds generalize better
    north_strength /= max(1.0, float(h))
    south_strength /= max(1.0, float(h))

    return north_strength, south_strength


def concavity_tb_label(A01: np.ndarray, margin: float = 0.03) -> str:
    """
    Convenience helper:
      returns "NORTH" if top concavity stronger (M-like)
              "SOUTH" if bottom concavity stronger (U/V/W-like)
              "FLAT"  if too close to call
    """
    north, south = concavity_tb_strength(A01)
    if north > south + margin:
        return "NORTH"
    if south > north + margin:
        return "SOUTH"
    return "FLAT"


def concavity_lr_strength(A01: np.ndarray):
    """
    CRAMM-style 'concavity direction' for left/right (west/east)
    using boundary profiles on the binary (not skeleton).

    Returns:
      west_strength: how much the LEFT boundary varies across y
      east_strength: how much the RIGHT boundary varies across y

    Interpretation:
      - large east_strength => right side has a notch / indentation (G-like)
      - small west_strength => left boundary is straight (L-like)
    """
    A = (A01 > 0).astype(np.uint8)

    ys, xs = np.where(A == 1)
    if xs.size == 0:
        return 0.0, 0.0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A[y0:y1, x0:x1]

    h, w = crop.shape

    left_profile  = np.full(h, -1, dtype=np.int32)
    right_profile = np.full(h, -1, dtype=np.int32)

    for y in range(h):
        row = crop[y, :]
        xx = np.where(row == 1)[0]
        if xx.size == 0:
            continue
        left_profile[y]  = int(xx[0])
        right_profile[y] = int(xx[-1])

    valid = left_profile >= 0
    if valid.sum() < 3:
        return 0.0, 0.0

    left = left_profile[valid]
    right = right_profile[valid]

    west_strength = float(left.max() - left.min())
    east_strength = float(right.max() - right.min())

    # normalize by width (x varies)
    west_strength /= max(1.0, float(w))
    east_strength /= max(1.0, float(w))

    return west_strength, east_strength

import numpy as np
import cv2

def count_branchpoints(skel01: np.ndarray) -> int:
    sk = (skel01 > 0).astype(np.uint8)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neigh = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return int(np.logical_and(sk == 1, neigh >= 3).sum())



# to differentiate L and J we look at where the emptiness is 

def side_open_score(A01: np.ndarray, band=(0.2, 0.8)):
    """
    Returns:
        left_fill_ratio
        right_fill_ratio
    Lower fill => more open
    """
    A = (A01 > 0).astype(np.uint8)

    ys, xs = np.where(A == 1)
    if xs.size == 0:
        return 0.0, 0.0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A[y0:y1, x0:x1]
    h, w = crop.shape

    # focus on middle band to avoid top/bottom artifacts
    y_start = int(band[0] * h)
    y_end   = int(band[1] * h)
    mid = crop[y_start:y_end, :]

    left_fill  = mid[:, :w//2].sum()
    right_fill = mid[:, w//2:].sum()

    total = mid.sum() + 1e-6
    return left_fill / total, right_fill / total

## -------------------------
## POINTS DRAWING
## --------------------------

def holes_after_line(A01, p0, p1):
    """
    Draw line on A01 and return hole count after.
    """
    A_line = draw_line(A01, p0, p1)
    return count_holes(A_line)

def line_creates_hole(A01, p0, p1):
    """
    True if drawing the line increases hole count.
    """
    h0 = count_holes(A01)
    h1 = holes_after_line(A01, p0, p1)
    return (h1 > h0), h0, h1

def test_line_TL_TR_creates_hole(A01, split=0.5):
    pts = get_banded_points(A01, split=split)
    if pts is None:
        return False, None
    TL, BL, TR, BR = pts
    ok, h0, h1 = line_creates_hole(A01, TL, TR)
    return ok, (TL, TR, h0, h1)

def test_line_BL_BR_creates_hole(A01, split=0.5):
    pts = get_banded_points(A01, split=split)
    if pts is None:
        return False, None
    TL, BL, TR, BR = pts
    ok, h0, h1 = line_creates_hole(A01, BL, BR)
    return ok, (BL, BR, h0, h1)


import numpy as np

def vertical_symmetry_lr_balance(A01: np.ndarray) -> float:
    """
    CRAMM-like vertical symmetry:
    Compare amount of foreground pixels in left half vs right half
    within the tight bounding box.

    Returns a score in [0, 1]:
      1  = perfectly symmetric (balanced mass)
      0  = totally unbalanced
    """
    A = (A01 > 0).astype(np.uint8)

    ys, xs = np.where(A == 1)
    if xs.size == 0:
        return 0.0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A[y0:y1, x0:x1]

    flip = np.fliplr(crop)
    overlap = np.logical_and(crop == 1, flip == 1).sum()
    union = np.logical_or(crop == 1, flip == 1).sum()

    return float(overlap) / float(union + 1e-6)

import numpy as np

def count_horizontal_bars_context(
    A01: np.ndarray,
    min_frac: float = 0.6,     # how wide the bar must be
    band: int = 3,             # smooth over +-band rows
    context: int = 6,          # how far above/below to check
    bg_frac: float = 0.25,     # "mostly black" threshold relative to width
    contrast: float = 0.30     # bar must exceed context by this fraction of width
) -> int:
    A = (A01 > 0).astype(np.uint8)
    ys, xs = np.where(A == 1)
    if xs.size == 0:
        return 0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A[y0:y1, x0:x1]
    h, w = crop.shape

    row_sum = crop.sum(axis=1).astype(np.float32)  # foreground pixels per row

    # Smooth a bit so the bar isn't split across adjacent rows
    if band > 0:
        k = 2 * band + 1
        kernel = np.ones(k, dtype=np.float32) / k
        row_sm = np.convolve(row_sum, kernel, mode="same")
    else:
        row_sm = row_sum

    bar_thresh = min_frac * w
    bg_thresh = bg_frac * w
    contrast_thresh = contrast * w

    candidates = []
    for y in range(h):
        if row_sm[y] < bar_thresh:
            continue

        ya0 = max(0, y - context)
        ya1 = max(0, y - 1)
        yb0 = min(h, y + 1)
        yb1 = min(h, y + context)

        # if no context on one side, skip (avoid top/bottom edge artifacts)
        if ya1 <= ya0 or yb1 <= yb0:
            continue

        above = float(row_sm[ya0:ya1].mean())
        below = float(row_sm[yb0:yb1].mean())

        # "above and below mostly black" + strong contrast
        if above <= bg_thresh and below <= bg_thresh:
            if (row_sm[y] - max(above, below)) >= contrast_thresh:
                candidates.append(y)

    # Merge nearby candidate rows into bars
    if not candidates:
        return 0
    bars = 1
    for i in range(1, len(candidates)):
        if candidates[i] - candidates[i-1] > 2 * band + 1:
            bars += 1
    return bars

def horizontal_symmetry_tb_balance(A01: np.ndarray) -> float:
    A01 = (A01 > 0).astype(np.uint8)
    ys, xs = np.where(A01 == 1)
    if xs.size == 0:
        return 0.0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A01[y0:y1, x0:x1]

    h, w = crop.shape
    mid = h // 2

    top = crop[:mid, :]
    bottom = crop[h - mid:, :]  # same height as top

    T = int(top.sum())
    B = int(bottom.sum())
    total = T + B
    if total == 0:
        return 0.0

    return 1.0 - (abs(T - B) / total)


def count_runs_1d(arr_1d):
    # count connected runs of 1s
    runs = 0
    in_run = False
    for v in arr_1d:
        if v and not in_run:
            runs += 1
            in_run = True
        elif not v:
            in_run = False
    return runs

def max_run_allow_gaps(col: np.ndarray, gap_allow: int = 2) -> int:
    """
    Longest run of 1s allowing small 0-gaps up to gap_allow.
    """
    best = 0
    run = 0
    gap = 0
    for v in col:
        if v:
            run += 1
            gap = 0
        else:
            if gap < gap_allow:
                run += 1
                gap += 1
            else:
                run = 0
                gap = 0
        if run > best:
            best = run
    return best

import numpy as np

def bottom_width_ratio(A01: np.ndarray, band_frac: float = 0.20) -> float:
    """
    Measures how wide the letter is near the bottom of its bounding box.

    Returns value in [0, 1]:
        1.0  -> bottom band spans full width
        0.0  -> no bottom presence

    band_frac = fraction of bbox height to use as bottom band
    """

    A = (A01 > 0).astype(np.uint8)

    ys, xs = np.where(A == 1)
    if xs.size == 0:
        return 0.0

    # Crop to tight bounding box
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A[y0:y1, x0:x1]

    h, w = crop.shape

    # Bottom band height
    band_h = max(1, int(round(band_frac * h)))

    # Extract bottom band
    bottom_band = crop[h - band_h : h, :]

    # Count how many columns contain at least one foreground pixel
    cols_with_fg = (bottom_band.max(axis=0) > 0).sum()

    # Ratio of active columns to total width
    return float(cols_with_fg) / float(max(1, w))



def close_1d(x: np.ndarray, gap: int = 3) -> np.ndarray:
    """
    Fill small 0-gaps (length <= gap) between 1s.
    Prevents one stroke being split into multiple runs.
    """
    x = x.astype(np.uint8).copy()
    n = len(x)
    i = 0

    while i < n:
        if x[i] == 0:
            j = i
            while j < n and x[j] == 0:
                j += 1

            if (j - i) <= gap:
                left_one = (i - 1 >= 0 and x[i - 1] == 1)
                right_one = (j < n and x[j] == 1)
                if left_one and right_one:
                    x[i:j] = 1

            i = j
        else:
            i += 1

    return x

def vertical_orientation_score(col_01: np.ndarray) -> float:
    """
    col_01 is a 1D binary vector (over y).
    Score high when transitions are few and runs are long (vertical-ish).
    """
    col = (col_01 > 0).astype(np.uint8)
    if col.sum() == 0:
        return 0.0
    # transitions count (how often it turns on/off)
    transitions = np.sum(col[:-1] != col[1:])
    # fewer transitions = more "one continuous vertical stroke"
    return 1.0 / (1.0 + transitions)

def horizontal_orientation_score(row_01: np.ndarray) -> float:
    """
    row_01 is a 1D binary vector (over x).
    Score high when transitions are few and runs are long (horizontal-ish).
    """
    row = (row_01 > 0).astype(np.uint8)
    if row.sum() == 0:
        return 0.0
    transitions = np.sum(row[:-1] != row[1:])
    return 1.0 / (1.0 + transitions)

import numpy as np



def find_runs_1d(x: np.ndarray):
    """Return list of (start,end) inclusive runs where x==1."""
    runs = []
    n = len(x)
    i = 0
    while i < n:
        if x[i] == 1:
            j = i
            while j < n and x[j] == 1:
                j += 1
            runs.append((i, j-1))
            i = j
        else:
            i += 1
    return runs

def count_vertical_strokes(A01: np.ndarray,
                           min_frac=0.6,
                           gap_allow=2,
                           band=5,
                           support_cols=3,
                           orient_thresh=0.12,
                           rel_height_keep=0.80,   # NEW
                           abs_min_len=None        # optional safety floor
                           ) -> int:
    A = (A01 > 0).astype(np.uint8)
    ys, xs = np.where(A == 1)
    if xs.size == 0:
        return 0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A[y0:y1, x0:x1]
    h, w = crop.shape

    min_len = int(min_frac * h)
    if abs_min_len is not None:
        min_len = max(min_len, int(abs_min_len))

    half = band // 2

    best_run = np.zeros(w, dtype=np.int32)
    raw_good = np.zeros(w, dtype=np.uint8)

    for x in range(w):
        xa = max(0, x - half)
        xb = min(w, x + half + 1)
        col = crop[:, xa:xb].max(axis=1)  # band-OR

        best = max_run_allow_gaps(col, gap_allow=gap_allow)
        best_run[x] = best

        if best < min_len:
            continue

        score = vertical_orientation_score(col)
        if score < orient_thresh:
            continue

        raw_good[x] = 1

    # thickness support
    good_cols = np.zeros_like(raw_good)
    r = support_cols // 2
    for x in range(w):
        a = max(0, x - r)
        b = min(w, x + r + 1)
        if raw_good[a:b].sum() >= support_cols:
            good_cols[x] = 1

    good_cols = close_1d(good_cols, gap=3)

    # --- NEW: height-based filtering of clusters ---
    runs = find_runs_1d(good_cols)
    if not runs:
        return 0

    cluster_heights = []
    cluster_widths = []

    for a, b in runs:
        cluster_heights.append(int(best_run[a:b+1].max()))
        cluster_widths.append(int(b - a + 1))

    H_ref = max(cluster_heights)
    W_ref = max(cluster_widths)

    keep = []
    for h_i, w_i in zip(cluster_heights, cluster_widths):

        keep_height = h_i >= rel_height_keep * H_ref

        # NEW: reject strokes that are much thinner than the main one
        keep_width = w_i >= 0.30 * W_ref   # 30% rule

        keep.append(keep_height and keep_width)

    return int(sum(keep))

def x_by_diagonal_angles(sk01: np.ndarray,
                         min_len_frac=0.35,
                         max_gap=3,
                         tilt_deg=30.0,
                         min_diag_lines=2):
    """
    Returns (is_X, debug_dict)

    Detect lines on skeleton using HoughLinesP and check if there are enough
    strongly diagonal lines (tilted > tilt_deg away from vertical).

    tilt away from vertical:
      - vertical is 90deg (or -90) in atan2 convention depending on orientation
      - we compute angle in degrees in [0,180)
      - distance from vertical = |angle - 90|
    """
    sk = (sk01 > 0).astype(np.uint8) * 255
    H, W = sk.shape

    min_len = int(min_len_frac * max(H, W))

    lines = cv2.HoughLinesP(sk, 1, np.pi/180, threshold=20,
                            minLineLength=max(10, min_len),
                            maxLineGap=max_gap)

    if lines is None:
        return False, {"diag": 0, "total": 0, "angles": []}

    angles = []
    diag = 0
    total = 0

    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue

        ang = (np.degrees(np.arctan2(dy, dx)) + 180) % 180  # [0,180)
        angles.append(float(ang))

        # distance from vertical (90 deg)
        dist_from_vert = abs(ang - 90.0)

        # diagonal if far from vertical by > tilt_deg
        if dist_from_vert > tilt_deg and dist_from_vert < (90.0 - 5.0):
            diag += 1
        total += 1

    is_x = diag >= min_diag_lines
    return is_x, {"diag": diag, "total": total, "angles": angles}


def center_density_ratio(A01: np.ndarray, frac=0.25) -> float:
    """
    Ratio of foreground pixels in the central box vs total.
    X tends to be higher (crossing).
    N tends to be lower.
    """
    A = (A01 > 0).astype(np.uint8)
    ys, xs = np.where(A == 1)
    if xs.size == 0:
        return 0.0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A[y0:y1, x0:x1]
    h, w = crop.shape

    cy0 = int((0.5 - frac/2) * h); cy1 = int((0.5 + frac/2) * h)
    cx0 = int((0.5 - frac/2) * w); cx1 = int((0.5 + frac/2) * w)

    total = crop.sum()
    center = crop[cy0:cy1, cx0:cx1].sum()
    return float(center) / (float(total) + 1e-6)

def endpoints_xy(skel01: np.ndarray):
    """
    Return list of (x, y) endpoints in skeleton image.
    Endpoint = pixel with exactly 1 foreground neighbour (8-connectivity).
    """
    sk = (skel01 > 0).astype(np.uint8)
    h, w = sk.shape
    pts = []

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if sk[y, x] == 0:
                continue

            # 8-neighbourhood
            neighbourhood = sk[y-1:y+2, x-1:x+2]
            neighbour_count = int(neighbourhood.sum()) - 1  # subtract self

            if neighbour_count == 1:
                pts.append((x, y))

    return pts

def count_horizontal_strokes(A01: np.ndarray, min_frac=0.6, gap_allow=2, band=5, support_rows=3,
                             orient_thresh=0.12, rel_width_keep=0.80, abs_min_len=None, close_gap=3) -> int:
    A = (A01 > 0).astype(np.uint8)
    ys, xs = np.where(A == 1)
    if xs.size == 0:
        return 0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A[y0:y1, x0:x1]
    h, w = crop.shape

    min_len = int(min_frac * w)
    if abs_min_len is not None:
        min_len = max(min_len, int(abs_min_len))

    half = band // 2
    best_run = np.zeros(h, dtype=np.int32)
    raw_good = np.zeros(h, dtype=np.uint8)

    for y in range(h):
        ya = max(0, y - half)
        yb = min(h, y + half + 1)
        row = crop[ya:yb, :].max(axis=0)

        best = max_run_allow_gaps(row, gap_allow=gap_allow)
        best_run[y] = best
        if best < min_len:
            continue

        score = horizontal_orientation_score(row)
        if score < orient_thresh:
            continue

        raw_good[y] = 1

    good_rows = np.zeros_like(raw_good)
    r = support_rows // 2
    for y in range(h):
        a = max(0, y - r)
        b = min(h, y + r + 1)
        if raw_good[a:b].sum() >= support_rows:
            good_rows[y] = 1

    good_rows = close_1d(good_rows, gap=close_gap)

    runs = find_runs_1d(good_rows)
    if not runs:
        return 0

    cluster_widths = [int(best_run[a:b+1].max()) for (a, b) in runs]
    W_ref = max(cluster_widths)
    keep = [cw >= rel_width_keep * W_ref for cw in cluster_widths]
    return int(sum(keep))

# def count_vertical_strokes(A01: np.ndarray, min_frac=0.6, gap_allow=2, band=5) -> int:
#     A = (A01 > 0).astype(np.uint8)

#     ys, xs = np.where(A == 1)
#     if xs.size == 0:
#         return 0

#     y0, y1 = ys.min(), ys.max() + 1
#     x0, x1 = xs.min(), xs.max() + 1
#     crop = A[y0:y1, x0:x1]

#     h, w = crop.shape
#     min_len = int(min_frac * h)
#     half = band // 2

#     good_cols = np.zeros(w, dtype=np.uint8)
#     for x in range(w):
#         xa = max(0, x - half)
#         xb = min(w, x + half + 1)
#         col = crop[:, xa:xb].max(axis=1)  # OR over band
#         best = max_run_allow_gaps(col, gap_allow=gap_allow)
#         if best >= min_len:
#             good_cols[x] = 1

#     # close small gaps between good columns so one stroke isn't split
#     good_cols = close_1d(good_cols, gap=3)
#     return count_runs_1d(good_cols)



# def count_horizontal_strokes(A01: np.ndarray, min_frac=0.6, gap_allow=2, band=5) -> int:
#     A = (A01 > 0).astype(np.uint8)

#     ys, xs = np.where(A == 1)
#     if xs.size == 0:
#         return 0

#     y0, y1 = ys.min(), ys.max() + 1
#     x0, x1 = xs.min(), xs.max() + 1
#     crop = A[y0:y1, x0:x1]

#     h, w = crop.shape
#     min_len = int(min_frac * w)
#     half = band // 2

#     good_rows = np.zeros(h, dtype=np.uint8)
#     for y in range(h):
#         ya = max(0, y - half)
#         yb = min(h, y + half + 1)
#         row = crop[ya:yb, :].max(axis=0)  # OR over band
#         best = max_run_allow_gaps(row, gap_allow=gap_allow)
#         if best >= min_len:
#             good_rows[y] = 1

#     good_rows = close_1d(good_rows, gap=3)
#     return count_runs_1d(good_rows)


def count_vertical_lines(skel01: np.ndarray, min_frac=0.35, gap_allow=2, band=5) -> int:
    sk = (skel01 > 0).astype(np.uint8)
    ys, xs = np.where(sk == 1)
    if xs.size == 0:
        return 0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = sk[y0:y1, x0:x1]

    h, w = crop.shape
    min_len = int(min_frac * h)

    half = band // 2
    good_cols = np.zeros(w, dtype=np.uint8)

    for x in range(w):
        xa = max(0, x - half)
        xb = min(w, x + half + 1)

        # OR over a small vertical band to tolerate sideways jitter
        band_col = crop[:, xa:xb].max(axis=1)

        best = max_run_allow_gaps(band_col, gap_allow=gap_allow)
        if best >= min_len:
            good_cols[x] = 1

    return count_runs_1d(good_cols)

def count_horizontal_lines(skel01: np.ndarray, min_frac=0.35, gap_allow=2, band=5) -> int:
    sk = (skel01 > 0).astype(np.uint8)
    ys, xs = np.where(sk == 1)
    if xs.size == 0:
        return 0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = sk[y0:y1, x0:x1]

    h, w = crop.shape
    min_len = int(min_frac * w)

    half = band // 2
    good_rows = np.zeros(h, dtype=np.uint8)

    for y in range(h):
        ya = max(0, y - half)
        yb = min(h, y + half + 1)

        # OR over a small horizontal band (tolerate vertical jitter)
        band_row = crop[ya:yb, :].max(axis=0)

        best = max_run_allow_gaps(band_row, gap_allow=gap_allow)
        if best >= min_len:
            good_rows[y] = 1

    return count_runs_1d(good_rows)

# -------------------------
# DEBUG / VISUALIZATION
# -------------------------
def debug_count_holes(A01: np.ndarray, min_hole_area: int = 30):
    """
    Plot hole detection steps + return filtered hole count.
    Filters tiny holes (specks) by area.
    """
    A01 = (A01 > 0).astype(np.uint8)
    inv = (1 - A01).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h, w = inv.shape

    border_labels = set(np.unique(np.concatenate([
        labels[0, :], labels[h-1, :], labels[:, 0], labels[:, w-1]
    ])))

    holes = 0
    hole_mask = np.zeros_like(inv)

    # mark only "real" holes
    for lab in range(1, num_labels):
        if lab in border_labels:
            continue
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_hole_area:
            holes += 1
            hole_mask[labels == lab] = 1

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(A01, cmap='gray')
    plt.title("A01 (foreground=1)")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(inv, cmap='gray')
    plt.title("Inverted (background=1)")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(labels, cmap='nipy_spectral')
    plt.title("Connected Components")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(hole_mask, cmap='gray')
    plt.title(f"Hole mask (count={holes})\nmin_area={min_hole_area}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return holes
import numpy as np
import cv2
import matplotlib.pyplot as plt

def debug_endpoints(skel01: np.ndarray, merge_dist: int = 3):
    """
    Plot endpoints on skeleton after merging nearby ones.
    Shows both raw and clustered endpoints.
    """

    skel = (skel01 > 0).astype(np.uint8)

    kernel = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ], dtype=np.uint8)

    neigh = cv2.filter2D(skel, -1, kernel)
    endpoints_mask = np.logical_and(skel == 1, neigh == 1)

    ys, xs = np.where(endpoints_mask)
    points = list(zip(xs, ys))

    # -------------------------------
    # cluster endpoints
    # -------------------------------
    clusters = []

    for p in points:
        px, py = p
        assigned = False

        for cluster in clusters:
            for cx, cy in cluster:
                if (px - cx)**2 + (py - cy)**2 <= merge_dist**2:
                    cluster.append(p)
                    assigned = True
                    break
            if assigned:
                break

        if not assigned:
            clusters.append([p])

    # cluster centers
    cluster_centers = []
    for cluster in clusters:
        xs_c = [p[0] for p in cluster]
        ys_c = [p[1] for p in cluster]
        cx = int(np.mean(xs_c))
        cy = int(np.mean(ys_c))
        cluster_centers.append((cx, cy))

    # -------------------------------
    # plot
    # -------------------------------
    plt.figure(figsize=(5,5))
    plt.imshow(skel, cmap='gray')

    # raw endpoints (light red)
    if len(points) > 0:
        pxs = [p[0] for p in points]
        pys = [p[1] for p in points]
        plt.scatter(pxs, pys, c='orange', s=10, label='raw')

    # clustered endpoints (final)
    if len(cluster_centers) > 0:
        cxs = [p[0] for p in cluster_centers]
        cys = [p[1] for p in cluster_centers]
        plt.scatter(cxs, cys, c='red', s=25, label='merged')

    plt.title(f"Endpoints: raw={len(points)}, merged={len(cluster_centers)}")
    plt.axis("off")
    plt.legend()
    plt.show()

    return len(cluster_centers)

def debug_line_draw(A01, split=0.5, which="TLTR"):
    """
    Visualize line drawing and hole-change effect.
    which: "TLTR" or "BLBR"
    """
    pts = get_banded_points(A01, split=split)
    if pts is None:
        print("No points found.")
        return

    TL, BL, TR, BR = pts
    if which.upper() == "TLTR":
        p0, p1 = TL, TR
        title = "Line TL→TR"
    else:
        p0, p1 = BL, BR
        title = "Line BL→BR"

    h0 = count_holes(A01)
    A_line = draw_line(A01, p0, p1)
    h1 = count_holes(A_line)

    diff = (A_line.astype(int) - A01.astype(int))

    plt.figure(figsize=(12,3))

    plt.subplot(1,3,1)
    plt.imshow(A01, cmap="gray")
    plt.title(f"Original (holes={h0})")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(A_line, cmap="gray")
    plt.title(f"{title} (holes={h1})")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(diff, cmap="gray")
    plt.title("Added pixels (diff)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"{title}: holes {h0} → {h1} | p0={p0} p1={p1}")


def debug_vertical_symmetry(A01: np.ndarray, title: str = "Vertical symmetry debug"):
    """
    Visualize what vertical_symmetry_score() is doing.

    Shows:
    1) cropped foreground
    2) left-right flipped crop
    3) intersection (overlap)
    4) union
    5) mismatch map

    Returns:
      symmetry score (IoU between crop and its flipped version)
    """
    A01 = (A01 > 0).astype(np.uint8)

    ys, xs = np.where(A01 == 1)
    if xs.size == 0:
        print("debug_vertical_symmetry: empty foreground")
        return 0.0

    # crop to bbox
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A01[y0:y1, x0:x1]
    flip = np.fliplr(crop)

    inter = np.logical_and(crop == 1, flip == 1).astype(np.uint8)
    union = np.logical_or(crop == 1, flip == 1).astype(np.uint8)
    mismatch = (crop != flip).astype(np.uint8)

    inter_sum = int(inter.sum())
    union_sum = int(union.sum())
    score = float(inter_sum / union_sum) if union_sum > 0 else 0.0

    plt.figure(figsize=(12, 3))
    plt.suptitle(f"{title} | score={score:.3f}", y=1.05)

    plt.subplot(1, 5, 1)
    plt.imshow(crop, cmap="gray")
    plt.title("Crop")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(flip, cmap="gray")
    plt.title("Flip (LR)")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(inter, cmap="gray")
    plt.title(f"Intersection\n({inter_sum})")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(union, cmap="gray")
    plt.title(f"Union\n({union_sum})")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(mismatch, cmap="gray")
    plt.title("Mismatch")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return score

def horizontal_symmetry_tb_balance(A01: np.ndarray) -> float:
    """
    CRAMM-like horizontal symmetry:
    Compare foreground mass in top half vs bottom half in bbox crop.
    Returns score in [0,1], 1 = perfectly balanced.
    """
    A01 = (A01 > 0).astype(np.uint8)
    ys, xs = np.where(A01 == 1)
    if xs.size == 0:
        return 0.0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A01[y0:y1, x0:x1]

    h, w = crop.shape
    mid = h // 2
    top = crop[:mid, :]
    bot = crop[h - mid:, :]   # same height as top

    T = int(top.sum())
    B = int(bot.sum())
    total = T + B
    if total == 0:
        return 0.0

    return 1.0 - (abs(T - B) / total)

#

def debug_vertical_symmetry_lr_balance(A01: np.ndarray):
    A01 = (A01 > 0).astype(np.uint8)

    ys, xs = np.where(A01 == 1)
    if xs.size == 0:
        print("Empty foreground.")
        return

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A01[y0:y1, x0:x1]

    h, w = crop.shape
    mid = w // 2
    left = crop[:, :mid]
    right = crop[:, w - mid:]

    L = int(left.sum())
    R = int(right.sum())
    score = 1.0 - (abs(L - R) / max(1, (L + R)))

    # show a vertical split line on the crop
    crop_vis = crop.copy()
    if mid < w:
        crop_vis[:, mid-1:mid+1] = 1  # white divider

    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.imshow(crop_vis, cmap="gray")
    plt.title("Crop + midline")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(left, cmap="gray")
    plt.title(f"Left sum={L}")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(right, cmap="gray")
    plt.title(f"Right sum={R}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"LR-balance symmetry score: {score:.3f}")
    return score


def debug_vertical_lines(skel01, min_frac=0.35, gap_allow=2, band=5):
    sk = (skel01 > 0).astype(np.uint8)

    ys, xs = np.where(sk == 1)
    if xs.size == 0:
        print("Empty skeleton.")
        return 0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = sk[y0:y1, x0:x1]

    h, w = crop.shape
    min_len = int(min_frac * h)

    half = band // 2
    good_cols = np.zeros(w, dtype=np.uint8)

    for x in range(w):
        xa = max(0, x - half)
        xb = min(w, x + half + 1)

        band_col = crop[:, xa:xb].max(axis=1)
        best = max_run_allow_gaps(band_col, gap_allow=gap_allow)

        if best >= min_len:
            good_cols[x] = 1

    vline_count = count_runs_1d(good_cols)

    # Visualize
    vis = np.dstack([crop * 255] * 3)
    for x in range(w):
        if good_cols[x] == 1:
            vis[:, x, 0] = 255
            vis[:, x, 1] = 0
            vis[:, x, 2] = 0

    plt.figure(figsize=(4,4))
    plt.imshow(vis)
    plt.title(f"Vertical lines: {vline_count}")
    plt.axis("off")
    plt.show()

    return vline_count

def debug_horizontal_lines(skel01: np.ndarray, min_frac=0.35, gap_allow=2, band=5):
    sk = (skel01 > 0).astype(np.uint8)
    ys, xs = np.where(sk == 1)
    if xs.size == 0:
        print("Empty skeleton.")
        return 0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = sk[y0:y1, x0:x1]

    h, w = crop.shape
    min_len = int(min_frac * w)

    half = band // 2
    good_rows = np.zeros(h, dtype=np.uint8)

    for y in range(h):
        ya = max(0, y - half)
        yb = min(h, y + half + 1)
        band_row = crop[ya:yb, :].max(axis=0)

        best = max_run_allow_gaps(band_row, gap_allow=gap_allow)
        if best >= min_len:
            good_rows[y] = 1

    hline_count = count_runs_1d(good_rows)

    vis = np.dstack([crop * 255] * 3)
    for y in range(h):
        if good_rows[y] == 1:
            vis[y, :, 0] = 0
            vis[y, :, 1] = 255
            vis[y, :, 2] = 0

    plt.figure(figsize=(4,4))
    plt.imshow(vis)
    plt.title(f"Horizontal lines: {hline_count}")
    plt.axis("off")
    plt.show()

    return hline_count
    

import numpy as np
import matplotlib.pyplot as plt

def debug_vertical_strokes(A01: np.ndarray,
    min_frac=0.6,
    gap_allow=2,
    band=5,
    support_cols=3,
    orient_thresh=0.12,
    rel_height_keep=0.80,
    rel_width_keep=0.30,
    abs_min_len=None,
    close_gap=3
) -> int:
    """
    Debug for count_vertical_strokes():
    Visualizes:
      - raw_good (passes len+orient)  = yellow
      - good_cols after support+close = orange
      - kept clusters (final counted) = red
      - rejected clusters             = gray
    Prints cluster heights + widths + keep flags.
    Returns the SAME count as count_vertical_strokes().
    """
    A = (A01 > 0).astype(np.uint8)
    ys, xs = np.where(A == 1)
    if xs.size == 0:
        print("Empty foreground.")
        return 0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A[y0:y1, x0:x1]
    h, w = crop.shape

    min_len = int(min_frac * h)
    if abs_min_len is not None:
        min_len = max(min_len, int(abs_min_len))

    half = band // 2

    best_run = np.zeros(w, dtype=np.int32)
    raw_good = np.zeros(w, dtype=np.uint8)

    # 1) raw filter (len + orientation)
    for x in range(w):
        xa = max(0, x - half)
        xb = min(w, x + half + 1)
        col = crop[:, xa:xb].max(axis=1)

        best = max_run_allow_gaps(col, gap_allow=gap_allow)
        best_run[x] = best
        if best < min_len:
            continue

        score = vertical_orientation_score(col)
        if score < orient_thresh:
            continue

        raw_good[x] = 1

    # 2) thickness support
    good_cols = np.zeros_like(raw_good)
    r = support_cols // 2
    for x in range(w):
        a = max(0, x - r)
        b = min(w, x + r + 1)
        if raw_good[a:b].sum() >= support_cols:
            good_cols[x] = 1

    # 3) close gaps
    good_cols_closed = close_1d(good_cols, gap=close_gap)

    # 4) cluster finding + filtering
    runs = find_runs_1d(good_cols_closed)
    if not runs:
        vis = np.dstack([crop * 255] * 3).astype(np.uint8)
        plt.figure(figsize=(5, 5))
        plt.imshow(vis)
        plt.title("Vertical strokes=0 (no runs)")
        plt.axis("off")
        plt.show()
        print("No vertical stroke clusters found.")
        return 0

    cluster_heights = []
    cluster_widths = []

    for a, b in runs:
        cluster_heights.append(int(best_run[a:b+1].max()))
        cluster_widths.append(int(b - a + 1))

    H_ref = max(cluster_heights)
    W_ref = max(cluster_widths)

    keep = []
    for h_i, w_i in zip(cluster_heights, cluster_widths):
        keep_height = h_i >= rel_height_keep * H_ref
        keep_width = w_i >= rel_width_keep * W_ref
        keep.append(keep_height and keep_width)

    kept_mask = np.zeros_like(good_cols_closed, dtype=np.uint8)
    rej_mask  = np.zeros_like(good_cols_closed, dtype=np.uint8)

    for (a, b), k in zip(runs, keep):
        if k:
            kept_mask[a:b+1] = 1
        else:
            rej_mask[a:b+1] = 1

    final_count = int(sum(keep))

    # --------- VISUAL OVERLAY ----------
    vis = np.dstack([crop * 255] * 3).astype(np.uint8)

    # raw_good = yellow
    for x in range(w):
        if raw_good[x]:
            vis[:, x] = [255, 255, 0]

    # support + close = orange
    for x in range(w):
        if good_cols_closed[x]:
            vis[:, x] = [255, 165, 0]

    # rejected = gray
    for x in range(w):
        if rej_mask[x]:
            vis[:, x] = [120, 120, 120]

    # kept = red
    for x in range(w):
        if kept_mask[x]:
            vis[:, x] = [255, 0, 0]

    plt.figure(figsize=(5, 5))
    plt.imshow(vis)
    plt.title(
        f"Vertical strokes={final_count}\n"
        f"min_len={min_len} band={band} gap_allow={gap_allow} close_gap={close_gap}\n"
        f"orient_thresh={orient_thresh} support_cols={support_cols}\n"
        f"H_ref={H_ref} rel_h={rel_height_keep} | W_ref={W_ref} rel_w={rel_width_keep} | runs={len(runs)}"
    )
    plt.axis("off")
    plt.show()

    # --------- PRINT CLUSTER DETAILS ----------
    print("Clusters (x_start..x_end): width, height, keep_height, keep_width, kept?")
    for i, ((a, b), hh, ww, kk) in enumerate(zip(runs, cluster_heights, cluster_widths, keep)):
        kh = hh >= rel_height_keep * H_ref
        kw = ww >= rel_width_keep * W_ref
        print(
            f"  #{i}: [{a:3d}..{b:3d}] "
            f"width={ww:3d} height={hh:3d} "
            f"keep_h={kh} keep_w={kw} keep={kk}"
        )

    top_idx = np.argsort(-best_run)[:8]
    print("Top columns by best_run:")
    for i in top_idx:
        if best_run[i] == 0:
            continue
        xa = max(0, i - half)
        xb = min(w, i + half + 1)
        col = crop[:, xa:xb].max(axis=1)
        sc = vertical_orientation_score(col)
        print(
            f"  x={int(i):3d} best={int(best_run[i]):3d} "
            f"raw={int(raw_good[i])} pre={int(good_cols_closed[i])} score={sc:.3f}"
        )

    return final_count

def debug_horizontal_strokes(
    A01: np.ndarray,
    min_frac=0.6,
    gap_allow=2,
    band=5,
    support_rows=3,
    orient_thresh=0.12,
    abs_min_len=None,
    close_gap=3,
    rel_width_keep=0.80
) -> int:
    A = (A01 > 0).astype(np.uint8)
    ys, xs = np.where(A == 1)
    if xs.size == 0:
        print("Empty foreground.")
        return 0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop0 = A[y0:y1, x0:x1]
    h, w = crop0.shape
    crop = crop0.copy()

    min_len = int(min_frac * w)
    if abs_min_len is not None:
        min_len = max(min_len, int(abs_min_len))

    half = band // 2
    raw_good = np.zeros(h, dtype=np.uint8)
    best_run = np.zeros(h, dtype=np.int32)
    score_at = np.zeros(h, dtype=np.float32)

    # 1) raw filter
    for y in range(h):
        ya = max(0, y - half)
        yb = min(h, y + half + 1)
        row = crop[ya:yb, :].max(axis=0)

        best = max_run_allow_gaps(row, gap_allow=gap_allow)
        best_run[y] = best
        if best < min_len:
            continue

        sc = horizontal_orientation_score(row)
        score_at[y] = sc
        if sc < orient_thresh:
            continue

        raw_good[y] = 1

    # 2) support
    good_rows = np.zeros_like(raw_good)
    r = support_rows // 2
    for y in range(h):
        a = max(0, y - r)
        b = min(h, y + r + 1)
        if raw_good[a:b].sum() >= support_rows:
            good_rows[y] = 1

    # 3) close
    good_rows_closed = close_1d(good_rows, gap=close_gap)

    runs = find_runs_1d(good_rows_closed)
    if not runs:
        vis = np.dstack([crop0 * 255] * 3).astype(np.uint8)
        plt.figure(figsize=(5, 5))
        plt.imshow(vis)
        plt.title("Horizontal strokes=0 (no runs)")
        plt.axis("off")
        plt.show()
        print("No horizontal stroke clusters found.")
        return 0

    cluster_widths = [int(best_run[a:b+1].max()) for (a, b) in runs]
    W_ref = max(cluster_widths)
    keep_mask = [cw >= rel_width_keep * W_ref for cw in cluster_widths]
    final_count = int(sum(keep_mask))

    kept_rows = np.zeros_like(good_rows_closed, dtype=np.uint8)
    rej_rows = np.zeros_like(good_rows_closed, dtype=np.uint8)

    for (a, b), k in zip(runs, keep_mask):
        if k:
            kept_rows[a:b+1] = 1
        else:
            rej_rows[a:b+1] = 1

    vis = np.dstack([crop0 * 255] * 3).astype(np.uint8)

    # raw rows = yellow
    for y in range(h):
        if raw_good[y]:
            vis[y, :] = [255, 255, 0]

    # support+close rows = orange
    for y in range(h):
        if good_rows_closed[y]:
            vis[y, :] = [255, 165, 0]

    # rejected rows = gray
    for y in range(h):
        if rej_rows[y]:
            vis[y, :] = [120, 120, 120]

    # kept rows = red
    for y in range(h):
        if kept_rows[y]:
            vis[y, :] = [255, 0, 0]

    plt.figure(figsize=(5, 5))
    plt.imshow(vis)
    plt.title(
        f"Horizontal strokes={final_count} runs={len(runs)}\n"
        f"min_len={min_len} band={band} gap_allow={gap_allow} close_gap={close_gap}\n"
        f"orient_thresh={orient_thresh} support_rows={support_rows}\n"
        f"W_ref={W_ref} rel_width_keep={rel_width_keep}"
    )
    plt.axis("off")
    plt.show()

    print("Runs (y_start..y_end): best_width, mean_score, kept?")
    for i, (a, b) in enumerate(runs):
        bw = int(best_run[a:b+1].max())
        ms = float(score_at[a:b+1].mean())
        kept = keep_mask[i]
        print(
            f"  #{i}: [{a:3d}..{b:3d}] "
            f"best_width={bw:3d} mean_score={ms:.3f} kept={kept}"
        )

    return final_count

def debug_horizontal_symmetry_tb_balance(A01: np.ndarray):
    A01 = (A01 > 0).astype(np.uint8)
    ys, xs = np.where(A01 == 1)
    if xs.size == 0:
        print("Empty foreground.")
        return 0.0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A01[y0:y1, x0:x1]

    h, w = crop.shape
    mid = h // 2
    top = crop[:mid, :]
    bot = crop[h - mid:, :]

    T = int(top.sum())
    B = int(bot.sum())
    score = 1.0 - (abs(T - B) / max(1, (T + B)))

    crop_vis = crop.copy()
    if mid < h:
        crop_vis[mid-1:mid+1, :] = 1  # horizontal divider

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1); plt.imshow(crop_vis, cmap="gray"); plt.title("Crop + midline"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(top, cmap="gray"); plt.title(f"Top sum={T}"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(bot, cmap="gray"); plt.title(f"Bottom sum={B}"); plt.axis("off")
    plt.tight_layout(); plt.show()

    print(f"TB-balance symmetry score: {score:.3f}")
    return score


def debug_hole_bbox_percentage(A01: np.ndarray):
    """
    Visualize:
      - tight bounding box
      - hole mask
      - hole % relative to bbox
    """

    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    A01 = (A01 > 0).astype(np.uint8)

    # --- find bbox of foreground ---
    ys, xs = np.where(A01 == 1)
    if xs.size == 0:
        print("Empty foreground.")
        return

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    bbox_area = (y1 - y0) * (x1 - x0)

    # --- find holes ---
    inv = (1 - A01).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    H, W = inv.shape
    border_labels = set(np.unique(np.concatenate([
        labels[0, :], labels[H-1, :],
        labels[:, 0], labels[:, W-1]
    ])))

    hole_mask = np.zeros_like(A01)
    hole_areas = []

    for k in range(1, num):
        if k not in border_labels:
            hole_mask[labels == k] = 1
            hole_areas.append(int(stats[k, cv2.CC_STAT_AREA]))

    hole_count = len(hole_areas)
    largest_hole = max(hole_areas) if hole_areas else 0

    hole_pct_bbox = 100.0 * largest_hole / max(1, bbox_area)

    fg_area = int(A01.sum())
    hole_pct_fg = 100.0 * largest_hole / max(1, fg_area)

    # --- visualisation ---
    vis = np.dstack([A01*255]*3)

    # draw bbox in red
    vis[y0:y1, x0] = [255, 0, 0]
    vis[y0:y1, x1-1] = [255, 0, 0]
    vis[y0, x0:x1] = [255, 0, 0]
    vis[y1-1, x0:x1] = [255, 0, 0]

    # highlight hole in green
    vis[hole_mask == 1] = [0, 255, 0]

    plt.figure(figsize=(5,5))
    plt.imshow(vis)
    plt.title("BBox (red) + Hole (green)")
    plt.axis("off")
    plt.show()

    print("Hole count:", hole_count)
    print("Largest hole area:", largest_hole)
    print("BBox area:", bbox_area)
    print("Foreground area:", fg_area)
    print("Hole % of BBox:", round(hole_pct_bbox, 2))
    print("Hole % of Foreground:", round(hole_pct_fg, 2))

    return hole_pct_bbox


def debug_concavity_tb(A01: np.ndarray):
    A = (A01 > 0).astype(np.uint8)

    ys, xs = np.where(A == 1)
    if xs.size == 0:
        print("Empty foreground.")
        return

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A[y0:y1, x0:x1]
    h, w = crop.shape

    top_profile = np.full(w, -1, dtype=np.int32)
    bot_profile = np.full(w, -1, dtype=np.int32)

    for x in range(w):
        col = crop[:, x]
        yy = np.where(col == 1)[0]
        if yy.size == 0:
            continue
        top_profile[x] = int(yy[0])
        bot_profile[x] = int(yy[-1])

    valid = top_profile >= 0
    top = top_profile.copy()
    bot = bot_profile.copy()

    north, south = concavity_tb_strength(A01)
    label = concavity_tb_label(A01)

    plt.figure(figsize=(5,5))
    plt.imshow(crop, cmap="gray")
    plt.title(f"Concavity={label} | north={north:.3f}    south={south:.3f}")
    xs_plot = np.arange(w)

    # plot boundary curves (note: y axis is downward in images)
    plt.plot(xs_plot[valid], top[valid], linewidth=2)  # top boundary
    plt.plot(xs_plot[valid], bot[valid], linewidth=2)  # bottom boundary
    plt.gca()  # makes plot feel like normal y-up
    plt.axis("off")
    plt.show()

def debug_bottom_width_ratio(A01: np.ndarray, band_frac: float = 0.20):
    """
    Visual debug for bottom_width_ratio().
    Shows:
      - bbox crop
      - highlighted bottom band
      - columns counted
      - printed ratio
    """

    A = (A01 > 0).astype(np.uint8)

    ys, xs = np.where(A == 1)
    if xs.size == 0:
        print("Empty foreground.")
        return 0.0

    # Crop to tight bbox
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = A[y0:y1, x0:x1]

    h, w = crop.shape

    # Bottom band
    band_h = max(1, int(round(band_frac * h)))
    y_start = h - band_h
    bottom_band = crop[y_start:h, :]

    # Which columns have foreground in bottom band?
    cols_with_fg = (bottom_band.max(axis=0) > 0)
    ratio = float(cols_with_fg.sum()) / float(max(1, w))

    # ----- Visualization -----
    vis = np.dstack([crop * 255] * 3).astype(np.uint8)

    # Highlight bottom band (light blue overlay)
    vis[y_start:h, :, 1] = 150  # green tint
    vis[y_start:h, :, 2] = 255  # blue tint

    # Highlight counted columns in red
    for x in range(w):
        if cols_with_fg[x]:
            vis[:, x, 0] = 255  # red

    plt.figure(figsize=(5,5))
    plt.imshow(vis)
    plt.title(f"Bottom Width Ratio = {ratio:.3f}  |  band_frac={band_frac}")
    plt.axis("off")
    plt.show()

    print("Band height:", band_h)
    print("Columns counted:", int(cols_with_fg.sum()), "/", w)
    print("Bottom width ratio:", round(ratio, 3))

    return ratio


def debug_branchpoints(skel01: np.ndarray):
    """
    Visualise branchpoints (>=3 neighbors) on skeleton.
    Red  = branchpoints
    Blue = endpoints
    """

    sk = (skel01 > 0).astype(np.uint8)

    # 8-neighbour count (excluding center)
    kernel = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ], dtype=np.uint8)

    neigh = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    branch = np.logical_and(sk == 1, neigh >= 3)
    endpoints = np.logical_and(sk == 1, neigh == 1)

    bp_count = int(branch.sum())
    ep_count = int(endpoints.sum())

    print(f"Branchpoints: {bp_count}")
    print(f"Endpoints: {ep_count}")

    # Create RGB visual
    vis = np.dstack([sk * 255] * 3).astype(np.uint8)

    # Branchpoints = red
    vis[branch] = [255, 0, 0]

    # Endpoints = blue
    vis[endpoints] = [0, 0, 255]

    plt.figure(figsize=(5,5))
    plt.imshow(vis)
    plt.title(f"Branchpoints={bp_count}  Endpoints={ep_count}")
    plt.axis("off")
    plt.show()

    return bp_count, ep_count


def debug_misclassified_sample(A01, skel01,
                                true_label, pred_label,
                                hole_count, hole_pct,
                                endpoints, sym_score,
                                vlines, hlines,
                                occ=None):
    """
    Run ALL debug visualisations when a sample is misclassified.
    """

    print("\n" + "="*70)
    print("MISCLASSIFIED SAMPLE")
    if occ is not None:
        print(f"Occurrence: {occ}")
    print(f"True: {true_label} | Predicted: {pred_label}")
    print(f"Holes: {hole_count} | Hole%: {hole_pct:.2f}")
    print(f"Endpoints: {endpoints}")
    print(f"Symmetry: {sym_score:.3f}")
    print(f"Vertical lines: {vlines}")
    print(f"Horizontal lines: {hlines}")
    print("="*70)



    # 1) Holes debug
    debug_count_holes(A01)

    # 2) Endpoints debug
    debug_endpoints(skel01)

    # 3) Vertical line detection debug
    debug_vertical_lines(skel01)

    # 4) Horizontal line detection debug
    debug_horizontal_lines(skel01)

    # 5) Symmetry debug
    debug_vertical_symmetry_lr_balance(A01)

    # 6) Optional full IoU symmetry
    debug_vertical_symmetry(A01)

    debug_horizontal_symmetry_tb_balance(A01)
