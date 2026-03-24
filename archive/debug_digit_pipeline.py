import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

from modules.preprocessing import preprocess_step1, thin
from modules.morphology import find_blobs
from modules.features import get_stems, get_banded_points, draw_line
from modules.classification import classify_with_blobs_from_A


def _imshow(ax, img, title):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def _to255(A01):
    return (A01.astype(np.uint8) * 255)

def resize_to_height(gray, target_h=240):
    h, w = gray.shape[:2]
    if h == target_h:
        return gray
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_AREA)


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

def show_digit_debug(img_path, min_blob_area_thick=30, min_blob_area_thin=10, split=0.5):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)

    # --- preprocess ---
    #   # --- preprocess (match run file) ---
    img = resize_to_height(img, target_h=240)

    A, cropped_vis, binary_norm = preprocess_step1(img, visualize=False)

    A255 = fill_small_holes((A > 0).astype(np.uint8) * 255, max_hole_frac=0.02)
    A01 = (A255 > 0).astype(np.uint8)

    A_thin = thin(A01)

    # --- group decision (as in your plate classifier: thick blobs) ---
    blobs_thick, nb_thick = find_blobs(A01, min_blob_area=min_blob_area_thick)
    blobs_thin,  nb_thin  = find_blobs(A_thin, min_blob_area=min_blob_area_thin)

    pred, group = classify_with_blobs_from_A(A01, visualize=False, debug=False)

    print("\n====================")
    print("FILE:", os.path.basename(img_path))
    print("pred:", pred, "| group:", group)
    print("thick blobs:", nb_thick, "| thin blobs:", nb_thin)
    print("====================")

    # --- base figures ---
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))

    _imshow(axs[0,0], img, "raw gray")
    _imshow(axs[0,1], binary_norm, "binary_norm (otsu/polarity)")
    _imshow(axs[0,2], cropped_vis, "cropped_vis (0/255)")
    _imshow(axs[0,3], _to255(A01), "A (0/1) thick")

    _imshow(axs[1,0], _to255(A_thin), "A_thin skeleton")
    _imshow(axs[1,1], _to255(blobs_thick), f"blobs THICK (n={nb_thick})")
    _imshow(axs[1,2], _to255(blobs_thin),  f"blobs THIN (n={nb_thin})")
    axs[1,3].axis("off")

    plt.tight_layout()
    plt.show()

    # --- Group 1 stem debug ---
    if nb_thick > 0:
        # stems computed on thin+thin-blobs (as you do)
        blobs_for_stems, _ = find_blobs(A_thin, min_blob_area=min_blob_area_thin)
        stems_img, n_stems, cents = get_stems(A_thin, blobs_for_stems)

        print(f"[Group1] stems detected: {n_stems} | cents={cents[:2] if cents else cents}")

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        _imshow(axs[0], _to255(A_thin), "A_thin")
        _imshow(axs[1], _to255(blobs_for_stems), "blobs_for_stems (thin)")
        _imshow(axs[2], _to255(stems_img), f"stems_img (n={n_stems})")
        plt.tight_layout()
        plt.show()
        return

    # --- Group 2 helper-line debug ---
    pts = get_banded_points(A_thin, split=split)
    if pts is None:
        print("[Group2] get_banded_points returned None")
        return

    TL, BL, TR, BR = pts
    print("[Group2] TL,BL,TR,BR:", TL, BL, TR, BR)

    # TL->BL
    A1 = draw_line(A_thin, TL, BL)
    blobs1, nb1 = find_blobs(A1, min_blob_area=min_blob_area_thin)
    stems1, n_stems1, cents1 = get_stems(A1, blobs1)

    # TL->BR
    A2 = draw_line(A_thin, TL, BR)
    blobs2, nb2 = find_blobs(A2, min_blob_area=min_blob_area_thin)

    # TR->BR
    A3 = draw_line(A_thin, TR, BR)
    blobs3, nb3 = find_blobs(A3, min_blob_area=min_blob_area_thin)

    print(f"[Group2] nb1(TL->BL)={nb1} | stems_after_TLBL={n_stems1} | nb2(TL->BR)={nb2} | nb3(TR->BR)={nb3}")

    # Visualize helper-line stages (+ points)
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))

    # show points overlay
    base = _to255(A_thin).copy()
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for (x,y), c in [(TL,(0,0,255)), (BL,(0,255,0)), (TR,(255,0,0)), (BR,(255,255,0))]:
        cv2.circle(base_rgb, (int(x),int(y)), 2, c, -1)

    axs[0,0].imshow(base_rgb)
    axs[0,0].set_title("A_thin + TL/BL/TR/BR")
    axs[0,0].axis("off")

    _imshow(axs[0,1], _to255(A1), "A1 after TL→BL")
    _imshow(axs[0,2], _to255(blobs1), f"blobs1 (n={nb1})")
    _imshow(axs[0,3], _to255(stems1), f"stems after TL→BL (n={n_stems1})")

    _imshow(axs[1,0], _to255(A2), "A2 after TL→BR")
    _imshow(axs[1,1], _to255(blobs2), f"blobs2 (n={nb2})")
    _imshow(axs[1,2], _to255(A3), "A3 after TR→BR")
    _imshow(axs[1,3], _to255(blobs3), f"blobs3 (n={nb3})")

    plt.tight_layout()
    plt.show()


def sample_and_debug(folder, k=5, seed=0):
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not paths:
        raise FileNotFoundError(folder)

    random.seed(seed)
    picks = random.sample(paths, min(k, len(paths)))
    for p in picks:
        show_digit_debug(p)


if __name__ == "__main__":
    ROOT = r"C:\Users\jadsa\GitRepos\final-year-project-jad\images\variations_numbers"

    # 1) Debug ONE specific image:
    # show_digit_debug(os.path.join(ROOT, "number7", "number7_003.png"))

    # 2) Or debug a few random samples from a digit folder:
    sample_and_debug(os.path.join(ROOT, "number7"), k=20, seed=1)