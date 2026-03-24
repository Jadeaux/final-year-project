import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.preprocess_for_segmented import preprocess_letters as preprocess_segmented
from modules.preprocessing import thin
from modules.classification_letters import classify_letter

from modules.features_letters import (
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
    debug_bottom_width_ratio,
    side_open_score,
)

ROOT = r"E:\EnglishImg (1)\English\Img\GoodImg\Bmp"
TARGET_FOLDERS = ["Sample011"]   # A, B, C


def list_images(folder, exts=("png", "jpg", "jpeg", "bmp")):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(folder, f"*.{e}"))
    return sorted(paths)


def sample_folder_to_letter(folder_name):
    idx = int(folder_name.replace("Sample", ""))
    if 11 <= idx <= 36:
        return chr(ord("A") + (idx - 11))
    return None


for folder in TARGET_FOLDERS:
    folder_path = os.path.join(ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    true_letter = sample_folder_to_letter(folder)
    images = list_images(folder_path)

    print("\n==============================")
    print("Inspecting letter:", true_letter)
    print("Folder:", folder)
    print("Num images:", len(images))
    print("==============================")

    for img_path in images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # -----------------------
        # LETTER PREPROCESSING
        # -----------------------
        A, cleaned, binary_norm = preprocess_segmented(img, visualize=False, plate_mode=True)
        A = (A > 0).astype(np.uint8)

        # optional: keep largest CC if neede

        skel = thin(A)
        sk_pruned = prune_spurs(skel, max_length=2)
        e = count_endpoints(sk_pruned)

        # -----------------------
        # LETTER CLASSIFICATION
        # -----------------------
        pred = classify_letter(A, skel)
        pred = pred if pred is not None else "?"

        # -----------------------
        # FEATURE VALUES
        # -----------------------

        hc, hpct = hole_count_and_largest_pct(A)
        # only show A samples where 0 blobs/holes were detected
        if true_letter == "A" and hc != 0:
            continue
        vs = vertical_symmetry_lr_balance(A)
        hs = horizontal_symmetry_tb_balance(A)

        vstrokes = count_vertical_strokes(A, min_frac=0.6)
        hstrokes = count_horizontal_strokes(
            A,
            min_frac=0.8,
            gap_allow=2,
            band=5,
            support_rows=3,
            orient_thresh=0.12,
            rel_width_keep=0.80
        )

        north, south = concavity_tb_strength(A)
        west, east = concavity_lr_strength(A)

        bw = bottom_width_ratio(A, band_frac=0.20)
        ratio = center_density_ratio(A, frac=0.35)
        left_ratio, right_ratio = side_open_score(A)

        # -----------------------
        # PRINTS
        # -----------------------
        print("\n--------------------------------")
        print("File:", os.path.basename(img_path))
        print("True LETTER:", true_letter)
        print("Predicted LETTER:", pred)
        print("Ratios (left, right):", round(left_ratio, 3), round(right_ratio, 3))
        print("Center density ratio:", round(ratio, 3))
        print("Bottom width ratio:", round(bw, 3))
        print("Endpoints(after prune):", e)
        print("Vertical symmetry score:", round(vs, 3))
        print("Horizontal symmetry score:", round(hs, 3))
        print("vstrokes:", vstrokes, "| hstrokes:", hstrokes)
        print("hole count:", hc, "| hole %:", round(hpct, 2))
        print(f"north={north:.3f} | south={south:.3f}")
        print(f"west={west:.3f} | east={east:.3f}")

        # -----------------------
        # MAIN PANELS
        # -----------------------
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Raw")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(binary_norm, cmap="gray")
        plt.title("Binary")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(A * 255, cmap="gray")
        plt.title("Final A")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(sk_pruned * 255, cmap="gray")
        plt.title(f"Pruned skeleton\ne={e}")
        plt.axis("off")

        plt.suptitle(
            f"True={true_letter} | Pred={pred}\n{os.path.basename(img_path)}",
            fontsize=12
        )
        plt.tight_layout()
        plt.show()

        # -----------------------
        # EXTRA DEBUG VISUALS
        # -----------------------
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

        debug_horizontal_strokes(
            A,
            min_frac=0.8,
            gap_allow=2,
            band=5,
            support_rows=3,
            orient_thresh=0.12,
            rel_width_keep=0.80
        )

        # optional if useful
        # debug_bottom_width_ratio(A, band_frac=0.20)

        input("Press Enter for next image...")
        plt.close("all")