import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.preprocessing_chars import preprocess_step1, thin
from modules.classification import classify_with_blobs_from_A
from modules.morphology_chars import find_blobs
from modules.features import get_stems, get_banded_points, draw_line

ROOT = r"E:\EnglishImg (1)\English\Img\GoodImg\Bmp"
TARGET_DIGITS = [7]


def list_images(folder, exts=("png", "jpg", "jpeg", "bmp")):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(folder, f"*.{e}"))
    return sorted(paths)


def get_digit_from_folder(folder_name):
    idx = int(folder_name.replace("Sample", ""))
    if 1 <= idx <= 10:
        return idx - 1
    return None


def count_endpoints_local(skel01: np.ndarray) -> int:
    sk = (skel01 > 0).astype(np.uint8)
    if sk.sum() == 0:
        return 0

    K = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]], dtype=np.uint8)

    nb = cv2.filter2D(sk, -1, K, borderType=cv2.BORDER_CONSTANT)
    endpoints = ((sk == 1) & (nb == 1)).sum()
    return int(endpoints)


for folder in sorted(os.listdir(ROOT)):
    digit = get_digit_from_folder(folder)

    if digit not in TARGET_DIGITS:
        continue

    folder_path = os.path.join(ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    images = list_images(folder_path)

    print("\n==============================")
    print("Inspecting digit:", digit)
    print("Folder:", folder)
    print("Num images:", len(images))
    print("==============================")

    for img_path in images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # same preprocessing as your working digit run
        A, cleaned, binary_norm = preprocess_step1(img, visualize=False)
        A01 = (A > 0).astype(np.uint8)

        # keep largest connected component
        num, labels, stats, _ = cv2.connectedComponentsWithStats(A01, connectivity=8)
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = 1 + int(np.argmax(areas))
            A01 = (labels == largest).astype(np.uint8)

        skel = thin(A01)

        # main stats
        blobs_thick, n_blobs_thick = find_blobs(A01)
        blobs_thin, n_blobs_thin = find_blobs(skel)

        # leftover stems after blob detection on original skeleton
        stems_after_blob_img, n_stems_after_blob, _ = get_stems(skel, blobs_thin)

        endpoints = count_endpoints_local(skel)

        pred = classify_with_blobs_from_A(A01, visualize=False, debug=False)

        pred_digit = pred
        pred_group = None
        if isinstance(pred, tuple):
            pred_digit = pred[0]
            pred_group = pred[1] if len(pred) > 1 else None

        # -----------------------------
        # helper-line debug
        # -----------------------------
        pts = get_banded_points(skel, split=0.5)

        A1 = A2 = A3 = None
        nb1 = nb2 = nb3 = None

        overlay = np.dstack([skel * 255] * 3).astype(np.uint8)

        if pts is not None:
            TL, BL, TR, BR = pts

            for (x, y) in [TL, BL, TR, BR]:
                cv2.circle(overlay, (x, y), 2, (255, 0, 0), -1)

            # TL -> BL
            A1 = draw_line(skel, TL, BL)
            _, nb1 = find_blobs(A1)

            # TL -> BR
            A2 = draw_line(skel, TL, BR)
            _, nb2 = find_blobs(A2)

            # TR -> BR
            A3 = draw_line(skel, TR, BR)
            _, nb3 = find_blobs(A3, min_blob_area=30)

        # -----------------------------
        # plotting
        # -----------------------------
        plt.figure(figsize=(24, 8))

        plt.subplot(2, 5, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Raw")
        plt.axis("off")

        plt.subplot(2, 5, 2)
        plt.imshow((A01 * 255), cmap="gray")
        plt.title("Binary A (largest CC)")
        plt.axis("off")

        plt.subplot(2, 5, 3)
        plt.imshow(blobs_thick, cmap="gray")
        plt.title(f"Thick blobs\nn={n_blobs_thick}")
        plt.axis("off")

        plt.subplot(2, 5, 4)
        plt.imshow(skel, cmap="gray")
        plt.title(
            f"Skeleton\nthin_blobs={n_blobs_thin}, "
            f"leftover_stems={n_stems_after_blob}, endpoints={endpoints}"
        )
        plt.axis("off")

        plt.subplot(2, 5, 5)
        if stems_after_blob_img is not None:
            plt.imshow(stems_after_blob_img, cmap="gray")
            plt.title(f"Leftover stems after blob\nn={n_stems_after_blob}")
        else:
            plt.imshow(np.zeros_like(skel), cmap="gray")
            plt.title("Leftover stems after blob\nN/A")
        plt.axis("off")

        plt.subplot(2, 5, 6)
        if pts is not None:
            plt.imshow(overlay)
            plt.title("Skeleton + TL/BL/TR/BR")
        else:
            plt.imshow(skel, cmap="gray")
            plt.title("No banded points found")
        plt.axis("off")

        plt.subplot(2, 5, 7)
        if A1 is not None:
            plt.imshow(A1, cmap="gray")
            plt.title(f"TL -> BL\nnb1={nb1}")
        else:
            plt.imshow(np.zeros_like(skel), cmap="gray")
            plt.title("TL -> BL\nN/A")
        plt.axis("off")

        plt.subplot(2, 5, 8)
        if A2 is not None:
            plt.imshow(A2, cmap="gray")
            plt.title(f"TL -> BR\nnb2={nb2}")
        else:
            plt.imshow(np.zeros_like(skel), cmap="gray")
            plt.title("TL -> BR\nN/A")
        plt.axis("off")

        plt.subplot(2, 5, 9)
        if A3 is not None:
            plt.imshow(A3, cmap="gray")
            plt.title(f"TR -> BR\nnb3={nb3}")
        else:
            plt.imshow(np.zeros_like(skel), cmap="gray")
            plt.title("TR -> BR\nN/A")
        plt.axis("off")

        plt.subplot(2, 5, 10)
        plt.imshow((binary_norm > 0).astype(np.uint8) * 255, cmap="gray")
        plt.title("Binary before CC filter")
        plt.axis("off")

        plt.suptitle(
            f"True={digit} | Pred={pred_digit} | Group={pred_group}\n"
            f"{os.path.basename(img_path)}",
            fontsize=13
        )

        plt.tight_layout()
        plt.show()

        input("Press Enter for next image...")
        plt.close("all")