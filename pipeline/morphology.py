"""
Morphological operations module.
Handles blob detection and skeleton neighbor analysis.
"""
import numpy as np
import cv2
from skimage.measure import label


def find_blobs(A, min_blob_area=30):
    """
    Detect internal holes (blobs) in a 0/1 image A.

    A: 0/1, foreground=1, background=0 (typically thinned digit)

    Returns:
        blobs_filtered : 0/1 image, only blobs with area >= min_blob_area
        n_blobs        : number of such blobs
    """
    A = A.astype(np.uint8)

    # --- existing flood-fill logic (unchanged) ---
    A_padded = np.pad(A, pad_width=1, mode='constant', constant_values=0)

    inv = 1 - A_padded
    h, w = inv.shape

    inv_ff = (inv * 255).astype(np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(inv_ff, mask, (0, 0), 128)

    # 128 = external background, 255 = internal holes, 0 = strokes
    blobs_padded = (inv_ff == 255).astype(np.uint8)

    # Remove padding
    blobs = blobs_padded[1:-1, 1:-1]

    # --- NEW: filter small blobs by area ---
    lab = label(blobs, connectivity=2)
    blobs_filtered = np.zeros_like(blobs, dtype=np.uint8)
    n_blobs = 0

    for lbl in range(1, lab.max() + 1):
        area = np.sum(lab == lbl)
        if area >= min_blob_area:
            blobs_filtered[lab == lbl] = 1
            n_blobs += 1

    return blobs_filtered, n_blobs

def skeleton_neighbor_counts(A_thin):
    """
    For each skeleton pixel (1), count how many 8-connected neighbors it has.
    """
    K = np.array([[1,1,1],
                  [1,0,1],
                  [1,1,1]], dtype=np.uint8)  # center 0 -> only neighbors
    nb = cv2.filter2D(A_thin.astype(np.uint8), -1, K,
                      borderType=cv2.BORDER_CONSTANT)
    return nb

