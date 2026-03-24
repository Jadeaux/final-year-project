"""
Morphological operations module.
Handles blob detection and skeleton neighbor analysis.
"""
import numpy as np
import cv2
from skimage.measure import label

'''
def find_blobs(A):
    """
    Detect internal holes (blobs) in a 0/1 image A.

    A: 0/1, foreground=1, background=0 (typically thinned digit)

    Returns:
        blobs   : 0/1 image where 1 = blob pixels (holes)
        n_blobs : number of connected blob components
    """
    A = A.astype(np.uint8)

    # Pad with a 1-pixel background border so (0,0) is guaranteed outside.
    A_padded = np.pad(A, pad_width=1, mode='constant', constant_values=0)

    # Invert: foreground->0, background->1
    inv = 1 - A_padded
    h, w = inv.shape

    # Prepare for floodFill
    inv_ff = (inv * 255).astype(np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood-fill from (0,0): this is now definitely external background
    cv2.floodFill(inv_ff, mask, (0, 0), 128)

    # After floodFill:
    #   - 128 = external background (connected to padded border)
    #   - 255 = internal regions not reached => HOLES
    #   - 0   = original foreground (digit strokes)

    blobs_padded = (inv_ff == 255).astype(np.uint8)

    # Remove the padding to get back to original size
    blobs = blobs_padded[1:-1, 1:-1]

    # Count connected components in blobs
    lab = label(blobs, connectivity=2)
    n_blobs = lab.max()

    return blobs, n_blobs
'''

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
'''

def find_blobs(A, min_blob_area=10):
    A = A.astype(np.uint8)

    A_padded = np.pad(A, pad_width=1, mode='constant', constant_values=0)

    inv = 1 - A_padded
    h, w = inv.shape

    inv_ff = (inv * 255).astype(np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(inv_ff, mask, (0, 0), 128)

    blobs_padded = (inv_ff == 255).astype(np.uint8)
    blobs = blobs_padded[1:-1, 1:-1]

    # ---- filter small blobs by area (your existing code) ----
    lab = label(blobs, connectivity=2)
    blobs_filtered = np.zeros_like(blobs, dtype=np.uint8)

    for lbl in range(1, lab.max() + 1):
        area = np.sum(lab == lbl)
        if area >= min_blob_area:
            blobs_filtered[lab == lbl] = 1

    # ✅ NEW: break thin bridges between holes (helps 8 => 2 blobs)
    kernel = np.ones((3, 3), np.uint8)
    blobs_clean = cv2.morphologyEx(blobs_filtered.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)

    # ✅ Recount after opening (important!)
    lab2 = label(blobs_clean, connectivity=2)
    n_blobs = lab2.max()

    return blobs_clean, n_blobs
'''

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

