"""
Utility functions module.
Contains debugging and evaluation utilities.
"""
import numpy as np
from collections import defaultdict
from modules.preprocessing import preprocess_step1, thin
from modules.morphology import find_blobs
from modules.features import get_stems


def debug_group1_stems(images, labels, max_samples_per_digit=300):
    """ 
    Check how many stems are detected for group 1 digits.
    
    Args:
        images: Array of images
        labels: Array of true labels
        max_samples_per_digit: Maximum number of samples to check per digit
    """
    # stats[digit][n_stems] = count
    stats = {d: defaultdict(int) for d in range(10)}

    for d in [0, 4, 6, 8, 9]:
        idxs = np.where(labels == d)[0][:max_samples_per_digit]

        for idx in idxs:
            img = images[idx]
            A, _, _ = preprocess_step1(img, visualize=False)
            A_thin = thin(A)
            blobs, n_blobs = find_blobs(A_thin)

            # only look at samples with exactly 1 blob
            if n_blobs != 1:
                continue

            stems_img, n_stems, _ = get_stems(A_thin, blobs)
            stats[d][n_stems] += 1

    # print summary
    print("\n[DEBUG] Stem counts for 1-blob samples (using current get_stems):")
    for d in [0, 4, 6, 8, 9]:
        total = sum(stats[d].values())
        if total == 0:
            print(f"Digit {d}: no 1-blob samples found.")
            continue
        print(f"Digit {d}: total 1-blob samples = {total}")
        for n_stems in sorted(stats[d].keys()):
            cnt = stats[d][n_stems]
            pct = 100.0 * cnt / total
            print(f"    stems={n_stems}: {cnt} ({pct:.1f}%)")

