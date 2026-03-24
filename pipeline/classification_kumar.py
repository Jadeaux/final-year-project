"""
Classification module.
Handles digit classification using morphological features.
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from .preprocessing import preprocess_step1, thin
from .morphology import find_blobs
from .features import get_stems, get_banded_points, draw_line



def classify_group1(A, blobs, n_blobs):
    """
    Classify digits in Group 1 using blob + stem rules.

    Inputs:
        A       : 0/1 thinned digit image
        blobs   : 0/1 blob image from find_blobs
        n_blobs : number of blobs

    Returns:
        digit (int) or None if pattern doesn't match.
    """

    # Two blobs → 8
    if n_blobs == 2:
        return 8

    # One blob: could be 0, 4, 6, 9
    if n_blobs == 1:
        stems_img, n_stems, stem_cents = get_stems(A, blobs)

        # 1 blob + 0 stems → 0
        if n_stems == 0:
            return 0


        # 1 blob + 1 stem → 6 or 9, depending on vertical position
        if n_stems == 1:
            # Blob centroid
            blob_labels = label(blobs, connectivity=2)
            blob_props = regionprops(blob_labels)
            if not blob_props:
                return None
            blob_y, blob_x = blob_props[0].centroid

            # Only stem centroid
            stem_y, stem_x = stem_cents[0]

            # Image coords: y increases downward
            # If stem is ABOVE blob → 6, else → 9
            if stem_y < blob_y:
                return 6
            else:
                return 9
            
        if n_stems == 2:
            return 4

    # Any other pattern → not handled here
    return None
def classify_with_blobs_from_A(A, visualize=False, debug=False):
    A_thin = thin(A)

    # 1) Decide group using THICK blobs (as you want for plates)
    blobs_thick, n_blobs_thick = find_blobs(A)

    if debug:
        print(f"[INSIDE] thick blobs = {n_blobs_thick}")

    if n_blobs_thick > 0:
        # 2) For Group 1, do stems/topology on THIN (more stable)
        blobs_thin, n_blobs_thin = find_blobs(A_thin)
        digit = classify_group1(A_thin, blobs_thin, n_blobs_thick)
        group = "Group 1"
    else:
        # 3) For Group 2, ALL helper-line + stems logic must be on THIN
        digit = classify_group2(A_thin, visualize=visualize)
        group = "Group 2"

    if debug:
        print(f"[INSIDE] chose {group}, digit={digit}")

    return digit, group

def debug_group1_stems(images, labels, max_samples_per_digit=300):
    """ checking how many stems are detected for group 1 digits """
    from collections import defaultdict

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



def classify_group2(A, visualize=False):
    """
    Classify digits when the ORIGINAL thinned image has NO blobs.
    Implements the helper-line rules...
    """
    pts = get_banded_points(A, split=0.5)
    if pts is None:
        return None
    TL, BL, TR, BR = pts

    # TL->BL (leftmost above mid to leftmost below mid)
    A1 = draw_line(A, TL, BL)
    blobs1, nb1 = find_blobs(A1)

    if visualize:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(A, cmap='gray')
        # show the exact TL / BL points we are using for the helper line
        plt.scatter([TL[0], BL[0], TR[0], BR[0]],
            [TL[1], BL[1], TR[1], BR[1]],
            c='red', s=25)
        plt.title("Group2 input (A_thin)")
        plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(A1, cmap='gray');  plt.title("After TL→BL line");     plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(blobs1, cmap='gray'); plt.title(f"Blobs after TL→BL (n={nb1})"); plt.axis('off')
        plt.tight_layout(); plt.show()

    # ---- Case 1: No blobs after TL->BL → digit is 1 or 4 ----
    if nb1 == 0:
        # Draw TL->BR diagonal
        A2 = draw_line(A, TL, BR)
        blobs2, nb2 = find_blobs(A2)

        if visualize:
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1); plt.imshow(A2, cmap='gray');   plt.title("After TL→BR line"); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(blobs2, cmap='gray'); plt.title(f"Blobs after TL→BR (n={nb2})"); plt.axis('off')
            plt.tight_layout(); plt.show()

        if nb2 == 0:
            return 1
        else:
            return 4  # diagonal created a blob → 4

    # ---- Case 2: Blobs appear after TL->BL → {2,5,3,7} ----
    stems_img, n_stems, stem_cents = get_stems(A1, blobs1)

    if visualize:
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1); plt.imshow(A1, cmap='gray');        plt.title("A1 (TL→BL)"); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(stems_img, cmap='gray'); plt.title(f"Stems (n={n_stems})"); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(blobs1, cmap='gray');    plt.title("Blobs (for stems)"); plt.axis('off')
        plt.tight_layout(); plt.show()

    # 2a) stems exist → {2,5}
    if n_stems > 0:
        blob_labels = label(blobs1, connectivity=2)
        blob_props = regionprops(blob_labels)
        if not blob_props or not stem_cents:
            return None

        blob = max(blob_props, key=lambda r: r.area)
        blob_y, blob_x = blob.centroid
        stem_y, stem_x = stem_cents[0]

        if stem_y < blob_y:
            return 5
        else:
            return 2

    # 2b) No stems after TL->BL + blobs → {3,7}
    A3 = draw_line(A, TR, BR)
    blobs3, nb3 = find_blobs(A3, min_blob_area=30)
    lab = label(blobs3, connectivity=2)
    props = regionprops(lab)
    print("nb3 =", nb3)
    if len(props) == 1:
        print("DEBUGGING =", props[0].area)
        print("blob area =", props[0].area)
    else:
        print("areas =", [p.area for p in props])

    if nb3 > 0:
        for i, p in enumerate(props):
            print(f"Blob {i+1} area:", p.area)
    else:
        print("No blobs detected.")
    if visualize:
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1); plt.imshow(A, cmap='gray');  plt.title("A (input)"); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(A3, cmap='gray'); plt.title("After TR→BR line"); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(blobs3, cmap='gray'); plt.title(f"Blobs after TR→BR (n={nb3})"); plt.axis('off')
        plt.tight_layout(); plt.show()

    if nb3 > 0:
        return 3
    else:
        return 7


def classify_with_blobs(img, visualize=True):
    """
    Main classification function that processes an image and returns the predicted digit.
    
    Args:
        img: Input image (numpy array)
        visualize: Whether to show visualization plots
        
    Returns:
        digit: Predicted digit (int) or None
        group: "Group 1" or "Group 2"
    """
    # Step 1: preprocess
    A, cropped_vis, _ = preprocess_step1(img, visualize=False)

    # Step 2: thin + blobs
    A_thin = thin(A)
    blobs, n_blobs = find_blobs(A_thin)

    if n_blobs > 0:
        digit = classify_group1(A_thin, blobs, n_blobs)
        group = "Group 1"
    else:
        # For Group 2 digits (no initial blobs), optionally visualize
        # the helper-line process (TL→BL, TL→BR, TR→BR).
        digit = classify_group2(A_thin, visualize=visualize)
        group = "Group 2"

    if visualize:
        plt.figure(figsize=(9,2.5))
        plt.subplot(1,3,1); plt.imshow(cropped_vis, cmap='gray'); plt.title('Cropped'); plt.axis('off')
        plt.subplot(1,3,2); plt.imshow(A_thin, cmap='gray'); plt.title('Thinned'); plt.axis('off')
        plt.subplot(1,3,3); plt.imshow(blobs, cmap='gray'); plt.title(f'Blobs at start (n={n_blobs})'); plt.axis('off')
        plt.tight_layout(); plt.show()
        print("Predicted:", digit, "|", group)

    return digit, group

'''
def classify_with_blobs(img, visualize=True):
    A, cropped_vis, _ = preprocess_step1(img, visualize=False)

    A_thin = thin(A)

    # blobs on THICK binary
    blobs, n_blobs = find_blobs(A)

    if n_blobs > 0:
        digit = classify_group1(A_thin, blobs, n_blobs)
        group = "Group 1"
    else:
        digit = classify_group2(A_thin, visualize=visualize)
        group = "Group 2"

    if visualize:
        plt.figure(figsize=(9,2.5))
        plt.subplot(1,3,1); plt.imshow(cropped_vis, cmap='gray'); plt.title('Cropped'); plt.axis('off')
        plt.subplot(1,3,2); plt.imshow(A_thin, cmap='gray'); plt.title('Thinned'); plt.axis('off')
        plt.subplot(1,3,3); plt.imshow(blobs, cmap='gray'); plt.title(f'Blobs at start (n={n_blobs})'); plt.axis('off')
        plt.tight_layout(); plt.show()
        print("Predicted:", digit, "|", group)

    return digit, group
'''
import numpy as np
from skimage.measure import label, regionprops
from .morphology import find_blobs
from .features import get_stems

def summarize_blobs_and_stems(A_thin, tag=""):
    blobs, n_blobs = find_blobs(A_thin)

    lab = label(blobs, connectivity=2)
    props = regionprops(lab)
    areas = sorted([p.area for p in props], reverse=True)

    stems_img, n_stems, stem_cents = get_stems(A_thin, blobs)

    # Basic polarity sanity
    fg = int(np.sum(A_thin == 1))
    bg = int(np.sum(A_thin == 0))

    print(f"\n[{tag}] fg_pixels={fg}, bg_pixels={bg}")
    print(f"[{tag}] n_blobs={n_blobs}, blob_areas={areas[:5]}")
    print(f"[{tag}] n_stems={n_stems}, stem_centroids={stem_cents[:3] if stem_cents else stem_cents}")

    return blobs, n_blobs, stems_img, n_stems
