import cv2
import matplotlib.pyplot as plt
import string
import os

from .preprocessing import preprocess_letters, thin
from .classification_letters import classify_letter
from .features_letters import (
    prune_spurs,
    count_endpoints,
    hole_count_and_largest_pct,
    vertical_symmetry_lr_balance,
    count_vertical_lines,
    count_horizontal_lines,
    concavity_tb_strength,
    debug_count_holes,
    debug_endpoints,
    debug_vertical_symmetry_lr_balance,
    debug_vertical_lines,
    debug_horizontal_lines,
    debug_horizontal_symmetry_tb_balance,
    debug_concavity_tb,
    count_vertical_strokes,
    count_horizontal_strokes,
    count_horizontal_bars_context,
    horizontal_symmetry_tb_balance,
    bottom_width_ratio, debug_bottom_width_ratio,
    center_density_ratio
)
#TODO: S is misclassified as Z due to the horizontal bar detection. Z is vertically symmetric, Q, fix X  (not)
# -----------------------
# 1) Load local letter
# -----------------------
IMG_PATH = r"C:\Users\jadsa\GitRepos\final-year-project-jad\images\letters/letterA2.png"
TRUE_LABEL = "S"

if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Missing: {IMG_PATH}")

print("Processing:", IMG_PATH, "True label:", TRUE_LABEL)

img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load {IMG_PATH}")



# -----------------------
# 2) Preprocess + thin
# -----------------------
A, cleaned, bin_norm = preprocess_letters(img, visualize=True)
skel = thin(A)

# prune (same as your classifier uses)
sk_pruned = prune_spurs(skel, max_length=2)

# -----------------------
# 3) Predict
# -----------------------
pred = classify_letter(A, skel)
print("True label:", TRUE_LABEL)
print("Predicted:", pred)

# -----------------------
# 4) Print all features
# -----------------------
hc, hpct = hole_count_and_largest_pct(A)
e = count_endpoints(sk_pruned)
sym = vertical_symmetry_lr_balance(A)
sym2 = horizontal_symmetry_tb_balance(A)
vlines = count_vertical_strokes(sk_pruned, min_frac=0.6)
hlines = count_horizontal_strokes(sk_pruned, min_frac=0.8, gap_allow=2, band=5, support_rows=3, orient_thresh=0.12, rel_width_keep=0.80)
north, south = concavity_tb_strength(A)
bw = bottom_width_ratio(A, band_frac=0.20)
ratio = center_density_ratio(A, frac=0.35)
print("Center density ratio:", ratio)
print("Bottom width ratio:", round(bw, 3))


# print("Hole count:", hc, "| Largest hole %:", round(hpct, 2))
print("Endpoints(after prune):", e)
print("Vertical symmetry score:", round(sym, 3))
print("Horizontal symmetry score:", round(sym2, 3))
print("vlines:", vlines, "| hlines:", hlines)
# print(f"north={north:.3f} | south={south:.3f}")
print("hole count:", hc, "| hole %:", round(hpct, 2))

vs = vertical_symmetry_lr_balance(A)
# print("Vertical symmetry score:", round(vs, 3))

# ----------------------
# 5) Visualise EVERYTHING
# -----------------------

# Show skeleton image
plt.figure(figsize=(4,4))
plt.imshow(sk_pruned, cmap="gray")
plt.title("Skeleton (pruned)")
# plt.axis("off")
# plt.show()

# # Holes debug
# debug_count_holes(A, min_hole_area=30)

# # Endpoints debug (red dots)
debug_endpoints(sk_pruned)

# # # Symmetry debug
# #debug_vertical_symmetry_lr_balance(A)
# debug_horizontal_symmetry_tb_balance(A)

# # Line detection debug
# debug_vertical_lines(sk_pruned, min_frac=0.25, gap_allow=3, band=7)
# debug_horizontal_lines(sk_pruned, min_frac=0.25, gap_allow=3, band=7)
from .features_letters import debug_vertical_strokes, debug_horizontal_strokes

debug_vertical_strokes(sk_pruned, min_frac=0.6, gap_allow=2, support_cols=3, orient_thresh=0.12)
debug_horizontal_strokes(sk_pruned, min_frac=0.8, gap_allow=2, band=5, support_rows=3, orient_thresh=0.12, rel_width_keep=0.80)
debug_bottom_width_ratio(A, band_frac=0.20)
# # Concavity debug
import matplotlib.pyplot as plt
# plt.imshow(A, cmap="gray")
# plt.title("A01 passed to features")
# plt.axis("off")
# plt.show()

# debug_concavity_tb(A)


import numpy as np
import cv2

def remove_branchpoints(skel01):
    """
    Remove skeleton pixels with >=3 neighbors.
    Returns cleaned skeleton.
    """
    sk = (skel01 > 0).astype(np.uint8)

    kernel = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ], dtype=np.uint8)

    neigh = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    branch = np.logical_and(sk == 1, neigh >= 3)

    sk_clean = sk.copy()
    sk_clean[branch] = 0

    return sk_clean



