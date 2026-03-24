import matplotlib.pyplot as plt
from preprocessing import preprocess_letters, thin
from features_letters import (
    prune_spurs,
    debug_count_holes,
    debug_endpoints,
    count_endpoints,
    hole_count_and_largest_pct,
    vertical_symmetry_lr_balance,
    debug_vertical_symmetry_lr_balance,
    count_vertical_lines,
    count_horizontal_lines, 
    debug_vertical_lines, debug_horizontal_lines, debug_misclassified_sample, horizontal_symmetry_tb_balance, debug_horizontal_symmetry_tb_balance, debug_concavity_tb, 
    concavity_tb_strength
)
from classification_letters import classify_letter
from emnist_data_loader import load_emnist_uppercase, get_letter_sample


ds_upper, class_list = load_emnist_uppercase(train=True)

TEST_LETTER = "W"   



# pick a sample to test
for occ in range(1, 11):
    img, label = get_letter_sample(ds_upper, class_list, TEST_LETTER, occurrence=occ)
    A, cleaned, _ = preprocess_letters(img, visualize=False)
    skel = thin(A)

    pred = classify_letter(A, skel)

    hc, hpct = hole_count_and_largest_pct(A)
    sk_pruned = prune_spurs(skel, max_length=2)
    e = count_endpoints(sk_pruned)
    sym = vertical_symmetry_lr_balance(A)

    vlines = count_vertical_lines(sk_pruned, min_frac=0.35)
    hlines = count_horizontal_lines(sk_pruned, min_frac=0.35)

    north, south = concavity_tb_strength(A)

    print(f"[{TEST_LETTER} #{occ}] pred={pred} | holes={hc} | "
          f"endp={e} | sym={sym:.3f} | "
          f"north={north:.3f} | south={south:.3f}")

    debug_vertical_lines(sk_pruned, min_frac=0.35)
    
    # if pred != TEST_LETTER:
    #     debug_misclassified_sample(
    #         A01=A,
    #         skel01=sk_pruned,
    #         true_label=TEST_LETTER,
    #         pred_label=pred,
    #         hole_count=hc,
    #         hole_pct=hpct,
    #         endpoints=e,
    #         sym_score=sym,
    #         vlines=vlines,
    #         hlines=hlines,
    #         occ=occ
    #     )


A, cleaned, _ = preprocess_letters(img, visualize=True)
skel = thin(A)

# --- CRAMM classification ---
pred = classify_letter(A, skel)
print("True label:", label)
print("Predicted:", pred)

# --- Print the routing features used by the blob tree ---
hc, hpct = hole_count_and_largest_pct(A)
print("Hole count:", hc, "| Largest hole %:", round(hpct, 2))

sk_pruned = prune_spurs(skel, max_length=2)
e = count_endpoints(sk_pruned)
print("Endpoints(after prune):", e)

sym = vertical_symmetry_lr_balance(A)
print("Vertical symmetry score:", round(sym, 3))


# --- Visuals ---
plt.figure(figsize=(4,4))
plt.imshow(sk_pruned, cmap="gray")
plt.title(f"Skeleton (label={label})")
plt.axis("off")
plt.show()

# debug_count_holes(A)
# debug_endpoints(sk_pruned)
# debug_vertical_symmetry_lr_balance(A)
# debug_vertical_lines(sk_pruned, min_frac=0.35)
# debug_horizontal_lines(sk_pruned, min_frac=0.35)

from features_letters import debug_hole_bbox_percentage

#debug_hole_bbox_percentage(A)

# Only run line drawing debug when you need it (later rules)
# debug_line_draw(A, which="TLTR")
# debug_line_draw(A, which="BLBR")
