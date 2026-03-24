import matplotlib.pyplot as plt

from emnist_data_loader import load_emnist_uppercase, get_letter_sample
from preprocessing import preprocess_letters, thin
from features_letters import prune_spurs, count_endpoints, debug_endpoints

LETTER = "W"
TARGET_E = 3
NEEDED = 10
PRUNE_LEN = 2   # use the same prune length you use in classification

ds_upper, class_list = load_emnist_uppercase(train=True)

found = 0
occ = 0
max_tries = 5000  # prevent infinite looping

while found < NEEDED and occ < max_tries:
    img, label = get_letter_sample(ds_upper, class_list, LETTER, occurrence=occ)

    A01, _, _ = preprocess_letters(img, visualize=False)
    skel = thin(A01)
    sk_pruned = prune_spurs(skel, max_length=PRUNE_LEN)

    e = count_endpoints(sk_pruned)

    if e == TARGET_E:
        found += 1
        print(f"\n[{LETTER}] Found #{found}: occurrence={occ} | endpoints={e}")

        # show the original + the endpoint overlay (your existing debug function)
        plt.figure(figsize=(4,4))
        plt.imshow(A01, cmap="gray")
        plt.title(f"{LETTER} crop (occ={occ})")
        plt.axis("off")
        plt.show()

        debug_endpoints(sk_pruned)  # this plots skeleton + red endpoint dots

    occ += 1

print(f"\nDone. Found {found}/{NEEDED} samples with {TARGET_E} endpoints.")