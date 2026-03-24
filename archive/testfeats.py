import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

from emnist_data_loader import load_emnist_uppercase, get_sample
from preprocessing import preprocess_letters, thin
from features_letters import count_holes, count_endpoints, prune_spurs


def audit_uppercase_features(
    n_per_class=200,
    prune_max_length=2,
    target_size=None,          # set to 120 if you re-enable resizing; else keep None
    visualize_bad=False,
    bad_per_class=3,
    bad_threshold=3,           # "bad" if |endpoints - mode| >= this
    seed=0
):
    """
    Audits holes + endpoints for each uppercase letter in EMNIST.

    Prints:
      - for each letter: holes mode (and %), endpoints mode (and %), min/median/max endpoints
    Optionally visualizes a few 'bad' samples per letter.

    NOTE: This doesn't assume Lewis "expected" values. It reports what your pipeline produces.
    """

    rng = np.random.default_rng(seed)

    ds_upper, class_list = load_emnist_uppercase(train=True)

    # map label char -> indices in dataset (we'll sample from these)
    # class_list should be list of chars aligned with your get_sample mapping
    letter_to_indices = defaultdict(list)
    for idx in range(len(ds_upper)):
        # get_sample returns label as a char in your loader, but if it returns int adjust below
        _, lab = get_sample(ds_upper, idx, class_list)
        letter_to_indices[lab].append(idx)

    # results
    holes_counts = {ch: [] for ch in class_list}
    endp_counts  = {ch: [] for ch in class_list}

    # collect "bad" examples for inspection
    bad_examples = {ch: [] for ch in class_list}

    for ch in class_list:
        indices = letter_to_indices[ch]
        if len(indices) == 0:
            continue

        # sample up to n_per_class
        take = min(n_per_class, len(indices))
        sampled = rng.choice(indices, size=take, replace=False)

        # first pass: gather features
        for idx in sampled:
            img, lab = get_sample(ds_upper, int(idx), class_list)

            if target_size is None:
                A, resized, _ = preprocess_letters(img, visualize=False)  # your current no-resize version
            else:
                A, resized, _ = preprocess_letters(img, visualize=False, target_size=target_size)

            skel = thin(A)
            skel2 = prune_spurs(skel, max_length=prune_max_length)

            h = count_holes(A)
            e = count_endpoints(skel2)

            holes_counts[ch].append(h)
            endp_counts[ch].append(e)

        # compute mode endpoints for "bad" detection
        e_mode = Counter(endp_counts[ch]).most_common(1)[0][0] if endp_counts[ch] else None

        # second pass (optional): store a few bad examples for plotting
        if visualize_bad and e_mode is not None:
            for idx in sampled:
                img, lab = get_sample(ds_upper, int(idx), class_list)

                if target_size is None:
                    A, resized, _ = preprocess_letters(img, visualize=False)
                else:
                    A, resized, _ = preprocess_letters(img, visualize=False, target_size=target_size)

                skel = thin(A)
                skel2 = prune_spurs(skel, max_length=prune_max_length)
                e = count_endpoints(skel2)

                if abs(e - e_mode) >= bad_threshold:
                    bad_examples[ch].append((img, A, skel2, e, e_mode, int(idx)))
                    if len(bad_examples[ch]) >= bad_per_class:
                        break

    # ---- print summary table ----
    print("\n=== FEATURE AUDIT SUMMARY ===")
    print(f"n_per_class={n_per_class}, prune_max_length={prune_max_length}, target_size={target_size}")
    print("Letter | Holes(mode,%) | Endpoints(mode,%) | Endp min/med/max | N")
    print("-" * 78)

    for ch in class_list:
        hs = holes_counts[ch]
        es = endp_counts[ch]
        if len(es) == 0:
            continue

        h_mode, h_mode_ct = Counter(hs).most_common(1)[0]
        e_mode, e_mode_ct = Counter(es).most_common(1)[0]

        h_mode_pct = 100.0 * h_mode_ct / len(hs)
        e_mode_pct = 100.0 * e_mode_ct / len(es)

        e_min = int(np.min(es))
        e_med = int(np.median(es))
        e_max = int(np.max(es))

        print(f"  {ch}    |   {h_mode} ({h_mode_pct:5.1f}%)  |     {e_mode} ({e_mode_pct:5.1f}%)   |"
              f"   {e_min:2d}/{e_med:2d}/{e_max:2d}     | {len(es)}")

    # ---- visualize bad examples ----
    if visualize_bad:
        for ch in class_list:
            if not bad_examples[ch]:
                continue

            for (img, A, skel2, e, e_mode, idx) in bad_examples[ch]:
                plt.figure(figsize=(10, 3))
                plt.suptitle(f"{ch}  idx={idx}  endpoints={e} (mode={e_mode})")

                plt.subplot(1, 3, 1)
                plt.imshow(img, cmap="gray")
                plt.title("Original")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(A, cmap="gray")
                plt.title("A01")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(skel2, cmap="gray")
                plt.title("Skeleton pruned")
                plt.axis("off")

                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
    audit_uppercase_features(
        n_per_class=200,
        prune_max_length=2,
        target_size=None,      # keep None if you removed resizing
        visualize_bad=True,    # flip to False if you just want the table
        bad_per_class=2,
        bad_threshold=3
    )
