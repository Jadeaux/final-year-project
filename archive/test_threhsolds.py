import numpy as np
from collections import defaultdict, Counter

from emnist_data_loader import load_emnist_uppercase, get_letter_sample
from preprocessing import preprocess_letters, thin
from features_letters import (
    hole_count_and_largest_pct,
    prune_spurs,
    count_endpoints,
    vertical_symmetry_score,
)

BLOB_LETTERS = ["A", "B", "D", "O", "P", "Q", "R"]

def predict_side(hole_count, hole_pct, thresh_pct):
    """
    Returns coarse group predicted by threshold only:
      'BIG'  -> A,B,P,R side
      'SMALL'-> D,O,Q side
    """
    if hole_count >= 2:
        return "BIG"   # treat as B-side
    return "BIG" if hole_pct >= thresh_pct else "SMALL"

def true_side(label):
    return "BIG" if label in {"A","B","P","R"} else "SMALL"

def collect_blob_samples(ds, class_list, n_per_class=200):
    """
    Collect (label, hole_count, hole_pct, endpoints, sym) for blob letters.
    Skips samples where preprocess produced empty foreground.
    """
    rows = []
    for L in BLOB_LETTERS:
        got = 0
        occ = 0
        while got < n_per_class:
            img, label = get_letter_sample(ds, class_list, L, occurrence=occ)
            occ += 1

            A01, _, _ = preprocess_letters(img, visualize=False)
            if A01.sum() == 0:
                continue

            skel = thin(A01)
            sk_pr = prune_spurs(skel, max_length=2)

            hc, hpct = hole_count_and_largest_pct(A01)
            endp = count_endpoints(sk_pr)
            sym = vertical_symmetry_score(A01)

            # keep only samples that actually have a hole (blob present)
            if hc <= 0:
                continue

            rows.append((label, hc, hpct, endp, sym))
            got += 1
    return rows

def sweep_threshold(rows, thresh_values):
    """
    Evaluate coarse side accuracy for each threshold.
    """
    results = []
    for t in thresh_values:
        correct = 0
        total = 0

        # optional: track where errors happen
        err_by_label = Counter()

        for (label, hc, hpct, endp, sym) in rows:
            pred = predict_side(hc, hpct, t)
            truth = true_side(label)
            total += 1
            if pred == truth:
                correct += 1
            else:
                err_by_label[label] += 1

        acc = correct / max(1, total)
        results.append((t, acc, err_by_label))
    # best by accuracy
    best = max(results, key=lambda x: x[1])
    return best, results

def main():
    ds_upper, class_list = load_emnist_uppercase(train=True)

    rows = collect_blob_samples(ds_upper, class_list, n_per_class=200)
    print(f"Collected blob samples: {len(rows)}")

    thresh_values = np.arange(5.0, 30.1, 0.5)  # 5% to 30% step 0.5
    best, results = sweep_threshold(rows, thresh_values)

    best_t, best_acc, best_errs = best
    print("\n=== BEST THRESHOLD (coarse BIG vs SMALL split) ===")
    print(f"best_thresh = {best_t:.1f}% | accuracy = {best_acc*100:.2f}%")
    print("errors by letter:", dict(best_errs))

    # show top few
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 thresholds:")
    for t, acc, _ in results_sorted:
        print(f"  {t:>5.1f}%  ->  {acc*100:>6.2f}%")

if __name__ == "__main__":
    main()
