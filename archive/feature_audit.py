import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from emnist_data_loader import load_emnist_uppercase, get_letter_sample
from preprocessing import preprocess_letters, thin
from features_letters import prune_spurs, count_endpoints, count_holes
from features_letters import vertical_symmetry_lr_balance

LETTERS = list(string.ascii_uppercase)
N_PER_LETTER = 200

# Choose endpoint buckets
# You can change these labels easily
BUCKETS = ["0", "1", "2", "3", "4", "5+"]

ds_upper, class_list = load_emnist_uppercase(train=True)

def bucket_endpoints(e: int) -> str:
    if e >= 5:
        return "5+"
    return str(e)

def audit_endpoints(prune_len=2):
    ds_upper, class_list = load_emnist_uppercase(train=True)

    # table counts: rows=true letter, cols=bucket
    counts = pd.DataFrame(0, index=LETTERS, columns=BUCKETS, dtype=int)

    for L in LETTERS:
        got = 0
        occ = 0
        while got < N_PER_LETTER:
            img, label = get_letter_sample(ds_upper, class_list, L, occurrence=occ)
            occ += 1

            A01, _, _ = preprocess_letters(img, visualize=False)
            if A01.sum() == 0:
                continue  # skip destroyed preprocessing samples

            skel = thin(A01)
            sk = prune_spurs(skel, max_length=prune_len)

            e = count_endpoints(sk)
            b = bucket_endpoints(e)

            counts.loc[label, b] += 1
            got += 1

        print(f"Done {L}")

    # Add row totals + row-normalised (%) view
    counts["total"] = counts.sum(axis=1)

    perc = counts[BUCKETS].div(counts["total"], axis=0) * 100
    perc = perc.round(1)

    print("\n=== Endpoint COUNT distribution (counts) ===")
    print(counts.to_string())

    print("\n=== Endpoint COUNT distribution (row %) ===")
    out = perc.copy()
    out["total"] = counts["total"]
    print(out.to_string())

    return counts, out



LETTERS = list(string.ascii_uppercase)
N_PER_LETTER = 200

BUCKETS = ["0", "1", "2", "3+"]

OUT_DIR = "audit_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def bucket_holes(h: int) -> str:
    if h >= 3:
        return "3+"
    return str(h)

def audit_holes():
    ds_upper, class_list = load_emnist_uppercase(train=True)

    counts = pd.DataFrame(0, index=LETTERS, columns=BUCKETS, dtype=int)

    for L in LETTERS:
        got = 0
        occ = 0

        while got < N_PER_LETTER:
            img, label = get_letter_sample(ds_upper, class_list, L, occurrence=occ)
            occ += 1

            A01, _, _ = preprocess_letters(img, visualize=False)
            if A01.sum() == 0:
                continue

            h = count_holes(A01)
            b = bucket_holes(h)
            counts.loc[label, b] += 1
            got += 1

        print(f"Done {L}")

    counts["total"] = counts.sum(axis=1)

    perc = counts[BUCKETS].div(counts["total"], axis=0) * 100
    perc = perc.round(1)
    perc["total"] = counts["total"]

    # Save
    counts_path = os.path.join(OUT_DIR, f"holes_counts_n{N_PER_LETTER}.csv")
    perc_path   = os.path.join(OUT_DIR, f"holes_percent_n{N_PER_LETTER}.csv")

    counts.to_csv(counts_path)
    perc.to_csv(perc_path)

    print("\n=== Hole COUNT distribution (counts) ===")
    print(counts.to_string())

    print("\n=== Hole COUNT distribution (row %) ===")
    print(perc.to_string())

    print("\nSaved to:")
    print(" ", counts_path)
    print(" ", perc_path)

    return counts, perc
LETTERS = list(string.ascii_uppercase)
N_PER_LETTER = 200

OUT_DIR = "audit_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SYMMETRIC_LETTERS = set(["A","H","I","M","O","T","U","V","W","X","Y"])


def audit_symmetry(ds_upper, class_list, save=True):
    rows = []

    for L in LETTERS:
        got = 0
        occ = 0

        while got < N_PER_LETTER:
            img, label = get_letter_sample(ds_upper, class_list, L, occurrence=occ)
            occ += 1

            A01, _, _ = preprocess_letters(img, visualize=False)
            if A01.sum() == 0:
                continue

            score = vertical_symmetry_lr_balance(A01)

            rows.append({
                "letter": L,
                "sym_score": score,
                "is_symmetric": 1 if L in SYMMETRIC_LETTERS else 0
            })

            got += 1

        print(f"Done {L}")

    df_sym = pd.DataFrame(rows)

    # per-letter stats
    letter_stats = df_sym.groupby("letter")["sym_score"].agg(["mean","std"]).sort_values("mean", ascending=False)
    print("\n=== Symmetry score per letter (mean/std) ===")
    print(letter_stats)

    # histogram
    sym_scores = df_sym[df_sym["is_symmetric"] == 1]["sym_score"]
    nonsym_scores = df_sym[df_sym["is_symmetric"] == 0]["sym_score"]

    plt.figure(figsize=(8,4))
    plt.hist(sym_scores, bins=30, alpha=0.6, label="Symmetric GT")
    plt.hist(nonsym_scores, bins=30, alpha=0.6, label="Non-symmetric GT")
    plt.legend()
    plt.title("Vertical Symmetry Score Distribution")
    plt.xlabel("sym_score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    if save:
        df_path = os.path.join(OUT_DIR, f"symmetry_scores_n{N_PER_LETTER}.csv")
        stats_path = os.path.join(OUT_DIR, f"symmetry_letter_stats_n{N_PER_LETTER}.csv")
        df_sym.to_csv(df_path, index=False)
        letter_stats.to_csv(stats_path)
        print("\nSaved to:")
        print(" ", df_path)
        print(" ", stats_path)

    return df_sym, letter_stats


import os
import string
import numpy as np
import pandas as pd

from emnist_data_loader import load_emnist_uppercase, get_letter_sample
from preprocessing import preprocess_letters, thin
from features_letters import prune_spurs
from features_letters import count_vertical_lines, count_horizontal_lines  # or wherever these live

LETTERS = list(string.ascii_uppercase)
N_PER_LETTER = 200

BUCKETS_LINES = ["0", "1", "2", "3+"]

OUT_DIR = "audit_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def bucket_3plus(k: int) -> str:
    return "3+" if k >= 3 else str(k)

def audit_lines(min_frac=0.35, prune_len=2, save=True):
    """
    Produces:
      - vertical_lines_counts, vertical_lines_percent
      - horizontal_lines_counts, horizontal_lines_percent
    Each is a (26 x 5) table: 0,1,2,3+,total
    """

    ds_upper, class_list = load_emnist_uppercase(train=True)

    v_counts = pd.DataFrame(0, index=LETTERS, columns=BUCKETS_LINES, dtype=int)
    h_counts = pd.DataFrame(0, index=LETTERS, columns=BUCKETS_LINES, dtype=int)

    for L in LETTERS:
        got = 0
        occ = 0

        while got < N_PER_LETTER:
            img, label = get_letter_sample(ds_upper, class_list, L, occurrence=occ)
            occ += 1

            A01, _, _ = preprocess_letters(img, visualize=False)
            if A01.sum() == 0:
                continue

            skel = thin(A01)
            sk = prune_spurs(skel, max_length=prune_len)

            v = count_vertical_lines(sk, min_frac=min_frac)
            h = count_horizontal_lines(sk, min_frac=min_frac)

            v_counts.loc[label, bucket_3plus(v)] += 1
            h_counts.loc[label, bucket_3plus(h)] += 1

            got += 1

        print(f"Done {L}")

    # totals
    v_counts["total"] = v_counts.sum(axis=1)
    h_counts["total"] = h_counts.sum(axis=1)

    # row %
    v_perc = (v_counts[BUCKETS_LINES].div(v_counts["total"], axis=0) * 100).round(1)
    h_perc = (h_counts[BUCKETS_LINES].div(h_counts["total"], axis=0) * 100).round(1)
    v_perc["total"] = v_counts["total"]
    h_perc["total"] = h_counts["total"]

    # print
    print("\n=== VERTICAL LINE COUNT distribution (counts) ===")
    print(v_counts.to_string())

    print("\n=== VERTICAL LINE COUNT distribution (row %) ===")
    print(v_perc.to_string())

    print("\n=== HORIZONTAL LINE COUNT distribution (counts) ===")
    print(h_counts.to_string())

    print("\n=== HORIZONTAL LINE COUNT distribution (row %) ===")
    print(h_perc.to_string())

    # save
    if save:
        v_counts_path = os.path.join(OUT_DIR, f"vlines_counts_n{N_PER_LETTER}_mf{min_frac}_pr{prune_len}.csv")
        v_perc_path   = os.path.join(OUT_DIR, f"vlines_percent_n{N_PER_LETTER}_mf{min_frac}_pr{prune_len}.csv")
        h_counts_path = os.path.join(OUT_DIR, f"hlines_counts_n{N_PER_LETTER}_mf{min_frac}_pr{prune_len}.csv")
        h_perc_path   = os.path.join(OUT_DIR, f"hlines_percent_n{N_PER_LETTER}_mf{min_frac}_pr{prune_len}.csv")

        v_counts.to_csv(v_counts_path)
        v_perc.to_csv(v_perc_path)
        h_counts.to_csv(h_counts_path)
        h_perc.to_csv(h_perc_path)

        print("\nSaved to:")
        print(" ", v_counts_path)
        print(" ", v_perc_path)
        print(" ", h_counts_path)
        print(" ", h_perc_path)

    return v_counts, v_perc, h_counts, h_perc

if __name__ == "__main__":
    ds_upper, class_list = load_emnist_uppercase(train=True)

    # run whichever audits you want
    # audit_holes()
    # audit_endpoints(prune_len=2)
    #audit_symmetry(ds_upper, class_list)
    audit_lines(min_frac=0.35, prune_len=2, save=True)