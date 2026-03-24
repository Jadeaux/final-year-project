import os
import string
import pandas as pd

from preprocessing import preprocess_letters, thin
from classification_letters import classify_letter
from emnist_data_loader import load_emnist_uppercase, get_letter_sample

# EXACTLY like run_letters:
ds_upper, class_list = load_emnist_uppercase(train=True)

N_PER_LETTER = 200
LETTERS = string.ascii_uppercase

results = []

for TEST_LETTER in LETTERS:
    correct = 0
    total = 0
    print(f"\nRunning letter {TEST_LETTER}...")
    # same idea as run_letters, just more occurrences
    for occ in range(1, N_PER_LETTER + 1):
        try:
            # EXACT SAME CALL STYLE AS YOUR run_letters
            img, label = get_letter_sample(ds_upper, class_list, TEST_LETTER, occurrence=occ)
        except Exception as e:
            print(f"[{TEST_LETTER}] stopped early at occ={occ} بسبب: {type(e).__name__}: {e}")
            break

        A, cleaned, _ = preprocess_letters(img, visualize=False)
        skel = thin(A)

        pred = classify_letter(A, skel)

        total += 1
        if pred == TEST_LETTER:
            correct += 1

    acc = (correct / total) if total else 0.0
    results.append({
        "Letter": TEST_LETTER,
        "Samples": total,
        "Correct": correct,
        "Accuracy_%": round(100 * acc, 2),
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))

# save table
out_dir = "lpr_outputs/eval"
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, "per_letter_accuracy.csv"), index=False)
print("\nSaved:", os.path.join(out_dir, "per_letter_accuracy.csv"))