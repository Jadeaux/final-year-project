import os
import glob
import csv
import random
import cv2
import numpy as np

from pipeline.preprocessing_chars import preprocess_step1, preprocess_letters

# =========================================================
# CONFIG
# =========================================================
ROOT = r"E:\EnglishImg (1)\English\Img\GoodImg\Bmp"
OUT_DIR = r"images/plates/chars74_k synthetic plates"

NUM_IMAGES = 200
PLATE_LEN = 7

DIGIT_LABELS = [str(i) for i in range(10)]
LETTER_LABELS = [chr(ord("A") + i) for i in range(26)]
ALL_LABELS = DIGIT_LABELS + LETTER_LABELS

CHAR_TARGET_HEIGHT = 60
MIN_GAP = 6
MAX_GAP = 14
TOP_PAD = 12
BOTTOM_PAD = 12
LEFT_PAD = 14
RIGHT_PAD = 14

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# LABEL / FOLDER HELPERS
# =========================================================
def sample_folder_to_label(folder_name):
    if not folder_name.startswith("Sample"):
        return None

    idx = int(folder_name.replace("Sample", ""))

    if 1 <= idx <= 10:
        return str(idx - 1)

    if 11 <= idx <= 36:
        return chr(ord("A") + (idx - 11))

    return None


def is_digit_folder(folder_name):
    idx = int(folder_name.replace("Sample", ""))
    return 1 <= idx <= 10


# =========================================================
# MAIN COMPONENT EXTRACTION
# =========================================================
def keep_largest_component(A01: np.ndarray) -> np.ndarray:
    """
    Input:
        A01: binary image, expected 0/1
    Output:
        largest connected component only, still 0/1
    """
    A01 = (A01 > 0).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(A01, connectivity=8)

    if num <= 1:
        return A01

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + np.argmax(areas)

    return (labels == largest).astype(np.uint8)


def crop_tight(bin255: np.ndarray) -> np.ndarray:
    ys, xs = np.where(bin255 > 0)
    if xs.size == 0:
        return bin255
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return bin255[y0:y1+1, x0:x1+1]


def resize_to_target_height(bin255: np.ndarray, target_h: int) -> np.ndarray:
    h, w = bin255.shape
    if h == 0 or w == 0:
        return bin255

    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))

    return cv2.resize(bin255, (new_w, target_h), interpolation=cv2.INTER_NEAREST)


# =========================================================
# CLEAN ONE CHARS74K IMAGE EXACTLY IN YOUR STYLE
# =========================================================
def clean_chars74k_image_for_synthetic(img_path, sample_folder):
    """
    Returns a clean 0/255 image containing only the main character.
    Uses the same branching logic as your evaluation script.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    try:
        if is_digit_folder(sample_folder):
            A, cleaned, binary_norm = preprocess_step1(img, visualize=False)
            A = (A > 0).astype(np.uint8)

            # same idea as your evaluation:
            # keep only the main component to remove neighbouring junk
            A = keep_largest_component(A)

        else:
            A, cleaned, binary_norm = preprocess_letters(img, visualize=False)
            A = (A > 0).astype(np.uint8)

            # IMPORTANT:
            # some letter samples may also contain neighbouring junk,
            # so do the same largest-component cleanup here as well
            A = keep_largest_component(A)

        A255 = (A * 255).astype(np.uint8)
        A255 = crop_tight(A255)

        ys, xs = np.where(A255 > 0)
        if xs.size == 0:
            return None

        return A255

    except Exception as e:
        print(f"Error preprocessing {img_path}: {e}")
        return None


# =========================================================
# BUILD A BANK OF CLEANED CHARACTER IMAGES
# =========================================================
def build_clean_char_bank(root):
    """
    Returns:
        char_bank[label] = [clean_img1, clean_img2, ...]
    """
    folders = sorted([
        f for f in os.listdir(root)
        if os.path.isdir(os.path.join(root, f))
    ])

    char_bank = {label: [] for label in ALL_LABELS}

    for folder in folders:
        true_label = sample_folder_to_label(folder)
        if true_label is None:
            continue

        folder_path = os.path.join(root, folder)
        image_paths = (
            glob.glob(os.path.join(folder_path, "*.png")) +
            glob.glob(os.path.join(folder_path, "*.jpg")) +
            glob.glob(os.path.join(folder_path, "*.jpeg")) +
            glob.glob(os.path.join(folder_path, "*.bmp"))
        )

        good_count = 0

        for img_path in image_paths:
            clean_img = clean_chars74k_image_for_synthetic(img_path, folder)
            if clean_img is None:
                continue

            char_bank[true_label].append(clean_img)
            good_count += 1

        print(f"{folder} -> {true_label}: {good_count} usable cleaned samples")

    # sanity check
    for label in ALL_LABELS:
        if len(char_bank[label]) == 0:
            raise RuntimeError(f"No usable cleaned samples found for label: {label}")

    return char_bank


# =========================================================
# RANDOM TEXT
# =========================================================
def make_random_plate_string(length=7):
    return "".join(random.choice(ALL_LABELS) for _ in range(length))


# =========================================================
# STITCH CHARACTERS INTO ONE BINARY WORD/PLATE IMAGE
# =========================================================
def make_plate_image(text, char_bank):
    chars = []

    for ch in text:
        src = random.choice(char_bank[ch]).copy()
        src = resize_to_target_height(src, CHAR_TARGET_HEIGHT)
        chars.append(src)

    gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(len(chars) - 1)]

    total_w = LEFT_PAD + RIGHT_PAD + sum(c.shape[1] for c in chars) + sum(gaps)
    total_h = TOP_PAD + BOTTOM_PAD + CHAR_TARGET_HEIGHT

    plate = np.zeros((total_h, total_w), dtype=np.uint8)

    x = LEFT_PAD
    for i, c in enumerate(chars):
        h, w = c.shape
        y = TOP_PAD + (CHAR_TARGET_HEIGHT - h) // 2

        roi = plate[y:y+h, x:x+w]
        plate[y:y+h, x:x+w] = np.maximum(roi, c)

        x += w
        if i < len(gaps):
            x += gaps[i]

    return plate


# =========================================================
# OPTIONAL DEBUG PREVIEW
# =========================================================
def save_char_bank_preview(char_bank, out_dir, max_per_label=10):
    preview_dir = os.path.join(out_dir, "_char_bank_preview")
    os.makedirs(preview_dir, exist_ok=True)

    for label, imgs in char_bank.items():
        if not imgs:
            continue

        shown = imgs[:max_per_label]
        gap = 8
        h = max(img.shape[0] for img in shown)
        w = sum(img.shape[1] for img in shown) + gap * (len(shown) - 1)

        canvas = np.zeros((h, w), dtype=np.uint8)

        x = 0
        for img in shown:
            ih, iw = img.shape
            y = (h - ih) // 2
            canvas[y:y+ih, x:x+iw] = np.maximum(canvas[y:y+ih, x:x+iw], img)
            x += iw + gap

        cv2.imwrite(os.path.join(preview_dir, f"{label}.png"), canvas)


# =========================================================
# MAIN
# =========================================================
def main():
    print("Building cleaned character bank...")
    char_bank = build_clean_char_bank(ROOT)

    print("Saving preview of cleaned character bank...")
    save_char_bank_preview(char_bank, OUT_DIR, max_per_label=10)

    csv_path = os.path.join(OUT_DIR, "labels.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "text", "expected_count"])

        for i in range(NUM_IMAGES):
            text = make_random_plate_string(PLATE_LEN)
            plate_img = make_plate_image(text, char_bank)

            image_name = f"synthetic_{i:04d}.png"
            out_path = os.path.join(OUT_DIR, image_name)

            cv2.imwrite(out_path, plate_img)
            writer.writerow([image_name, text, PLATE_LEN])

            print(f"[{i+1}/{NUM_IMAGES}] {image_name} -> {text}")

    print("\nDONE")
    print(f"Synthetic plates saved to: {OUT_DIR}")
    print(f"Labels CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()