import os
import csv
import cv2
import numpy as np

# =========================================================
# CONFIG
# =========================================================
DATASET_DIR = r"images/plates/chars74_k synthetic plates"
IMG_DIR = DATASET_DIR
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")

SHOW_DEBUG = False
SAVE_DEBUG = True
DEBUG_DIR = os.path.join(DATASET_DIR, "debug_segmentation")

VIS_DIR = os.path.join(DATASET_DIR, "segmentation_visuals")
os.makedirs(VIS_DIR, exist_ok=True)

os.makedirs(DEBUG_DIR, exist_ok=True)

# =========================================================
# HOLE FILL
# =========================================================
def fill_small_holes(bin255, max_hole_frac=0.05):
    """
    Fill small black holes inside white foreground objects.
    Input: 0/255 uint8
    Output: 0/255 uint8
    """
    A = (bin255 > 0).astype(np.uint8)

    h, w = A.shape
    flood = A.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    flood255 = (flood * 255).astype(np.uint8)
    cv2.floodFill(flood255, mask, (0, 0), 128)

    holes = (flood255 == 0).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)

    out = A.copy()
    total_fg = max(1, int(np.sum(A > 0)))

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area / total_fg <= max_hole_frac:
            out[labels == i] = 1

    return (out * 255).astype(np.uint8)

# =========================================================
# SEGMENTATION
# =========================================================
def segment_characters_from_word_mask(word_mask):
    img_bin = word_mask.copy()
    h, w = img_bin.shape

    num, labels, stats, _ = cv2.connectedComponentsWithStats(img_bin, connectivity=8)

    chars = []
    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 120:
            continue
        if ch < 0.45 * h:
            continue
        if cw < 5:
            continue

        chars.append((x, y, cw, ch, area))

    # -------------------------------
    # REAL segmentation count
    # -------------------------------
    raw_count = len(chars)

    # -------------------------------
    # Optional pipeline cap
    # Keep this only to inspect how your
    # normal pipeline behaves afterward
    # -------------------------------
    MAX_CHARS = 7
    capped_chars = chars.copy()

    if len(capped_chars) > MAX_CHARS:
        capped_chars = sorted(capped_chars, key=lambda c: c[4])   # smallest first
        capped_chars = capped_chars[len(capped_chars) - MAX_CHARS:]  # keep largest 7

    capped_chars = sorted(capped_chars, key=lambda c: c[0])

    chars_xywh = [(x, y, cw, ch) for (x, y, cw, ch, area) in capped_chars]
    final_count = len(chars_xywh)

    # ---------------------------------------
    # PROCESS CHARACTERS PROPERLY
    # ---------------------------------------
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    processed_chars = []

    for (x, y, cw, ch) in chars_xywh:
        char_img = img_bin[y:y+ch, x:x+cw].copy()

        char_img = fill_small_holes(char_img, max_hole_frac=0.05)
        char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, k1, iterations=1)
        char_img = cv2.dilate(char_img, k1, iterations=1)
        char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, k2, iterations=1)

        processed_chars.append(char_img)

    # ---------------------------------------
    # NORMALIZE PROCESSED CHARS
    # ---------------------------------------
    normalized_chars = []
    target_size = 96

    for char_img in processed_chars:
        ys, xs = np.where(char_img > 0)
        if xs.size == 0:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        char_img = char_img[y0:y1+1, x0:x1+1]

        hh, ww = char_img.shape
        size = max(hh, ww)
        square = np.zeros((size, size), dtype=np.uint8)

        y_off = (size - hh) // 2
        x_off = (size - ww) // 2
        square[y_off:y_off+hh, x_off:x_off+ww] = char_img

        resized = cv2.resize(square, (target_size, target_size),
                             interpolation=cv2.INTER_NEAREST)

        normalized_chars.append(resized)

    return chars_xywh, processed_chars, normalized_chars, raw_count, final_count

# =========================================================
# DEBUG DRAWING
# =========================================================
def draw_boxes(img_bin, boxes):
    vis = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(vis, str(i), (x, max(10, y-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def make_segmentation_visual(original, normalized_chars, boxes,
                             canvas_height=120, gap=10):
    """
    Creates a single image:
    [ original | char1 | char2 | ... ]

    original: grayscale plate image
    normalized_chars: list of 96x96 chars
    boxes: used only for count display
    """

    # resize original to match height
    h, w = original.shape
    scale = canvas_height / h
    new_w = int(w * scale)
    original_resized = cv2.resize(original, (new_w, canvas_height))

    # convert to BGR
    original_vis = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)

    # prepare character visuals
    char_imgs = []
    for i, c in enumerate(normalized_chars):
        c_vis = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        c_vis = cv2.resize(c_vis, (canvas_height, canvas_height))

        # label each char index
        cv2.putText(c_vis, str(i),
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

        char_imgs.append(c_vis)

    # compute total width
    total_w = original_vis.shape[1] + gap
    total_w += len(char_imgs) * (canvas_height + gap)

    canvas = np.zeros((canvas_height, total_w, 3), dtype=np.uint8)

    x = 0

    # place original
    canvas[:, x:x+original_vis.shape[1]] = original_vis
    x += original_vis.shape[1] + gap

    # place chars
    for c in char_imgs:
        canvas[:, x:x+canvas_height] = c
        x += canvas_height + gap

    return canvas

# =========================================================
# MAIN EVALUATION
# =========================================================
def main():
    rows = []
    total = 0
    raw_correct_count = 0
    final_correct_count = 0

    with open(LABELS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_name = row["image_name"]
            gt_text = row["text"]
            expected_count = int(row["expected_count"])

            path = os.path.join(IMG_DIR, image_name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARNING] Could not read {path}")
                continue

            boxes, processed_chars, normalized_chars, raw_count, final_count = \
                segment_characters_from_word_mask(img)

            # ---------------------------------------
            # SAVE VISUAL COMPARISON
            # ---------------------------------------
            vis_img = make_segmentation_visual(
                img,
                normalized_chars,
                boxes
            )

            vis_path = os.path.join(VIS_DIR, image_name)
            cv2.imwrite(vis_path, vis_img)

            raw_count_correct = int(raw_count == expected_count)
            final_count_correct = int(final_count == expected_count)

            total += 1
            raw_correct_count += raw_count_correct
            final_correct_count += final_count_correct

            rows.append({
                "image_name": image_name,
                "gt_text": gt_text,
                "expected_count": expected_count,
                "raw_count": raw_count,
                "final_count": final_count,
                "raw_count_correct": raw_count_correct,
                "final_count_correct": final_count_correct
            })

            print(
                f"{image_name}: "
                f"expected={expected_count}, "
                f"raw={raw_count}, "
                f"final={final_count}, "
                f"raw_correct={raw_count_correct}, "
                f"final_correct={final_count_correct}"
            )

            if SAVE_DEBUG:
                vis = draw_boxes(img, boxes)
                debug_path = os.path.join(DEBUG_DIR, image_name)
                cv2.imwrite(debug_path, vis)

            if SHOW_DEBUG:
                vis = draw_boxes(img, boxes)
                cv2.imshow("segmentation_debug", vis)
                key = cv2.waitKey(0)
                if key == 27:
                    break

    if SHOW_DEBUG:
        cv2.destroyAllWindows()

    out_csv = os.path.join(DATASET_DIR, "segmentation_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "gt_text",
                "expected_count",
                "raw_count",
                "final_count",
                "raw_count_correct",
                "final_count_correct"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    raw_acc = 100.0 * raw_correct_count / total if total > 0 else 0.0
    final_acc = 100.0 * final_correct_count / total if total > 0 else 0.0

    print("\n==============================")
    print(f"Total images tested:       {total}")
    print(f"Correct raw count (=7):    {raw_correct_count}")
    print(f"Raw count accuracy:        {raw_acc:.2f}%")
    print(f"Correct final count (=7):  {final_correct_count}")
    print(f"Final count accuracy:      {final_acc:.2f}%")
    print("==============================")
    print(f"Results CSV: {out_csv}")
    print(f"Debug images: {DEBUG_DIR}")

if __name__ == "__main__":
    main()