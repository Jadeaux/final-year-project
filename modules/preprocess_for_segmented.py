import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_letters(img, visualize=True, plate_mode=False):
    """
    Clean topology-safe preprocessing for letters.
    plate_mode=True -> much lighter morphology for segmented plate chars.
    """

    # 1) Grayscale
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.astype(np.uint8)

    # 2) Slight blur (stabilizes skeleton)
    # For plate chars, blur can hurt edges, so reduce it.
    if plate_mode:
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    else:
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # 3) Otsu binarize
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4) Polarity normalize using border
    border = np.concatenate([binary[0,:], binary[-1,:], binary[:,0], binary[:,-1]])
    if np.mean(border == 255) > 0.5:
        binary = cv2.bitwise_not(binary)

    binary_norm = binary.copy()

    # 5) Light close to connect micro-gaps
    cleaned = binary_norm

    # 6) Keep largest CC (OPTIONAL for plate chars)
    if not plate_mode:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = 1 + int(np.argmax(areas))
            mask = np.zeros_like(cleaned)
            mask[labels == largest_idx] = 255
            cleaned = mask

    # 6.5) THIS is what breaks plate chars -> disable or shrink in plate_mode
    if plate_mode:
        # very tiny cleanup only
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k, iterations=1)
    else:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  k, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k, iterations=1)

    # 7) Add small border
    cleaned = cv2.copyMakeBorder(cleaned, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

    # 8) 0/1
    A = (cleaned == 255).astype(np.uint8)

    if visualize:
        plt.figure(figsize=(12, 2.5))
        plt.subplot(1, 4, 1); plt.imshow(img_gray, cmap='gray'); plt.title("Original"); plt.axis("off")
        plt.subplot(1, 4, 2); plt.imshow(binary_norm, cmap='gray'); plt.title("Binary"); plt.axis("off")
        plt.subplot(1, 4, 3); plt.imshow(cleaned, cmap='gray'); plt.title("Cleaned"); plt.axis("off")
        plt.subplot(1, 4, 4); plt.imshow(A, cmap='gray'); plt.title("Final 0/1"); plt.axis("off")
        plt.tight_layout(); plt.show()

    return A, cleaned, binary_norm