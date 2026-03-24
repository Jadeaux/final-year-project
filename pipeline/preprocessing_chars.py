"""
Preprocessing module for digit recognition.
Handles image preprocessing, binarization, and skeletonization.
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.morphology import skeletonize


def preprocess_step1(img, visualize=True, min_size=100, max_size=200):
    """
    STEP 1: Preprocessing for morphological digit recognition

    1. Ensure grayscale
    2. Binarize (Otsu)
    3. Normalize polarity so: digit = white (255), background = black (0)
    4. Morphological closing (connect small gaps)
    5. Morphological opening (remove tiny noise)
    6. Crop to bounding box of the digit
    7. Scale up image for better line drawing accuracy
    8. Return:
        - A: cleaned binary image as 0/1 (foreground=1)
        - cropped: cleaned 0/255 image (for display)
        - binary_norm: normalized 0/255 binary before morph ops (for display)

    Args:
        min_size: Minimum dimension (width or height) after scaling. Images smaller
                  than this will be scaled up. Default: 100
        max_size: Maximum dimension (width or height) after scaling. Images larger
                  than this will be scaled down. Default: 200
    """

    # Grayscale
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.astype(np.uint8)

    # Convert to binary
    # Otsu finds an optimal cutoff to separate foreground and background based on the histogram.
    _, binary = cv2.threshold(
        img_gray,
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3)  we want digit = white (255), background = black (0).
    white = np.sum(binary == 255)
    black = np.sum(binary == 0)
    border = np.concatenate([
    binary[0, :],
    binary[-1, :],
    binary[:, 0],
    binary[:, -1]
    ])

    # if border is mostly white, that means background is white,
    # so invert to make digit white on black
    if np.mean(border == 255) > 0.5:
        binary = cv2.bitwise_not(binary)

    binary_norm = binary.copy()
    
    # morphological closing: connect small gaps in strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # morphological opening: remove small isolated noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # crop to make it full and easier to draw lines later
    ys, xs = np.where(opened == 255)
    if len(xs) > 0:
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        cropped = opened[y0:y1, x0:x1]
    else:
        cropped = opened.copy()  # no foreground

    # Scale image for better line drawing accuracy
    # Maintain aspect ratio while anensuring size is within [min_size, max_size]
    h, w = cropped.shape[:2]
    """ 
    if h > 0 and w > 0:
        # Determine target size based on the larger dimension
        current_max_dim = max(h, w)
        
        if current_max_dim < min_size:
            # Scale up if too small
            scale = min_size / current_max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
        elif current_max_dim > max_size:
            # Scale down if too large
            scale = max_size / current_max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            # No scaling needed
            new_w, new_h = w, h
        
        # Resize using INTER_NEAREST to preserve binary nature
        if new_w != w or new_h != h:
            cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
 """
    # Convert to 0/1 for morphology math
    A = (cropped == 255).astype(np.uint8)

    if visualize:
        plt.figure(figsize=(12, 2.5))

        plt.subplot(1,5,1)
        plt.imshow(img_gray, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1,5,2)
        plt.imshow(binary_norm, cmap='gray')
        plt.title('Binary (norm)')
        plt.axis('off')

        plt.subplot(1,5,3)
        plt.imshow(opened, cmap='gray')
        plt.title('After Morphology')
        plt.axis('off')

        plt.subplot(1,5,4)
        plt.imshow(cropped, cmap='gray')
        plt.title(f'Cropped & Scaled\n{cropped.shape[1]}x{cropped.shape[0]}')
        plt.axis('off')

        plt.subplot(1,5,5)
        plt.imshow(A, cmap='gray')
        plt.title('Final Binary (0/1)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return A, cropped, binary_norm

import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_letters(img, visualize=True):
    """
    Clean topology-safe preprocessing for letters.
    NO resizing.
    NO aggressive morphology.
    """

    # 1) Grayscale
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.astype(np.uint8)

    # 2) Slight blur (stabilizes skeleton)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # 3) Otsu binarize
    _, binary = cv2.threshold(
        img_gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4) Polarity normalize
    # decide polarity by looking at image border
    border = np.concatenate([binary[0,:], binary[-1,:], binary[:,0], binary[:,-1]])
    if np.mean(border == 255) > 0.5:
        binary = cv2.bitwise_not(binary)

    binary_norm = binary.copy()

    # 5) Light 2x2 close at native scale
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # 6) Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        mask = np.zeros_like(cleaned)
        mask[labels == largest_idx] = 255
        cleaned = mask

    # 6.5) Smooth contour: remove tiny burrs then reconnect tiny nicks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  k, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k, iterations=1)


    # 7) Add small border (prevents edge skeleton artifacts)
    cleaned = cv2.copyMakeBorder(
        cleaned, 2, 2, 2, 2,
        cv2.BORDER_CONSTANT, value=0
    )

    # 8) Convert to 0/1
    A = (cleaned == 255).astype(np.uint8)

    if visualize:
        plt.figure(figsize=(12, 2.5))

        plt.subplot(1, 4, 1)
        plt.imshow(img_gray, cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(binary_norm, cmap='gray')
        plt.title("Binary")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(cleaned, cmap='gray')
        plt.title("Cleaned")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(A, cmap='gray')
        plt.title("Final 0/1")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return A, cleaned, binary_norm

def thin(A):
    """
    Thinning (skeletonization) of the cleaned 0/1 image.

    Input:
        A : np.ndarray of 0/1, foreground=1
    Output:
        skel : np.ndarray of 0/1, one-pixel-thin skeleton
    """
    skel = skeletonize(A.astype(bool))   
    return skel.astype(np.uint8)


