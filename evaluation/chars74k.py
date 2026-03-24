import os
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os
# =========================
# IMPORT YOUR FUNCTIONS
# =========================
from pipeline.preprocessing import preprocess_step1, preprocess_letters, thin
from pipeline.classification import classify_with_blobs_from_A
from pipeline.classification_letters_california import classify_letter
import matplotlib.pyplot as plt


ROOT = r"E:\EnglishImg (1)\English\Img\GoodImg\Bmp"

def is_digit_folder(folder_name):
    """
    Sample001 to Sample010 -> digits
    Sample011 onward       -> letters
    """
    idx = int(folder_name.replace("Sample", ""))
    return 1 <= idx <= 10


ROOT = r"E:\EnglishImg (1)\English\Img\GoodImg\Bmp"

sample_folder = "Sample014"   # this is A, so letter
folder_path = os.path.join(ROOT, sample_folder)

image_paths = glob.glob(os.path.join(folder_path, "*.png")) + \
              glob.glob(os.path.join(folder_path, "*.jpg")) + \
              glob.glob(os.path.join(folder_path, "*.bmp"))

print(f"Found {len(image_paths)} images in {sample_folder}")

img_path = image_paths[0]
print("Using:", img_path)

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(4, 4))
plt.imshow(img, cmap="gray")
plt.title(f"Raw image: {sample_folder}")
plt.axis("off")
plt.show()

if is_digit_folder(sample_folder):
    print("Routing to DIGIT pipeline")
    
    A, cleaned, binary_norm = preprocess_step1(img, visualize=True)
    A_thin = thin(A)

    plt.figure(figsize=(4, 4))
    plt.imshow(A_thin, cmap="gray")
    plt.title("Digit skeleton")
    plt.axis("off")
    plt.show()

else:
    print("Routing to LETTER pipeline")
    
    A, cleaned, binary_norm = preprocess_letters(img, visualize=True)
    A_thin = thin(A)

    plt.figure(figsize=(4, 4))
    plt.imshow(A_thin, cmap="gray")
    plt.title("Letter skeleton")
    plt.axis("off")
    plt.show()

if is_digit_folder(sample_folder):
    pred = classify_with_blobs_from_A(A)
else:
    pred = classify_letter(A, A_thin)

print("Prediction:", pred)



def predict_chars74k_image(img_path, sample_folder, visualize=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "ERROR"

    try:
        if is_digit_folder(sample_folder):
            A, cleaned, binary_norm = preprocess_step1(img, visualize=visualize)
            pred = classify_with_blobs_from_A(A)
        else:
            A, cleaned, binary_norm = preprocess_letters(img, visualize=visualize)
            A_thin = thin(A)
            pred = classify_letter(A, A_thin)

        if visualize:
            plt.figure(figsize=(4, 4))
            plt.imshow(A, cmap="gray")
            plt.title(f"Final A | Pred={pred}")
            plt.axis("off")
            plt.show()

        return str(pred) if pred is not None else "ERROR"

    except Exception as e:
        print(f"Error on {img_path}: {e}")
        return "ERROR"
    
