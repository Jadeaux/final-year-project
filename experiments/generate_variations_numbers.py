import os
import cv2
import numpy as np

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def center_crop(img, out_h, out_w):
    h, w = img.shape[:2]
    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    return img[y0:y0+out_h, x0:x0+out_w]

def pad_center(img, pad_frac=0.25):
    """Pad equally on all sides so geometry transforms don't clip."""
    h, w = img.shape[:2]
    pad_x = int(pad_frac * w)
    pad_y = int(pad_frac * h)
    return cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)

def random_perspective_safe(img, max_warp=0.03):
    """
    Mild perspective warp on a padded canvas, then crop back later.
    max_warp is relative to width/height.
    """
    h, w = img.shape[:2]
    src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])

    dx = max_warp * w
    dy = max_warp * h

    # Keep the warp mild so the glyph stays inside the padded canvas
    dst = np.float32([
        [np.random.uniform(-dx, dx), np.random.uniform(-dy, dy)],
        [np.random.uniform(w-1-dx, w-1+dx), np.random.uniform(-dy, dy)],
        [np.random.uniform(w-1-dx, w-1+dx), np.random.uniform(h-1-dy, h-1+dy)],
        [np.random.uniform(-dx, dx), np.random.uniform(h-1-dy, h-1+dy)]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def random_rotate_scale_centered(img, max_deg=3, scale_range=(0.75, 1.0)):
    """
    Rotate + scale around center. Scale is <= 1.0 to avoid cropping.
    """
    h, w = img.shape[:2]
    angle = np.random.uniform(-max_deg, max_deg)
    scale = np.random.uniform(scale_range[0], scale_range[1])

    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def random_brightness_contrast(img):
    alpha = np.random.uniform(0.80, 1.25)   # contrast
    beta  = np.random.uniform(-20, 20)      # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def add_gradient_lighting(img):
    h, w = img.shape[:2]
    gx = np.linspace(0, 1, w)[None, :]
    gy = np.linspace(0, 1, h)[:, None]
    g = 0.6 * gx + 0.4 * gy
    strength = np.random.uniform(-35, 35)
    mask = (g * strength).astype(np.float32)

    if img.ndim == 2:
        return np.clip(img.astype(np.float32) + mask, 0, 255).astype(np.uint8)

    out = img.astype(np.float32)
    out[..., 0] = np.clip(out[..., 0] + mask, 0, 255)
    out[..., 1] = np.clip(out[..., 1] + mask, 0, 255)
    out[..., 2] = np.clip(out[..., 2] + mask, 0, 255)
    return out.astype(np.uint8)

def random_blur(img):
    if np.random.rand() < 0.6:
        k = np.random.choice([1, 3])
        return cv2.GaussianBlur(img, (k, k), 0)
    # mild motion-ish blur
    k = np.random.choice([3, 5, 7])
    kernel = np.zeros((k, k), dtype=np.float32)
    if np.random.rand() < 0.5:
        kernel[k//2, :] = 1.0
    else:
        kernel[:, k//2] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)

def random_morph(img):
    op = np.random.choice(["none", "erode", "dilate", "open", "close"],
                          p=[0.45, 0.15, 0.15, 0.125, 0.125])
    if op == "none":
        return img
    k = np.random.choice([2, 3])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    iters = np.random.choice([1, 2])

    if op == "erode":
        return cv2.erode(img, kernel, iterations=iters)
    if op == "dilate":
        return cv2.dilate(img, kernel, iterations=iters)
    if op == "open":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iters)
    if op == "close":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iters)


def make_variation_centered(base):
    H, W = base.shape[:2]

    # 1) pad equally so transforms never clip
    img = pad_center(base, pad_frac=0.30)

    # 2) geometry but keep scale <= 1.0 (only smaller)
    img = random_rotate_scale_centered(img, max_deg=3, scale_range=(0.75, 1.0))
    if np.random.rand() < 0.6:
        img = random_perspective_safe(img, max_warp=0.03)

    # 3) crop back to original size (keeps it centered)
    img = center_crop(img, H, W)

    # 4) lighting + blur + artifacts
    img = random_brightness_contrast(img)
    if np.random.rand() < 0.6:
        img = add_gradient_lighting(img)

    if np.random.rand() < 0.7:
        img = random_blur(img)

    if np.random.rand() < 0.7:
        img = random_morph(img)


    return img

def generate_variations(input_path, out_dir, n=200):
    ensure_dir(out_dir)
    base = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    for i in range(n):
        var = make_variation_centered(base)
        cv2.imwrite(os.path.join(out_dir, f"A_{i:03d}.png"), var)



import os
import glob
import cv2

# --- keep all your augmentation functions as-is above this line ---
# (ensure_dir, make_variation_centered, etc.)

def generate_variations_for_image(input_path, out_dir, n=200):
    ensure_dir(out_dir)
    base = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    label = os.path.splitext(os.path.basename(input_path))[0]  # e.g. "letterA"
    # turn "letterA" -> "A" (fallback to full name if it doesn't match)
    if label.lower().startswith("letter") and len(label) > 6:
        label = label[6:]

    for i in range(n):
        var = make_variation_centered(base)
        cv2.imwrite(os.path.join(out_dir, f"{label}_{i:03d}.png"), var)

def generate_all_numbers(templates_dir="images/numbers",
                         out_root="images/variations_numbers",
                         n=200):
    ensure_dir(out_root)

    # grabs number0.png, number1.png, ... (also works for digits if named number0.png etc)
    paths = sorted(glob.glob(os.path.join(templates_dir, "number*.png")))
    if not paths:
        raise FileNotFoundError(f"No templates found in {templates_dir} matching letter*.png")

    print(f"Found {len(paths)} templates.")
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]  # letterA
        label = name[6:] if name.lower().startswith("letter") else name

        out_dir = os.path.join(out_root, label)  # images/variations/A
        print(f"Generating {n} for {label} -> {out_dir}")
        generate_variations_for_image(p, out_dir, n=n)

    print("Done: generated variations for all templates.")

if __name__ == "__main__":
     generate_all_numbers(
         templates_dir="images/numbers",
         out_root="images/variations_numbers",
         n=200
     )