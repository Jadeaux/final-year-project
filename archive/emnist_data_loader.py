import numpy as np
from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import Subset
import string

def fix_emnist_orientation(img: np.ndarray) -> np.ndarray:
    return np.flipud(img.T)

def load_emnist_uppercase(train=True):
    """
    Loads EMNIST byclass and filters only uppercase A–Z.
    """
    ds = EMNIST(
        root="./data",
        split="byclass",
        train=train,
        download=True,
        transform=transforms.ToTensor()
    )

    uppercase_set = set(string.ascii_uppercase)

    # Find class IDs corresponding to A–Z
    keep_class_ids = [i for i, name in enumerate(ds.classes)
                      if name in uppercase_set]

    keep_class_ids_set = set(keep_class_ids)

    # Keep only samples whose label is uppercase
    keep_indices = [idx for idx, y in enumerate(ds.targets.tolist())
                    if int(y) in keep_class_ids_set]

    ds_upper = Subset(ds, keep_indices)

    return ds_upper, ds.classes

def get_sample(ds_subset, index, class_list):
    """
    Returns (img_fixed, label_char)
    """
    x, y = ds_subset[index]
    img = (x.squeeze(0).numpy() * 255).astype(np.uint8)
    img_fixed = fix_emnist_orientation(img)

    label_char = class_list[y]  # now returns 'A', 'B', etc.

    return img_fixed, label_char

def get_letter_sample(ds_upper, class_list, target_letter, occurrence=0):
    """
    Fetch the N-th occurrence of a specific uppercase letter.
    
    target_letter : str, e.g. 'X'
    occurrence    : which occurrence to return (0 = first)
    """
    count = 0

    for idx in range(len(ds_upper)):
        img, label = get_sample(ds_upper, idx, class_list)

        if label == target_letter:
            if count == occurrence:
                return img, label
            count += 1

    raise ValueError(f"Letter {target_letter} not found.")
    