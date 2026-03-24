# Modules Structure

This directory contains the modularized code from `demo.py`, organized by functionality to allow easy testing of multiple versions.

## Structure

```
modules/
├── __init__.py              # Package initialization, exports main functions
├── preprocessing.py         # Image preprocessing (binarization, morphology, cropping)
├── morphology.py            # Morphological operations (blob detection, neighbor counting)
├── features.py              # Feature extraction (stems, extreme points, line drawing)
├── classification.py        # Digit classification (Group 1, Group 2, main classifier)
├── utils.py                 # Utility functions (debugging, evaluation)
├── preprocessing_v2_example.py  # Example of version 2 (template for your experiments)
├── README.md                # This file
└── VERSION_GUIDE.md         # Guide for managing multiple versions
```

## Module Descriptions

### preprocessing.py
- `preprocess_step1()`: Main preprocessing pipeline (grayscale, binarize, morphology, crop)
- `thin()`: Skeletonization/thinning of binary images

### morphology.py
- `find_blobs()`: Detect internal holes (blobs) in digit images
- `skeleton_neighbor_counts()`: Count 8-connected neighbors for skeleton pixels

### features.py
- `get_stems()`: Detect stems (open branches) in thinned digits
- `get_extreme_points()`: Get corner coordinates of foreground
- `draw_line()`: Draw helper lines for classification

### classification.py
- `classify_group1()`: Classify digits with blobs (0, 4, 6, 8, 9)
- `classify_group2()`: Classify digits without blobs (1, 2, 3, 5, 7)
- `classify_with_blobs()`: Main classification function

### utils.py
- `debug_group1_stems()`: Debugging utility for stem detection analysis

## Usage

### Basic Usage

```python
from modules import preprocess_step1, thin, find_blobs, classify_with_blobs

# Process an image
A, cropped, _ = preprocess_step1(img, visualize=True)
A_thin = thin(A)
blobs, n_blobs = find_blobs(A_thin)
pred, group = classify_with_blobs(img, visualize=True)
```

### Testing Multiple Versions

```python
from modules.preprocessing import preprocess_step1 as preprocess_v1
from modules.preprocessing_v2_example import preprocess_step1 as preprocess_v2

# Compare versions
result_v1 = preprocess_v1(img)
result_v2 = preprocess_v2(img)
```

### Import Specific Modules

```python
from modules.preprocessing import preprocess_step1, thin
from modules.morphology import find_blobs
from modules.classification import classify_with_blobs
```

## Creating New Versions

See `VERSION_GUIDE.md` for detailed instructions on creating and testing multiple versions of functions.

## Benefits

1. **Organization**: Code is split by functionality, making it easier to understand and maintain
2. **Version Testing**: Easy to test multiple implementations side-by-side
3. **Modularity**: Each module can be imported independently
4. **Extensibility**: Simple to add new versions or modify existing ones

