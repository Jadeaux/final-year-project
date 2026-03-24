"""
Modules package for digit recognition.
Provides organized access to preprocessing, morphology, features, classification, and utilities.

To use different versions of functions, you can:
1. Import directly: from modules.preprocessing import preprocess_step1
2. Use version aliases (if versions are created)
3. Create version-specific files (e.g., preprocessing_v2.py) and import from there
"""

# Import main functions for easy access
from modules.preprocessing import preprocess_step1, thin
from modules.morphology import find_blobs, skeleton_neighbor_counts
from modules.features import get_stems, get_extreme_points, draw_line
from modules.classification import classify_group1, classify_group2, classify_with_blobs
from modules.utils import debug_group1_stems

__all__ = [
    # Preprocessing
    'preprocess_step1',
    'thin',
    # Morphology
    'find_blobs',
    'skeleton_neighbor_counts',
    # Features
    'get_stems',
    'get_extreme_points',
    'draw_line',
    # Classification
    'classify_group1',
    'classify_group2',
    'classify_with_blobs',
    # Utils
    'debug_group1_stems',
]

# Version management
# To add new versions, create files like preprocessing_v2.py, morphology_v2.py, etc.
# Then import and expose them here with version-specific names

