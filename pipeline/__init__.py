"""
Pipeline package for digit and letter recognition.
Provides organized access to preprocessing, morphology, features, and classification.

Usage:
    from pipeline.preprocessing import preprocess_step1
    from pipeline.classification import classify_with_blobs
"""

# Import main functions for easy access
from pipeline.preprocessing import preprocess_step1, thin
from pipeline.morphology import find_blobs, skeleton_neighbor_counts
from pipeline.features import get_stems, get_extreme_points, draw_line
from pipeline.classification import classify_group1, classify_group2, classify_with_blobs

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
]
