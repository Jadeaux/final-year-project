# Morphological Character Recognition & License Plate Recognition

Rule-based character recognition system using topological and morphological features (holes, stems, helper lines, symmetry, concavity). Classifies digits (0-9) and uppercase letters (A-Z) without trained ML models. Includes a full license plate recognition (LPR) pipeline with character segmentation.

## Repository Structure

```
project/
├── pipeline/              # Core recognition modules (importable package)
│   ├── preprocessing.py           # Image preprocessing, binarization, skeletonization
│   ├── preprocessing_chars.py     # Character-specific preprocessing
│   ├── preprocess_for_segmented.py # Preprocessing for segmented plate characters
│   ├── morphology.py              # Blob (hole) detection, skeleton neighbor counting
│   ├── morphology_chars.py        # Character morphology variant
│   ├── features.py                # Stem detection, extreme points, helper lines
│   ├── features_letters.py        # Letter features (holes, endpoints, symmetry, strokes, concavity)
│   ├── classification.py          # Digit classifier (Group 1: with holes, Group 2: without)
│   ├── classification_kumar.py    # Kumar digit classifier variant
│   ├── classification_letters_california.py    # Letter classifier (California plates)
│   ├── classification_letters_CRAMM.py         # Letter classifier (CRAMM method)
│   ├── classification_letters_california_synthetictest.py  # Letter classifier tuned on synthetic data
│   ├── classification_data.py     # Data-tuned digit classifier
│   ├── classification_emnist.py   # EMNIST digit classifier variant
│   ├── license_plate.py           # Main LPR pipeline (California plates)
│   └── license_plate_cars.py      # Car-specific LPR pipeline
│
├── evaluation/            # Accuracy evaluation scripts
│   ├── batch_evaluate_plates.py       # Batch plate evaluation with confusion matrices
│   ├── chars74k.py                    # Chars74K dataset evaluation
│   ├── chars74k_runall.py             # Full Chars74K evaluation harness
│   ├── evaluate_chars74k.py           # Detailed digit trace evaluation
│   ├── evaluate_features_chars.py     # Letter feature evaluation
│   ├── run_letters_license_data.py    # Letter evaluation on plate data
│   └── run_numbers_license_data.py    # Digit evaluation with rule-path instrumentation
│
├── experiments/           # Experimental pipelines and tools
│   ├── license_plate_licensestest.py      # License plate test pipeline
│   ├── license_plate_datasettest.py       # Dataset test pipeline
│   ├── license_plate_cars_features.py     # Feature debug visualization for cars
│   ├── make_synthetic_plate_dataset.py    # Synthetic plate dataset generator
│   ├── test_segmentation_on_synthetic.py  # Segmentation testing on synthetic data
│   ├── generate_variations.py             # Letter image augmentation
│   └── generate_variations_numbers.py     # Digit image augmentation
│
├── archive/               # Old/debug scripts from development
├── audit_outputs/         # CSV feature analysis results
├── lpr_outputs/           # Recognition accuracy results and confusion matrices
└── results_chars74k/      # Chars74K evaluation results
```

## Dependencies

```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib pandas sympy
```

## Pipeline Architecture

```
Image Input
  |
  v
Preprocessing (grayscale -> Otsu binarization -> morphological ops -> crop -> scale)
  |
  v
Skeletonization (scikit-image thinning)
  |
  v
Feature Extraction (blobs, stems, helper lines, symmetry, concavity)
  |
  v
Group Decision (has holes? -> Group 1 or Group 2)
  |
  v
Classification (rule-based using morphological features)
  |
  v
Digit / Letter Output
```

**Digit classification:**
- **Group 1** (digits with holes: 0, 4, 6, 8, 9) — blob count + stem detection
- **Group 2** (digits without holes: 1, 2, 3, 5, 7) — helper lines + morphological features

**Letter classification** uses hole count, endpoints, horizontal/vertical strokes, symmetry balance, concavity, and width ratios.

## Usage

```python
from pipeline import preprocess_step1, thin, find_blobs, classify_with_blobs

# Classify a digit image
A, cropped, _ = preprocess_step1(img, visualize=True)
A_thin = thin(A)
pred, group = classify_with_blobs(img, visualize=True)
```

```python
from pipeline.classification_letters_california import classify_letter

# Classify a letter image
label = classify_letter(preprocessed_img)
```

## Running Evaluation Scripts

```bash
python evaluation/chars74k.py
python evaluation/chars74k_runall.py
python evaluation/batch_evaluate_plates.py
python evaluation/run_numbers_license_data.py
python evaluation/run_letters_license_data.py
```

Results are written as CSV files to `lpr_outputs/` and `audit_outputs/`.
