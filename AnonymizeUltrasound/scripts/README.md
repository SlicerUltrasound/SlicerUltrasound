# auto_anonymize.py – Command-line Ultrasound DICOM Anonymizer

Automated fan-masking and PHI removal for ultrasound DICOM files, replicating the 3D Slicer AnonymizeUltrasound extension functionality.

## Quick Start

Install dependencies:
```bash
uv venv --python 3.9.10
source .venv/bin/activate
uv pip install -r requirements-cpu.txt
```

Basic anonymization:
```bash
python -m auto_anonymize input_dicoms/ output_dicoms/ headers_out/ \
    --model-path model_trace.pt \
    --device cuda \
    --overview-dir overviews/
```

Header-only anonymization (no masking):
```bash
python -m auto_anonymize input_dicoms/ output_dicoms/ headers_out/ \
    --no-mask-generation
```

With ground truth evaluation:
```bash
python -m auto_anonymize input_dicoms/ output_dicoms/ headers_out/ \
    --model-path model_trace.pt \
    --ground-truth-dir ground_truth_masks/ \
    --overview-dir overviews/
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `input_folder` | ✓ | - | Root directory to scan for DICOM files (Modality == "US" only) |
| `output_folder` | ✓ | - | Directory for anonymized DICOMs |
| `headers_folder` | ✓ | - | Directory for headers/keys and the `keys.csv` mapping file |
| `--model-path` | * | - | Path to `.pt` checkpoint for corner prediction model |
| `--device` | | `cpu` | Inference device: `cpu`, `cuda`, or `mps` |
| `--skip-single-frame` | | off | Skip single-frame studies |
| `--no-hash-patient-id` | | off | **Dangerous**: Keep original PatientID instead of hashing |
| `--filename-prefix` | | - | Prefix for output filenames |
| `--no-preserve-directory-structure` | | off | Flatten output (don't mirror input tree) |
| `--resume-anonymization` | | off | Skip existing output files |
| `--overview-dir` | | - | Save before/after PNG comparisons for QC |
| `--no-mask-generation` | | off | Header-only anonymization (skip masking) |
| `--ground-truth-dir` | | - | Directory with ground truth masks for evaluation |

*Required unless using `--no-mask-generation`

## Outputs

| Location | Content |
|----------|---------|
| `output_folder/` | Anonymized DICOM files with fan masking applied |
| `headers_folder/keys.csv` | Mapping of original → anonymized filenames/UIDs |
| `headers_folder/*_DICOMHeader.json` | Anonymized header copies (PHI removed) |
| `overview_dir/` | Before/after PNG grids for manual QC |
| `overview_dir/metrics.csv` | Quantitative evaluation metrics (with `--ground-truth-dir`) |

## Evaluation & Quality Control

When `--ground-truth-dir` is specified, the script computes segmentation metrics by comparing predicted masks against ground truth configurations:

**Metrics included**: Dice coefficient, IoU, pixel accuracy, precision, recall, F1-score, sensitivity, specificity

**Ground truth format**: JSON files with mask configurations matching the anonymized filename structure.

## Processing Overview

1. **Directory scan**: Build index of ultrasound DICOM files
2. **Key generation**: Create `<hash(PatientID)>_<hash(SOPInstanceUID)>.dcm` filenames  
3. **Inference**: Predict corner points using Attention U-Net + DSNT model (unless `--no-mask-generation`)
4. **Masking**: Generate and apply fan-shaped masks to all frames
5. **DICOM assembly**: Re-encode with JPEG baseline, anonymize headers, shift dates
6. **Evaluation**: Compute metrics against ground truth (if provided)

**Anonymization details**:
- Patient name/ID cleared or hashed
- Birth date truncated to year only  
- Dates randomly shifted ≤30 days (consistent per patient)
- Fresh SeriesInstanceUID generated

## Logging & Resume

Logs saved to `auto_anonymize_*.log`. Use `--resume-anonymization` to skip completed files in interrupted runs.

**Exit codes**: 0 (success), 1 (≥1 failure)