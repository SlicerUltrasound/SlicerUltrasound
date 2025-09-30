# AnonymizeUltrasound Scripts

Automated command-line tools for ultrasound DICOM anonymization, model evaluation, and batch processing.

---

## Table of Contents

1. [Setup & Installation](#setup--installation)
2. [Python Scripts](#python-scripts)
   - [auto_anonymize.py](#auto_anonymizepy)
   - [model_eval.py](#model_evalpy)
3. [Bash Scripts](#bash-scripts)
   - [batch_auto_anonymize.sh](#batch_auto_anonymizesh)
   - [batch_model_eval.sh](#batch_model_evalsh)
4. [Dependencies](#dependencies)
5. [Common Workflows](#common-workflows)

---

## Setup & Installation

### Prerequisites

- Python 3.9.10 or higher
- `uv` package manager (recommended) or `pip`
- Bash shell (for batch scripts)
- DICOM files with Modality == "US" (ultrasound)

### Installation Steps

1. **Navigate to the scripts directory:**
   ```bash
   cd /path/to/SlicerUltrasound/AnonymizeUltrasound/scripts
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Using uv (recommended)
   uv venv --python 3.9.10
   source .venv/bin/activate

   # Or using venv
   python3.9 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   **For CPU-only (recommended for most use cases):**
   ```bash
   uv pip install -r requirements-cpu.txt
   # Or: pip install -r requirements-cpu.txt
   ```

   **For GPU support (CUDA):**
   ```bash
   pip install -r requirements-gpu.txt
   ```

   **For Apple Silicon (M1/M2 Macs):**
   ```bash
   uv pip install -r requirements-cpu.txt
   # MPS (Metal Performance Shaders) supported via --device mps
   ```

4. **Verify installation:**
   ```bash
   python -m auto_anonymize --help
   python -m model_eval --help
   ```

---

## Python Scripts

### auto_anonymize.py

**Purpose:** Automated fan-masking and PHI removal for ultrasound DICOM files.

#### Basic Usage

```bash
python -m auto_anonymize \
    --input_dir <input_dicoms/> \
    --output_dir <output_dicoms/> \
    --headers_dir <headers_out/> \
    --model_path <path/to/model.pt> \
    --device cpu
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input_dir` | ✓ | - | Root directory to scan for DICOM files (Modality == "US" only) |
| `--output_dir` | ✓ | - | Directory for anonymized DICOMs |
| `--headers_dir` | ✓ | - | Directory for headers/keys and the `keys.csv` mapping file |
| `--model_path` | * | - | Path to `.pt` checkpoint for corner prediction model |
| `--device` | | `cpu` | Inference device: `cpu`, `cuda`, or `mps` |
| `--skip_single_frame` | | off | Skip single-frame studies |
| `--no_hash_patient_id` | | off | **Dangerous**: Keep original PatientID instead of hashing |
| `--filename_prefix` | | - | Prefix for output filenames |
| `--no_preserve_directory_structure` | | off | Flatten output (don't mirror input tree) |
| `--overwrite_files` | | off | Overwrite existing output files |
| `--overview_dir` | | - | Save before/after PNG comparisons for QC |
| `--no_mask_generation` | | off | Header-only anonymization (skip masking) |
| `--ground_truth_dir` | | - | Directory with ground truth masks for evaluation |
| `--phi_only_mode` | | off | Only anonymize the PHI part of the image |
| `--remove_phi_from_image` | | on | Remove the PHI part of the image |
| `--top_ratio` | | 0.1 | Ratio of top of image to anonymize |

*Required unless using `--no_mask_generation`

#### Examples

**1. Full anonymization with fan masking:**
```bash
python -m auto_anonymize \
    --input_dir /data/raw_dicoms/ \
    --output_dir /data/anonymized/ \
    --headers_dir /data/headers/ \
    --model_path ../Resources/checkpoints/baseline_unet_dsnt.pt \
    --device cpu \
    --overview_dir /data/overview/
```

**2. PHI-only mode with image redaction:**
```bash
python -m auto_anonymize \
    --input_dir /data/raw_dicoms/ \
    --output_dir /data/anonymized/ \
    --headers_dir /data/headers/ \
    --model_path ../Resources/checkpoints/baseline_unet_dsnt.pt \
    --device cpu \
    --phi_only_mode \
    --remove_phi_from_image
```

#### Outputs

| Location | Content |
|----------|---------|
| `output_dir/` | Anonymized DICOM files with fan masking applied |
| `headers_dir/keys.csv` | Mapping of original → anonymized filenames/UIDs |
| `headers_dir/*_DICOMHeader.json` | Anonymized header copies (PHI removed) |
| `overview_dir/` | Before/after PNG grids for manual QC |
| `overview_dir/metrics.csv` | Quantitative evaluation metrics (with `--ground_truth_dir`) |
| `logs/auto_anonymize_*.log` | Detailed processing logs |

#### Anonymization Details

- **Patient name/ID:** Cleared or hashed (SHA-256)
- **Birth date:** Truncated to year only
- **Dates:** Randomly shifted ≤30 days (consistent per patient)
- **UIDs:** Fresh SeriesInstanceUID generated
- **Encoding:** Re-encoded with JPEG baseline compression

#### Exit Codes

- `0`: Success
- `1`: One or more failures occurred

---

### model_eval.py

**Purpose:** Evaluate model performance on DICOM files with ground truth masks.

#### Basic Usage

```bash
python -m model_eval \
    --input_dir <input_dicoms/> \
    --ground_truth_dir <ground_truth_masks/> \
    --overview_dir <overviews/> \
    --model_path <path/to/model.pt> \
    --device cpu
```
#### Examples

**1. Basic model evaluation:**
```bash
python -m model_eval \
    --input_dir /data/test_dicoms/ \
    --ground_truth_dir /data/ground_truth/ \
    --overview_dir /data/eval_results/ \
    --model_path ../Resources/checkpoints/baseline_unet_dsnt.pt \
    --device cpu
```

**2. GPU-accelerated evaluation:**
```bash
python -m model_eval \
    --input_dir /data/test_dicoms/ \
    --ground_truth_dir /data/ground_truth/ \
    --overview_dir /data/eval_results/ \
    --model_path ../Resources/checkpoints/baseline_unet_dsnt.pt \
    --device cuda
```

#### Outputs

| Location | Content |
|----------|---------|
| `overview_dir/metrics_YYYYMMDD_HHMMSS.csv` | Detailed evaluation metrics per file |
| `overview_dir/*.png` | Visual comparisons of predictions vs ground truth |
| `logs/model_eval_*.log` | Detailed evaluation logs |
---

## Bash Scripts

### batch_auto_anonymize.sh

**Purpose:** Batch process nested DICOM directory structures using `auto_anonymize.py`.

#### Usage

```bash
./batch_auto_anonymize.sh <input_root_dir> <output_root_dir> <headers_root_dir> <mode> [options]
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `input_root_dir` | Root directory to scan for DICOM files |
| `output_root_dir` | Directory for anonymized DICOMs |
| `headers_root_dir` | Directory for headers/keys |
| `mode` | Anonymization mode: `full` or `phi-only` |

#### Optional Arguments

| Argument | Description |
|----------|-------------|
| `[overview_root_dir]` | Positional: Directory for QC overview images (backward compatible) |
| `--overview-dir <path>` | Named: Directory for QC overview images (only for `full` mode) |
| `--model-path <path>` | Path to model checkpoint file (`.pt`)<br>Default: `../Resources/checkpoints/baseline_unet_dsnt.pt` |
| `--script-dir <path>` | Path to scripts directory<br>Default: `../AnonymizeUltrasound/scripts` |

#### Modes

- **`full`**: Full anonymization with fan masking. Optionally generates before/after comparison images for QC.
- **`phi-only`**: Only anonymize PHI region at top of image. Uses `--phi_only_mode` and `--remove_phi_from_image` flags.

#### Features

- ✅ Automatically finds all directories containing DICOM files
- ✅ Preserves directory structure in output
- ✅ Flexible anonymization modes (full or PHI-only)
- ✅ Configurable model and script paths
- ✅ Colored progress output
- ✅ Memory usage tracking
- ✅ Error handling and summary reporting
- ✅ Interactive confirmation before processing

#### Examples

**1. Full anonymization with overview images (positional):**
```bash
./batch_auto_anonymize.sh \
    /data/input \
    /data/output \
    /data/headers \
    full \
    /data/overview
```

**2. Full anonymization with named arguments:**
```bash
./batch_auto_anonymize.sh \
    /data/input \
    /data/output \
    /data/headers \
    full \
    --overview-dir /data/overview
```

**3. PHI-only mode (default behavior):**
```bash
./batch_auto_anonymize.sh \
    /data/R21-Batch001-2Pts \
    /data/output \
    /data/headers \
    phi-only
```

**4. Full mode with custom model path:**
```bash
./batch_auto_anonymize.sh \
    /data/input \
    /data/output \
    /data/headers \
    full \
    --model-path /custom/models/v2_model.pt \
    --overview-dir /data/qc
```

**5. All custom paths:**
```bash
./batch_auto_anonymize.sh \
    /data/input \
    /data/output \
    /data/headers \
    full \
    --overview-dir /data/overview \
    --model-path /custom/model.pt \
    --script-dir /opt/scripts
```

#### How It Works

1. Scans the input root directory recursively
2. Identifies all leaf directories containing DICOM files (`.dcm` or `.dicom`)
3. Processes each directory individually with `auto_anonymize.py`
4. Applies the selected anonymization mode (full or PHI-only)
5. Maintains the same directory structure in output and headers locations
6. Reports success/failure for each directory
7. Provides final summary with total counts

#### Output Example

```bash
[INFO] === DICOM BATCH ANONYMIZATION SCRIPT ===
[INFO] Validation passed
[INFO] Preview of directories that will be processed:
  1. Patient001/Study001 (145 DICOM files)
  2. Patient001/Study002 (89 DICOM files)
...
[INFO] Total directories to process: 12
Do you want to proceed with processing? (y/N): y
[INFO] Processing: Patient001/Study001 (145 DICOM files) [Mode: full]
[INFO] Memory usage before: 124MB
[SUCCESS] Completed: Patient001/Study001 (45s, Memory: 124MB → 156MB)
...
[INFO] === BATCH PROCESSING SUMMARY ===
[INFO] Total directories processed: 12
[SUCCESS] Successful: 12
```

---

### batch_model_eval.sh

**Purpose:** Run model evaluation on multiple subdirectories separately to avoid memory issues.

#### Usage

```bash
./batch_model_eval.sh <parent_dir> <ground_truth_dir> <overview_dir> <model_path> <device> [options]
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `parent_dir` | Parent directory containing subdirectories with DICOM files |
| `ground_truth_dir` | Directory containing ground truth annotations |
| `overview_dir` | Directory to save evaluation overview images and metrics |
| `model_path` | Path to the model checkpoint (`.pt` file) |
| `device` | Device to use: `cpu`, `cuda`, or `mps` |

#### Optional Arguments

| Argument | Description |
|----------|-------------|
| `--script-dir <path>` | Path to scripts directory<br>Default: `../AnonymizeUltrasound/scripts` |

#### Features

- ✅ Processes each subdirectory independently
- ✅ Memory management with delays between batches
- ✅ DICOM file counting and validation
- ✅ Colored progress output with timing
- ✅ Comprehensive error reporting
- ✅ Interactive confirmation with preview
- ✅ Path validation before execution
- ✅ Configurable script directory

#### Examples

**1. Basic usage with default script directory:**
```bash
./batch_model_eval.sh \
    /data/output_001_R21 \
    /data/R21_anonymized \
    /data/eval_overview \
    /path/to/baseline_unet_dsnt.pt \
    mps
```

**2. With custom script directory:**
```bash
./batch_model_eval.sh \
    /data/output_001_R21 \
    /data/R21_anonymized \
    /data/eval_overview \
    /models/baseline_unet_dsnt.pt \
    cpu \
    --script-dir /custom/scripts
```

**3. Full example with Apple Silicon GPU:**
```bash
./batch_model_eval.sh \
    /Users/username/data/output_R21 \
    /Users/username/data/R21_ground_truth \
    /Users/username/data/evaluation_results \
    /Users/username/models/baseline_unet_dsnt.pt \
    mps \
    --script-dir /Users/username/SlicerUltrasound/AnonymizeUltrasound/scripts
```

#### How It Works

1. Finds all immediate subdirectories in `parent_dir` (depth 1)
2. Counts DICOM files in each subdirectory
3. Skips subdirectories with no DICOM files
4. Shows preview and asks for confirmation
5. Processes each subdirectory with `model_eval.py`
6. Creates subdirectory-specific overview folders
7. Adds 5-second delays between batches for memory management
8. Reports detailed timing and memory usage

#### Output Example

```bash
[INFO] === BATCH MODEL EVALUATION SCRIPT ===
[INFO] Validation passed
[INFO] Preview of subdirectories that will be processed:
  1. R21-Batch001-2Pts (290 DICOM files)
  2. R21-Batch002-3pt (435 DICOM files)
...
[INFO] Total subdirectories to process: 15
Do you want to proceed with evaluation? (y/N): y
[INFO] Starting batch model evaluation...
[INFO] Processing subdirectory: R21-Batch001-2Pts (290 DICOM files)
[INFO] Starting evaluation for R21-Batch001-2Pts...
[INFO] Memory before: 145MB
[SUCCESS] Completed: R21-Batch001-2Pts (67s, Memory: 145MB → 234MB)
[INFO] Waiting 5 seconds before next batch...
...
[INFO] === BATCH EVALUATION SUMMARY ===
[INFO] Total subdirectories processed: 15
[SUCCESS] Successful: 15
[SUCCESS] All evaluations completed successfully in 1h 23m 45s
```

#### Memory Management

The script includes several memory management features:

- 5-second delays between batches
- 2-second garbage collection pauses after each batch
- Memory usage tracking (before/after each batch)
- Per-subdirectory isolation prevents memory accumulation

---

## Dependencies

See the following files for complete dependency lists:

- **`requirements.txt`**: Base dependencies
- **`requirements-cpu.txt`**: CPU-only optimized (includes PyTorch CPU)
- **`requirements-gpu.txt`**: CUDA GPU support (includes PyTorch with CUDA)

### Key Dependencies

- `torch` (PyTorch): Deep learning framework for model inference
- `pydicom`: DICOM file reading/writing
- `numpy`: Numerical computations
- `Pillow`: Image processing
- `scikit-image`: Image evaluation metrics
- `scipy`: Scientific computing utilities

---

## Common Workflows

### Workflow 1: Full Batch Anonymization with QC

Process a large dataset with full fan masking and generate QC images:

```bash
# Step 1: Run batch anonymization in full mode
./batch_auto_anonymize.sh \
    /data/raw_dicoms \
    /data/anonymized \
    /data/headers \
    full \
    --overview-dir /data/qc_images

# Step 2: Review QC images in /data/qc_images
# Step 3: Check summary in console output
```

### Workflow 2: PHI-Only Batch Processing

Quick PHI removal for datasets where fan masking isn't needed:

```bash
./batch_auto_anonymize.sh \
    /data/raw_dicoms \
    /data/anonymized \
    /data/headers \
    phi-only
```

### Workflow 3: Model Evaluation Pipeline

Evaluate model performance on multiple batches:

```bash
# Step 1: Run batch evaluation
./batch_model_eval.sh \
    /data/test_batches \
    /data/ground_truth \
    /data/eval_results \
    /models/baseline_unet_dsnt.pt \
    mps

# Step 2: Review metrics in /data/eval_results/*/metrics_*.csv
# Step 3: Check visual comparisons in /data/eval_results/*/*.png
```

### Workflow 4: Custom Pipeline with Non-Standard Paths

Using custom model and script locations:

```bash
./batch_auto_anonymize.sh \
    /data/input \
    /data/output \
    /data/headers \
    full \
    --model-path /experiments/models/improved_model_v3.pt \
    --script-dir /opt/anonymize/scripts \
    --overview-dir /data/qc
```

### Workflow 5: Cross-System Portable Setup

For running on different machines or CI/CD:

```bash
# Define paths as variables for easy modification
MODEL_PATH="/path/to/model.pt"
SCRIPT_DIR="/path/to/scripts"
INPUT_DIR="/path/to/input"
OUTPUT_DIR="/path/to/output"
HEADERS_DIR="/path/to/headers"

# Run with explicit paths (no hardcoded values)
./batch_auto_anonymize.sh \
    "$INPUT_DIR" \
    "$OUTPUT_DIR" \
    "$HEADERS_DIR" \
    full \
    --model-path "$MODEL_PATH" \
    --script-dir "$SCRIPT_DIR"
```

---

## Troubleshooting

### Common Issues

**Issue**: Script directory not found
```bash
[ERROR] Script directory does not exist: /path/to/scripts
```
**Solution**: Use `--script-dir` to specify the correct path:
```bash
--script-dir /correct/path/to/SlicerUltrasound/AnonymizeUltrasound/scripts
```

**Issue**: Model file not found
```bash
[ERROR] Model file does not exist: /path/to/model.pt
```
**Solution**: Use `--model-path` to specify the correct model location:
```bash
--model-path /correct/path/to/baseline_unet_dsnt.pt
```

**Issue**: Memory errors during large batch processing
**Solution**: Process in smaller batches or use CPU instead of GPU:
```bash
# Use cpu device instead of mps/cuda
./batch_model_eval.sh ... cpu --script-dir /path/to/scripts
```

---

## Exit Codes

Both batch scripts use consistent exit codes:

- `0`: All processing completed successfully
- `1`: One or more failures occurred (details in console output)

---

## Logging

All Python scripts create detailed logs in the `logs/` directory:

- `logs/auto_anonymize_YYYYMMDD_HHMMSS.log`
- `logs/model_eval_YYYYMMDD_HHMMSS.log`

Bash scripts output colored logs directly to console with timestamps and memory usage tracking.

---

**Last Updated**: 2025-09-30