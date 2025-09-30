#!/bin/bash

# batch_model_eval.sh
# Script to run model_eval.py on each subdirectory separately to avoid memory issues
# Usage: ./batch_model_eval.sh <parent_dir> <ground_truth_dir> <overview_dir> <model_path> <device> [options]

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables set by parse_arguments
PARENT_DIR=""
GROUND_TRUTH_DIR=""
OVERVIEW_DIR=""
MODEL_PATH=""
DEVICE=""
SCRIPT_DIR=""

# Function to print colored output
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << 'EOF'
Usage: ./batch_model_eval.sh <parent_dir> <ground_truth_dir> <overview_dir> <model_path> <device> [options]

Required Arguments:
  parent_dir        Parent directory containing subdirectories with DICOM files
  ground_truth_dir  Directory containing ground truth annotations
  overview_dir      Directory to save evaluation overview images and metrics
  model_path        Path to the model checkpoint (.pt file)
  device            Device to use: cpu, cuda, or mps

Optional Arguments:
  --script-dir <path>    Path to scripts directory
                         Default: ~/path/to/scripts

Examples:
  # Basic usage with default script directory
  ./batch_model_eval.sh \
     ~/path/to/output_001_R21 \
     ~/path/to/dropbox_data/anonymized/R21_anonymized \
     ~/path/to/demo/overview \
     ~/path/to/baseline_unet_dsnt.pt \
     mps

  # With custom script directory
  ./batch_model_eval.sh \
     /data/output_001_R21 \
     /data/R21_anonymized \
     /data/eval_overview \
     /path/to/baseline_unet_dsnt.pt \
     cpu \
     --script-dir /custom/scripts

This script will:
  1. Find all immediate subdirectories in parent_dir (e.g., R21-Batch001-2Pts, R21-Batch002-3pt)
  2. Run model_eval.py on each subdirectory separately
  3. Save results to subdirectory-specific folders in overview_dir
  4. Add delays between batches to help with memory management
  5. Report success/failure for each subdirectory

Benefits:
  - Processes each subdirectory independently to avoid memory accumulation
  - Provides clear progress tracking for each batch
  - Isolates failures to individual subdirectories
  - Allows resuming from a specific point if needed
EOF
}

# Function to parse arguments
parse_arguments() {
    # Required positional arguments
    if [ $# -lt 5 ]; then
        show_usage
        exit 1
    fi

    PARENT_DIR="$1"
    GROUND_TRUTH_DIR="$2"
    OVERVIEW_DIR="$3"
    MODEL_PATH="$4"
    DEVICE="$5"

    # Parse optional named arguments
    shift 5

    while [ $# -gt 0 ]; do
        case "$1" in
            --script-dir)
                if [ $# -lt 2 ]; then
                    log_error "--script-dir requires a value"
                    exit 1
                fi
                SCRIPT_DIR="$2"
                shift 2
                ;;
            -*)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                log_error "Unknown argument: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Function to count DICOM files recursively in a directory
count_dicom_files() {
    local dir="$1"
    find "$dir" -type f \( -iname "*.dcm" -o -iname "*.dicom" \) | wc -l
}

# Function to get memory usage in MB
get_memory_usage() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - show Python process memory if exists
        ps aux | grep "python.*model_eval" | grep -v grep | awk '{print int($6/1024)}' | head -1 || echo "0"
    else
        # Linux
        ps aux | grep "python.*model_eval" | grep -v grep | awk '{print int($6/1024)}' | head -1 || echo "0"
    fi
}

# Function to process a single subdirectory
process_subdirectory() {
    local input_dir="$1"
    local ground_truth_dir="$2"
    local overview_base_dir="$3"
    local model_path="$4"
    local device="$5"
    local subdir_name="$6"
    local script_dir="$7"

    local dicom_count=$(count_dicom_files "$input_dir")

    log_info "Processing subdirectory: $subdir_name ($dicom_count DICOM files)"

    # Create subdirectory-specific overview directory
    local overview_dir="$overview_base_dir/$subdir_name"
    mkdir -p "$overview_dir"

    # Run the model evaluation
    local start_time=$(date +%s)
    local mem_before=$(get_memory_usage)

    log_info "Starting evaluation for $subdir_name..."
    log_info "Memory before: ${mem_before}MB"

    if cd "$script_dir" && python -m model_eval \
        --input_dir "$input_dir" \
        --ground_truth_dir "$ground_truth_dir" \
        --overview_dir "$overview_dir" \
        --model_path "$model_path" \
        --device "$device"; then

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local mem_after=$(get_memory_usage)

        log_success "Completed: $subdir_name (${duration}s, Memory: ${mem_before}MB → ${mem_after}MB)"

        # Force garbage collection by adding a small delay
        sleep 2

        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "Failed: $subdir_name (${duration}s)"
        return 1
    fi
}

# Function to process all subdirectories
process_batch() {
    local parent_dir="$1"
    local ground_truth_dir="$2"
    local overview_base_dir="$3"
    local model_path="$4"
    local device="$5"
    local script_dir="$6"

    local total_processed=0
    local total_success=0
    local total_failed=0
    local failed_dirs=()

    log_info "Starting batch model evaluation..."
    log_info "Parent directory: $parent_dir"
    log_info "Ground truth: $ground_truth_dir"
    log_info "Overview output: $overview_base_dir"
    log_info "Model: $model_path"
    log_info "Device: $device"
    log_info "Script directory: $script_dir"

    # Find all immediate subdirectories (depth 1)
    while IFS= read -r -d '' subdir; do
        local subdir_name=$(basename "$subdir")
        local dicom_count=$(count_dicom_files "$subdir")

        # Skip if no DICOM files
        if [ "$dicom_count" -eq 0 ]; then
            log_warning "Skipping $subdir_name (no DICOM files found)"
            continue
        fi

        total_processed=$((total_processed + 1))

        if process_subdirectory "$subdir" "$ground_truth_dir" "$overview_base_dir" "$model_path" "$device" "$subdir_name" "$script_dir"; then
            total_success=$((total_success + 1))
        else
            total_failed=$((total_failed + 1))
            failed_dirs+=("$subdir_name")
        fi

        # Add delay between batches to help with memory management
        if [ $total_processed -lt $(find "$parent_dir" -mindepth 1 -maxdepth 1 -type d | wc -l) ]; then
            log_info "Waiting 5 seconds before next batch..."
            sleep 5
        fi

    done < <(find "$parent_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

    # Print summary
    echo
    log_info "=== BATCH EVALUATION SUMMARY ==="
    log_info "Total subdirectories processed: $total_processed"
    log_success "Successful: $total_success"

    if [ $total_failed -gt 0 ]; then
        log_error "Failed: $total_failed"
        log_error "Failed subdirectories:"
        for dir in "${failed_dirs[@]}"; do
            log_error "  - $dir"
        done
    fi

    return $total_failed
}

# Function to validate inputs
validate_inputs() {
    local parent_dir="$1"
    local ground_truth_dir="$2"
    local model_path="$3"
    local script_dir="$4"

    # Check if parent directory exists
    if [ ! -d "$parent_dir" ]; then
        log_error "Parent directory does not exist: $parent_dir"
        exit 1
    fi

    # Check if ground truth directory exists
    if [ ! -d "$ground_truth_dir" ]; then
        log_error "Ground truth directory does not exist: $ground_truth_dir"
        exit 1
    fi

    # Check if model file exists
    if [ ! -f "$model_path" ]; then
        log_error "Model file does not exist: $model_path"
        exit 1
    fi

    # Check if script directory exists
    if [ ! -d "$script_dir" ]; then
        log_error "Script directory does not exist: $script_dir"
        exit 1
    fi

    log_info "Validation passed"
}

# Function to preview what will be processed
preview_processing() {
    local parent_dir="$1"

    log_info "Preview of subdirectories that will be processed:"
    local count=0

    while IFS= read -r -d '' subdir; do
        local subdir_name=$(basename "$subdir")
        local dicom_count=$(count_dicom_files "$subdir")

        if [ "$dicom_count" -gt 0 ]; then
            count=$((count + 1))
            echo "  $count. $subdir_name ($dicom_count DICOM files)"
        else
            echo "  - $subdir_name (0 DICOM files - will skip)"
        fi
    done < <(find "$parent_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

    if [ $count -eq 0 ]; then
        log_warning "No subdirectories with DICOM files found!"
        exit 1
    fi

    echo
    log_info "Total subdirectories to process: $count"
}

# Main script execution
main() {
    # Parse arguments
    parse_arguments "$@"

    # Convert to absolute paths
    PARENT_DIR=$(cd "$PARENT_DIR" && pwd)
    GROUND_TRUTH_DIR=$(cd "$GROUND_TRUTH_DIR" && pwd)
    OVERVIEW_DIR=$(mkdir -p "$OVERVIEW_DIR" && cd "$OVERVIEW_DIR" && pwd)
    MODEL_PATH=$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")
    SCRIPT_DIR=$(realpath "$SCRIPT_DIR")

    log_info "=== BATCH MODEL EVALUATION SCRIPT ==="

    # Validate inputs
    validate_inputs "$PARENT_DIR" "$GROUND_TRUTH_DIR" "$MODEL_PATH" "$SCRIPT_DIR"

    # Show preview
    preview_processing "$PARENT_DIR"

    # Ask for confirmation
    echo
    read -p "Do you want to proceed with evaluation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Evaluation cancelled by user"
        exit 0
    fi

    # Start processing
    local start_time=$(date +%s)

    if process_batch "$PARENT_DIR" "$GROUND_TRUTH_DIR" "$OVERVIEW_DIR" "$MODEL_PATH" "$DEVICE" "$SCRIPT_DIR"; then
        local end_time=$(date +%s)
        local total_duration=$((end_time - start_time))
        local hours=$((total_duration / 3600))
        local minutes=$(((total_duration % 3600) / 60))
        local seconds=$((total_duration % 60))

        log_success "All evaluations completed successfully in ${hours}h ${minutes}m ${seconds}s"
        exit 0
    else
        local end_time=$(date +%s)
        local total_duration=$((end_time - start_time))
        local hours=$((total_duration / 3600))
        local minutes=$(((total_duration % 3600) / 60))
        local seconds=$((total_duration % 60))

        log_error "Some subdirectories failed to process (${hours}h ${minutes}m ${seconds}s total)"
        exit 1
    fi
}

# Run the main function with all arguments
main "$@"